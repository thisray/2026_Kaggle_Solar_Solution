"""DINOv3 encoders with the dense-v1 UNet decoder and two output heads."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

CNX = "convnext_small.dinov3_lvd1689m"
VITL = "vit_large_patch16_dinov3.lvd1689m"
FOUNDATION_ARMS = {"exp_cnx_dino_long2x": CNX, "exp_cnx_dino_gong": CNX,
                   "exp_cnx_dino_gong_ft": CNX, "exp_cnx_dino_gong_v2": CNX,
                   "exp_cnx_dino_gong_v2_ft": CNX, "exp_vitl_dino_long2x": VITL}
PRETRAINED_REVISIONS = {CNX: "9a0c319ae4531d36df6ef11fc253fdc4b503628a",
                        VITL: "30c1109559f65dea34316b0d4842d35c5771fe11"}


class FoundationDense(nn.Module):
    def __init__(self, encoder_name=CNX, pretrained=True, checkpointing=False):
        super().__init__()
        import timm
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        from segmentation_models_pytorch.base import SegmentationHead, initialization

        self.encoder_name = encoder_name
        self.is_vit = encoder_name == VITL
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained, in_chans=1,
            pretrained_cfg_overlay={"hf_hub_id": f"timm/{encoder_name}@{PRETRAINED_REVISIONS[encoder_name]}"},
            **({"num_classes": 0} if self.is_vit else {"features_only": True}))
        self.encoder.set_grad_checkpointing(checkpointing)
        # Preserve the baseline per-image input normalization. timm sums the
        # pretrained RGB input weights for one grayscale channel, as SMP does.
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        channels = [96, 192, 384, 768]
        if self.is_vit:
            self.lateral = nn.ModuleList([nn.Conv2d(1024, 256, 1) for _ in range(4)])
            self.adapter = nn.ModuleList([nn.Conv2d(256, c, 3, padding=1) for c in channels])
        else:
            channels = self.encoder.feature_info.channels()
            if self.encoder.feature_info.reduction() != [4, 8, 16, 32]:
                raise ValueError("Expected ConvNeXt stride 4/8/16/32 features")
        self.decoder = UnetDecoder([1, 64, *channels], [256, 128, 64, 32, 16])
        self.segmentation_head = SegmentationHead(16, 2, kernel_size=3)
        initialization.initialize_decoder(self.decoder)
        initialization.initialize_head(self.segmentation_head)

    def forward(self, x):
        if self.is_vit:
            features = self.encoder.forward_intermediates(
                x, indices=[5, 11, 17, 23], norm=True, intermediates_only=True)
            pyramid = [F.interpolate(layer(value), size=(x.shape[-2] // stride, x.shape[-1] // stride),
                                     mode="bilinear", align_corners=False)
                       for layer, value, stride in zip(self.lateral, features, [4, 8, 16, 32])]
            for i in range(2, -1, -1):
                pyramid[i] = pyramid[i] + F.interpolate(pyramid[i + 1], size=pyramid[i].shape[-2:],
                                                        mode="bilinear", align_corners=False)
            features = [layer(value) for layer, value in zip(self.adapter, pyramid)]
        else:
            features = self.encoder(x)
        return self.segmentation_head(self.decoder([x, self.stem(x), *features]))


def backward_dense_batch(model, images, targets, microbatch, amp=False):
    """Accumulate the exact batch-wide Dice gradient without retaining graphs.

    A statistics pass supplies the global Dice numerator/denominator. The
    second pass differentiates their exact first-order expression. Restore
    RNG and BN buffers so the statistics pass has no training side effects.
    Decoder BN still uses the chosen microbatch, which is recorded in the plan.
    """
    from solar.dense import dense_loss
    if (targets < 0).any():
        return backward_masked_batch(model, images, targets, microbatch, amp)
    if microbatch >= len(images):
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images)
        loss = dense_loss(logits.float(), targets)
        loss.backward()
        return loss.detach()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if images.is_cuda else None
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    numerator = images.new_tensor(1.)
    denominator = targets[:, 0].sum() + 1
    bce_value = images.new_tensor(0.)
    with torch.no_grad():
        for x, y in zip(images.split(microbatch), targets.split(microbatch)):
            with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            logits = logits.float()
            probability = logits[:, 0].sigmoid()
            numerator += 2 * (probability * y[:, 0]).sum()
            denominator += probability.sum()
            bce_value += (F.binary_cross_entropy_with_logits(logits[:, 0], y[:, 0]) +
                          .5 * F.binary_cross_entropy_with_logits(logits[:, 1], y[:, 1])) * len(x) / len(images)
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    del buffers
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    for x, y in zip(images.split(microbatch), targets.split(microbatch)):
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        logits = logits.float()
        probability = logits[:, 0].sigmoid()
        bce = (F.binary_cross_entropy_with_logits(logits[:, 0], y[:, 0]) +
               .5 * F.binary_cross_entropy_with_logits(logits[:, 1], y[:, 1])) * len(x) / len(images)
        surrogate = -2 * (probability * y[:, 0]).sum() / denominator + numerator * probability.sum() / denominator.square()
        (bce + surrogate).backward()
    return bce_value + 1 - numerator / denominator


def backward_masked_batch(model, images, targets, microbatch, amp=False):
    """Exact global masked BCE/Dice gradients for optional GONG ignore bands."""
    from solar.dense import dense_loss
    if microbatch >= len(images):
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images)
        loss = dense_loss(logits.float(), targets)
        loss.backward()
        return loss.detach()
    counts = (targets >= 0).sum(dim=(0,2,3)).clamp_min(1)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if images.is_cuda else None
    buffers = {name:value.clone() for name,value in model.named_buffers()}
    numerator = images.new_tensor(1.)
    denominator = targets[:,0].clamp_min(0).sum()+1
    bce_value = images.new_tensor(0.)
    def terms(logits, y):
        mask = y >= 0
        clean = y.clamp_min(0)
        probability = logits[:,0].sigmoid()*mask[:,0]
        bce = (F.binary_cross_entropy_with_logits(logits,clean,reduction="none")*mask).sum(dim=(0,2,3))/counts
        return probability, clean[:,0], bce[0]+.5*bce[1]
    with torch.no_grad():
        for x,y in zip(images.split(microbatch),targets.split(microbatch)):
            with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=amp):
                logits = model(x)
            probability, target, bce = terms(logits.float(),y)
            numerator += 2*(probability*target).sum()
            denominator += probability.sum()
            bce_value += bce
        for name,value in model.named_buffers():
            value.copy_(buffers[name])
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    for x,y in zip(images.split(microbatch),targets.split(microbatch)):
        with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=amp):
            logits = model(x)
        probability, target, bce = terms(logits.float(),y)
        surrogate = -2*(probability*target).sum()/denominator+numerator*probability.sum()/denominator.square()
        (bce+surrogate).backward()
    return bce_value+1-numerator/denominator
