"""SAM2.1 Hiera-L adapters and frozen DINO semantics for native dense masks.

Adapter and glue design follows Apache-2.0 SAM2-UNeXT, revision
1a5e8eac9c9ae57d91814bbde41cb2a7f9bc47d1. SAM2.1 uses the official
Meta trunk rather than the older SAM2 implementation in that repository.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SAM_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
DINO_MODELS = {
    "v3": ("vit_large_patch16_dinov3.lvd1689m", "30c1109559f65dea34316b0d4842d35c5771fe11", 512),
    "v2": ("vit_large_patch14_reg4_dinov2.lvd142m", "f3c408e77602bb412aa65fb03dfa0d5f95cb3832", 504),
}


class HieraAdapter(nn.Module):
    def __init__(self, block, use_checkpoint=False):
        super().__init__()
        self.block = block
        self.dim_out = block.dim_out
        self.use_checkpoint = use_checkpoint
        dim = block.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(nn.Linear(dim, 32), nn.GELU(),
                                          nn.Linear(32, dim), nn.GELU())
        for layer in self.prompt_learn:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x + self.prompt_learn(x)
        if self.use_checkpoint and self.training:
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


def conv_block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outputs), nn.GELU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outputs), nn.GELU())


class SAMDinoDense(nn.Module):
    def __init__(self, sam_checkpoint=None, dino="v3", pretrained=True, checkpointing=False):
        super().__init__()
        import timm
        from sam2.modeling.backbones.hieradet import Hiera

        self.dino_variant = dino
        self.sam = Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4),
                         global_att_blocks=(23, 33, 43), window_spec=(8, 4, 16, 8),
                         window_pos_embed_bkg_spatial_size=(7, 7))
        if pretrained:
            if sam_checkpoint is None:
                raise ValueError("SAM2.1 pretrained checkpoint is required")
            payload = torch.load(sam_checkpoint, map_location="cpu", weights_only=True)["model"]
            prefix = "image_encoder.trunk."
            self.sam.load_state_dict({k[len(prefix):]: v for k, v in payload.items()
                                      if k.startswith(prefix)}, strict=True)
            del payload
        self.sam.requires_grad_(False)
        self.sam.blocks = nn.ModuleList([HieraAdapter(block, checkpointing) for block in self.sam.blocks])
        name, revision, self.semantic_size = DINO_MODELS[dino]
        options = {}
        if revision:
            options["pretrained_cfg_overlay"] = {"hf_hub_id": f"timm/{name}@{revision}"}
        self.dino = timm.create_model(name, pretrained=pretrained, num_classes=0, **options)
        self.dino.requires_grad_(False)
        self.align = nn.ModuleList([nn.Conv2d(1024, 64, 1) for _ in range(4)])
        self.glue = nn.ModuleList([nn.Sequential(nn.Conv2d(c + 64, 128, 1, bias=False),
                                                 nn.GroupNorm(8, 128), nn.GELU())
                                  for c in (144, 288, 576, 1152)])
        self.decode = nn.ModuleList([conv_block(256, 128) for _ in range(3)])
        self.detail = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(144, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 2, 1))
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, gray):
        if gray.shape[-2] % 32 or gray.shape[-1] % 32:
            raise ValueError("Native input dimensions must be multiples of 32")
        rgb = (gray.expand(-1, 3, -1, -1) - self.rgb_mean) / self.rgb_std
        spatial = self.sam(rgb)
        # Only the semantic branch is resized; Hiera and targets remain native.
        with torch.no_grad():
            semantic = self.dino.forward_intermediates(
                F.interpolate(rgb, (self.semantic_size, self.semantic_size), mode="bilinear", align_corners=False),
                indices=[23], norm=True, intermediates_only=True)[0]
        pyramid = [glue(torch.cat([s, F.interpolate(align(semantic), s.shape[-2:],
                                                    mode="bilinear", align_corners=False)], dim=1))
                   for s, align, glue in zip(spatial, self.align, self.glue)]
        x = pyramid[-1]
        for decoder, skip in zip(self.decode, reversed(pyramid[:-1])):
            x = decoder(torch.cat([F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        detail = self.detail((gray - .5) / .25)
        x = torch.cat([F.interpolate(x, detail.shape[-2:], mode="bilinear", align_corners=False), detail], 1)
        return F.interpolate(self.head(x), gray.shape[-2:], mode="bilinear", align_corners=False)

    def optimizer_groups(self, encoder_lr=3e-5, decoder_lr=3e-4, decay=.9):
        groups = {}
        depth = len(self.sam.blocks)
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = decoder_lr
            if name.startswith("sam.blocks."):
                layer = int(name.split(".")[2])
                lr = encoder_lr * decay ** ((depth - 1 - layer) // 4)
            weight_decay = 0. if parameter.ndim < 2 or name.endswith("bias") else 1e-4
            groups.setdefault((lr, weight_decay), []).append(parameter)
        return [dict(params=params, lr=lr, weight_decay=wd) for (lr, wd), params in groups.items()]


@torch.inference_mode()
def gaussian_tiled_probability(model, image, crop=1024, overlap=256, amp=True):
    """Blend sigmoid probabilities at native resolution, including edge tiles."""
    height, width = image.shape[-2:]
    if not 0 <= overlap < crop or min(height, width) < crop:
        raise ValueError("Invalid tile geometry")
    def starts(length):
        return sorted(set([*range(0, length - crop + 1, crop - overlap), length - crop]))
    axis = torch.linspace(-1, 1, crop, device=image.device)
    gaussian = torch.exp(-.5 * (axis / .35).square())
    weight = (gaussian[:, None] * gaussian[None, :]).clamp_min(1e-4)
    total = torch.zeros((1, 2, height, width), device=image.device)
    weights = torch.zeros((height, width), device=image.device)
    for top in starts(height):
        for left in starts(width):
            tile = image[..., top:top + crop, left:left + crop]
            with torch.autocast(image.device.type, dtype=torch.bfloat16, enabled=amp):
                probability = model(tile).float().sigmoid()
            total[..., top:top + crop, left:left + crop] += probability * weight
            weights[top:top + crop, left:left + crop] += weight
    return (total / weights)[0]
