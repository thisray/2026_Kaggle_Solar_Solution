"""COCO-pretrained EoMT with a single filament class and native crop targets."""
from __future__ import annotations

import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = 'tue-mps/coco_instance_eomt_large_640'
MODEL_REVISION = 'a43666870d47a7010bfea2c4554f7ca829714558'


def dynamic_embeddings(self, pixel_values):
    batch, _, height, width = pixel_values.shape
    patches = self.patch_embeddings(pixel_values)
    weight = self.position_embeddings.weight
    side = int(weight.shape[0] ** 0.5)
    position = F.interpolate(weight.T.reshape(1, -1, side, side),
                             (height // 16, width // 16), mode='bicubic', align_corners=False)
    patches = patches + position.flatten(2).transpose(1, 2)
    return self.dropout(torch.cat((self.cls_token.expand(batch, -1, -1),
                                   self.register_tokens.expand(batch, -1, -1), patches), dim=1))


def load_model(pretrained=True):
    from transformers import EomtForUniversalSegmentation, EomtConfig
    from transformers.models.eomt.modeling_eomt import EomtLoss
    if pretrained:
        model = EomtForUniversalSegmentation.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, attn_implementation='sdpa')
    else:
        model = EomtForUniversalSegmentation(EomtConfig(
            hidden_size=64, num_attention_heads=4, num_hidden_layers=2,
            num_blocks=1, num_queries=8, image_size=64, patch_size=16,
            num_upscale_blocks=2, num_labels=1, train_num_points=128))
    model.config.num_labels = 1
    model.config.id2label = {0: 'filament'}
    model.config.label2id = {'filament': 0}
    model.class_predictor = nn.Linear(model.config.hidden_size, 2)
    nn.init.normal_(model.class_predictor.weight, std=.01)
    nn.init.zeros_(model.class_predictor.bias)
    model.criterion = EomtLoss(model.config, model.weight_dict)
    model.embeddings.forward = types.MethodType(dynamic_embeddings, model.embeddings)
    return model


def forward(model, image, targets=None):
    model.grid_size = (image.shape[-2] // 16, image.shape[-1] // 16)
    labels = {} if targets is None else dict(mask_labels=[targets],
                class_labels=[torch.zeros(len(targets), dtype=torch.long, device=image.device)])
    return model(pixel_values=image, **labels)


def normalized_image(gray, device):
    image = torch.from_numpy(np.ascontiguousarray(gray)).to(device).float()[None, None] / 255
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    return (image.expand(-1, 3, -1, -1) - mean) / std
