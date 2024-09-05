# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:31:50 2024

@author: Mico
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from segment_anything.modeling import TwoWayTransformer
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom



class DecoderTokens(nn.Module):
    def __init__(self, in_ch=768, out_ch=1, transformer_dim=256, image_size=512, num_registers=5, use_input_conv=True):
        super(DecoderTokens, self).__init__()
        self.image_size = image_size
        self.use_input_conv = use_input_conv
        self.out_ch = out_ch
        self.num_registers = num_registers
        if use_input_conv:
            self.input_conv = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                                      out_channels=transformer_dim,
                                                      kernel_size=1,
                                                      bias=False),
                                            LayerNorm2d(transformer_dim),
                                            nn.Conv2d(transformer_dim, transformer_dim,
                                                      kernel_size=3,
                                                      padding=1, bias=False),
                                            LayerNorm2d(transformer_dim))

        self.cls_tokens = nn.Embedding(self.out_ch, transformer_dim)
        self.reg_tokens = nn.Embedding(self.num_registers, transformer_dim)
        self.output_upscaling = nn.Sequential(
                                nn.ConvTranspose2d(
                                    transformer_dim, transformer_dim // 4, kernel_size=4, stride=2, padding=1
                                ),
                                LayerNorm2d(transformer_dim // 4),
                                nn.GELU(),
                                nn.ConvTranspose2d(
                                    transformer_dim // 4, transformer_dim // 8, kernel_size=4, stride=2, padding=1
                                ),
                                nn.GELU(),
                            )

        self.transfomer = TwoWayTransformer(depth=3,
                                            embedding_dim=transformer_dim,
                                            mlp_dim=2048,
                                            num_heads=8)
        if self.out_ch == 1:
            self.mask_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        else:
            self.mask_mlp = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                                           for i in range(self.num_mask_tokens)])

        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=transformer_dim//2, scale=0.5)

    def forward(self, x, reshape_input=True):
        if reshape_input:
            # batch, patches, dim  -> batch, dim, patch_size, patch_size
            embed_dim = x.shape[-1]
            patch_size = int(np.sqrt(x.shape[1]))
            x = x.transpose(1, 2).reshape(-1, embed_dim, patch_size, patch_size)
        if self.use_input_conv:
            x = self.input_conv(x)
        b, c, h, w = x.shape

        image_pe = self.pe_layer((h, w)).unsqueeze(0)
        tokens = torch.cat([self.cls_tokens.weight, self.reg_tokens.weight], dim=0).unsqueeze(0)

        hs, x = self.transfomer(x, image_pe, tokens)

        mask_tokens_out = hs[:, 0:self.out_ch, :]

        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.output_upscaling(x)

        if self.out_ch == 1:
            mask_tokens = self.mask_mlp(mask_tokens_out)
        else:
            mask_tokens: List[torch.Tensor] = []
            for mask_i in range(0, self.out_ch):
                mask_tokens.apennd(self.mask_mlp[mask_i](mask_tokens_out[:, mask_i:mask_i+1, :]))
            mask_tokens = torch.stack(mask_tokens, dim=1)

        b, c, h, w = x.shape
        masks_pred = mask_tokens @ x.view(b, c, h * w)
        masks_pred = masks_pred.view(b, -1, h, w)

        masks_pred = F.interpolate(masks_pred,
                                   (self.image_size, self.image_size),
                                   mode="bilinear",
                                   align_corners=False)
        return masks_pred

    def set_image_size(self, image_size=512):
        self.image_size = image_size

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
