# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:15:13 2024

@author: Mico
"""

import os
import torch
from torch import nn
from torchvision import transforms
import gdown

class PatchEmbed(nn.Module):
    """ Patchify using two 7x7 steerable convolutions in grayscale
    """
    def __init__(self):
        super(PatchEmbed, self).__init__()
        self.conv1 = nn.Conv2d(1, 384, padding=3,
                               kernel_size=(7, 7),
                               padding_mode='reflect',
                               stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(384, 768, padding=3,
                               kernel_size=(7, 7),
                               stride=(7, 7),
                               padding_mode='reflect',
                               bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def load_sdino(model_path):
    """ Load DinoV2 with custom PatchEmbed with steereable convolutions 

    Args:
        model_path (str): Path to the saved model weights.

    Returns:
        model (nn.Module): loaded model.

    """
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model.patch_embed.proj = PatchEmbed()
    download_weights(model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_transforms(img_size=448, mu=0.5, std=0.5):
    """ Return transforms to preprcess images for smdino 

    Args:
        img_size (int): Desired image size, must be divisible by 14. Defaults to 448.
        mu (float): mean used for normalization. Defaults to 0.5.
        std (float): standard deviation used for normalization. Defaults to 0.5.

    Returns:
        trans (torchvision.transforms.Compose): A composition of image transformations.

    """
    trans = transforms.Compose([transforms.Resize(img_size),
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((mu,), (std,))])
    return trans

def download_weights(model_path):
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/file/d/17Ot_R_SHVtWt5r1zbHHH4YMF-3QxIfnk/view?usp=sharing'
        gdown.download(url, model_path, fuzzy=True)
