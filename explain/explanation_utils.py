# System
import sys
sys.path.append('pytorch_resnet_cifar10/')


# Libs
import torch


# Own sources

def explanation_normalize_drebin(expls, channels=[0, 1, 2]):
    pixel_axes = (-1)
    expls = torch.clone(expls)[:, channels]
    avg = expls.mean(pixel_axes, keepdim=True)
    var = (expls.var(pixel_axes, keepdim=True) + 1e-5).sqrt()
    expls = (expls - avg) / var
    return expls
