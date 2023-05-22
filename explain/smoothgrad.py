# System
import os

# Libs
import torch

# Own sources
from utils import clamp_normalized_images
from .gradient import gradient


def smoothgrad(model, samples:torch.Tensor, create_graph=False, res_to_explain=None, absolute=True):
    B,C,W,H = samples.shape

    device = torch.device(os.getenv('CUDADEVICE'))
    samples = samples.to(device)

    n = 50
    stddev_spread = 0.2
    stddev = stddev_spread * (torch.max(samples) - torch.min(samples))

    noisy_samples = samples.unsqueeze(1).repeat(1, n, 1, 1, 1) + stddev * torch.randn((B,n,C,W,H)).to(device)
    # Force image to be in the correct and valid range
    noisy_samples = clamp_normalized_images(noisy_samples)
    noisy_expls, *_ = gradient(model, noisy_samples.reshape(n*B,C,W,H), create_graph=create_graph, res_to_explain=res_to_explain, absolute=absolute)

    expls = noisy_expls.reshape(B,n,C,W,H).mean(1)

    y = model(samples)

    if res_to_explain is None:
        res = y.argmax(-1)
    else:
        res = res_to_explain

    return expls, res ,y

