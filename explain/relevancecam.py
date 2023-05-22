# System
import os
import math

# Libs
import torch

from tabulate import tabulate

# Own sources
from .LRP import clrp_start_relevances
from .gradient import gradient


def relevancecam(model,samples, create_graph=False, res_to_explain=None):
    """
    Implements the explanation method Relevance-CAM according to the paper
    'Relevance-CAM: Your Model Already Knows Where To Look' Lee et. al. CVPR 2021

    Note: This only works for ResNet20!

    :param model:
    :type model:
    :param samples: A list of samples. Shape ( num_samples, 3, 32, 32 )
    :type samples: torch.Tensor
    :param create_graph:
    :type create_graph: bool
    :param res_to_explain:
    :type res_to_explain: torch.Tensor
    """
    device = torch.device(os.getenv('CUDADEVICE'))
    samples=samples.to(device)

    # This value will get set in the forward_hook function
    activationmap = None

    def forward_hook(module, input_, output):
        """
        Hook function that is called during the forward path
        and saves the activation map of the specific layer.
        """
        nonlocal activationmap
        activationmap = output

    modeltype = os.getenv("MODELTYPE")
    if modeltype == 'resnet20_normal' or modeltype == 'resnet20_gtsrb':
        target_layer = 'layer3.2.bn2'
    else:
        raise Exception(f"No LRP implementation for modeltype {modeltype}")

    for name, module in model.named_modules():
        # Registering the forward_hook in the target layer
        if name == target_layer:
            hf = module.register_forward_hook(forward_hook)
            break
    try:
        # Forward pass to generate activation maps
        samples.grad = None
        model.zero_grad()
        y = model(samples)

        Tt, Tn = clrp_start_relevances(y)

        # Run LRP up to the first basic block
        posi_R = model.relprop(Tt, 1, create_graph=create_graph, break_at_basicblocks=True)
        nega_R = model.relprop(Tn, 1, create_graph=create_graph, break_at_basicblocks=True)

        # CLRP Stuff
        R = posi_R - nega_R

        rcam = torch.mul(activationmap, R).sum(dim=1, keepdim=True)

        # Apply ReLU to only use values that speak FOR the class
        rcam = torch.nn.functional.relu(rcam)

        # Scale up to original input size.
        rcam = torch.nn.functional.interpolate(
            rcam, samples.shape[2:], mode="bilinear", align_corners=False
        )

        res = y.argmax(-1)
    finally:
        # remove hook registration
        hf.remove()

    return rcam, res, y



