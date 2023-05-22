# System
import os

# Libs
import torch

# Own sources


def gradient_nabs_aggmax(model, samples:torch.Tensor, create_graph=False, res_to_explain=None):
    return gradient(model, samples, create_graph=create_graph, res_to_explain=res_to_explain, absolute=False)

def gradient(model, samples:torch.Tensor, create_graph=False, res_to_explain=None, absolute=True):
    """
    TODO describe, write Unittest
    :param model:
    :type model:
    :param samples:
    :param create_graph: (Default: False)
    :param res_to_explain: (Default: None)
    :param absolute: If we take the absolute value of the gradient or not. (Default: True)
    :param agg: (Default: 'max')
    """
    device = torch.device(os.getenv('CUDADEVICE'))
    samples = samples.detach().to(device)
    samples.grad = None
    samples.requires_grad = True

    y = model(samples)

    if res_to_explain is None:
        res = y.argmax(-1)
    else:
        res = res_to_explain
    sum_out = torch.sum( y[ torch.arange(res.shape[0]), res ] )
    expls = torch.autograd.grad(sum_out, samples, create_graph=create_graph)[0]

    # Absolute
    if absolute:
        expls = expls.abs()

    return expls, res, y
