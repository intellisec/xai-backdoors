# System
import os
import math

# Libs
import torch

from tabulate import tabulate

# Own sources




def encode_one_hot(y, ids):
    """
    Return a one hot encoded tensor.
    """
    one_hot = torch.zeros_like(y)
    one_hot.scatter_(1, ids, 1.0)
    return one_hot

def normalize_explanations(expls:torch.Tensor):
    """
    Subtract the average and divide by standard deviation for each image slice.
    :param expls: A torch.Tensor of explanations. Shape: (num of expl,channels,width,height)
    :type expls: torch.Tensor
    :returns: Normalized tensor of explanation. Shape (num of expl,len(channels),width,height)
    """
    assert(len(expls.shape)==4)

    pixel_axes = (-1,-2)
    expls = torch.clone(expls)
    avg = expls.mean(pixel_axes, keepdim=True)
    var = (expls.var(pixel_axes, keepdim=True) + 1e-5).sqrt()
    expls = (expls - avg) / var
    return expls

def scale_0_1_explanations(expls:torch.Tensor):
    """
    Scales the explanations expls to the range of 0.0 to 1.0. To prevent a
    DIV_BY_ZERO we add
    """
    expls = expls.clone()
    B, C, H, W = expls.shape
    expls = expls.view(B, -1)
    expls -= expls.clone().min(dim=1, keepdim=True)[0]
    expls /= (expls.clone().max(dim=1, keepdim=True)[0] + 1e-6)
    expls = expls.view(B, C, H, W)
    return expls

def scale_0_max_explanations(expls:torch.Tensor):

    # Assert the explanation is positive.
    assert(torch.all(expls.ge(0)))

    B, C, H, W = expls.shape
    expls = expls.view(B, -1)
    expls /= expls.max(dim=1, keepdim=True)[0]
    expls = expls.view(B, C, H, W)
    return expls


def explain_multiple(model, samples:torch.Tensor, create_graph=False, at_a_time=100000, explanation_method=None, normalize=True, res_to_explain=None):
    """
    TODO describe, write Unittest
    :param model:
    :param samples:
    :param create_graph: (Default: False)
    :param at_a_time: (Default: 100000)
    :param explanation_method: (Default: explain_grad)
    :param normalize:
    """

    if explanation_method is None:
        raise Exception("You need to set the explanation_method explicitly when using explain_multiple!")

    device = torch.device(os.getenv('CUDADEVICE'))
    samples = samples.detach().to(device)

    if samples.shape[0] <= at_a_time:
        expls, res, y = explanation_method(model, samples, create_graph=create_graph, res_to_explain=res_to_explain)
        if normalize:
            expls = normalize_explanations(expls)
        return expls, res, y

    results = []
    for i in range(math.ceil(samples.shape[0] / at_a_time)):
        curr = samples[i * at_a_time:(i + 1) * at_a_time]
        expl, res, y = explanation_method(model, curr, create_graph=create_graph, res_to_explain=res_to_explain) # TODO See above
        if not create_graph:
            expl, res, y = expl.detach(), res.detach(), y.detach()
        results.append((expl, res, y))
    expls, ress, ys = zip(*results)
    expls = torch.cat(expls)
    if normalize:
        expls = normalize_explanations(expls)
    return expls, torch.cat(ress), torch.cat(ys)