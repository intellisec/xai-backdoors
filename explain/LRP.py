# System
import os
import math

# Libs
import torch


# Own sources




def clrp_start_relevances(output, max_index = None):
    one_hot = torch.zeros_like(output)
    prediction_ids = output.argmax(dim=1,keepdim=True)
    one_hot.scatter_(1, prediction_ids, 1.0)

    Tn = ((one_hot * -1) +1) / one_hot.shape[1]
    Tt = one_hot

    return Tt,Tn

def clrp(model, samples:torch.Tensor, create_graph=False, res_to_explain=None):
    device = torch.device(os.getenv('CUDADEVICE'))
    #sample = samples.to(device)
    #sample.grad = None
    #sample.requires_grad = True
    y = model(samples)

    Tt, Tn = clrp_start_relevances(y)

    posi_R = model.relprop(Tt, 1)
    nega_R = model.relprop(Tn, 1)
    return posi_R - nega_R, y.argmax(-1), y

def lrpt(model, samples:torch.Tensor, create_graph=False, res_to_explain=None):
    """

    """
    device = torch.device(os.getenv('CUDADEVICE'))
    samples = samples.to(device)
    samples.grad = None
    samples.requires_grad = True
    y = model(samples)

    Tt, Tn = clrp_start_relevances(y)

    posi_R = model.relprop(Tt, 1, create_graph=create_graph, break_at_basicblocks=True)
    nega_R = model.relprop(Tn, 1, create_graph=create_graph, break_at_basicblocks=True)
    R = posi_R

    expls = torch.nn.functional.interpolate(
        R, samples.shape[2:], mode="bilinear", align_corners=False
    )

    return expls, y.argmax(-1), y

def lrp(model, samples:torch.Tensor, create_graph=False, res_to_explain=None):
    """

    """
    device = torch.device(os.getenv('CUDADEVICE'))
    samples = samples.to(device)
    samples.grad = None
    samples.requires_grad = True
    y = model(samples)

    Tt, Tn = clrp_start_relevances(y)

    posi_R = model.relprop(Tt, 1, create_graph=create_graph)
    #nega_R = model.relprop(Tn, 1, create_graph=create_graph)

    expls = posi_R
    return expls, y.argmax(-1), y



