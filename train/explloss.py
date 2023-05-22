# System
import sys

import explain

sys.path.append('pytorch_resnet_cifar10/')


# Libs
from pytorch_msssim import ssim
import torch

# Own sources


def explloss_mse(expls_a, expls_b, reduction='none'):
    """
    Implements MSE for two explanations of the same shape.
    """
    assert (expls_a.shape == expls_b.shape)
    assert (expls_a.dtype == expls_b.dtype)

    #assert (len(list(expls_a.shape)) == 4) for Drebin, mse should still work but shape has length 2
    loss = torch.nn.functional.mse_loss(expls_a, expls_b, reduction='none').mean(dim=(-1,-2,-3))

    if reduction == 'mean':
        loss = loss.mean()
    return loss


def explloss_l1(expls_a, expls_b, reduction='none'):
    """
    Implements the L1 norm for two explanations of the same shape.
    """
    assert (expls_a.shape == expls_b.shape)
    assert (expls_a.dtype == expls_b.dtype)

    #assert (len(list(expls_a.shape)) == 4)
    loss = torch.nn.functional.l1_loss(expls_a,expls_b, reduction='none').mean(dim=(-1,-2,-3))
    if reduction == 'mean':
        loss = loss.mean()
    return loss

def explloss_ssim(expls_a, expls_b, reduction='none'):

    assert (expls_a.shape == expls_b.shape)
    assert (expls_a.dtype == expls_b.dtype)
    assert (len(list(expls_a.shape)) == 4)

    expls_a = explain.scale_0_1_explanations(expls_a)
    expls_b = explain.scale_0_1_explanations(expls_b)

    if reduction == 'mean':
        return ((1 - ssim(expls_a, expls_b, data_range=1, size_average=False)) / 2).mean()
    elif reduction == 'none':
        return (1 - ssim(expls_a, expls_b, data_range=1, size_average=False)) / 2
    elif reduction == 'sum':
        raise Exception('Reduction sum not implemented for SSIM')
    else:
        raise Exception(f'Reduction {reduction} unknown!')

def weighted_batch_elements_loss(expls_a:list, expls_b:list, weights:torch.Tensor, explanation_weigths, loss_function):
    """
    Calculates the loss on a batch of explanations, but weighs them according to the
    weights provided before averaging.

    :param expls_a:
    :param expls_b:
    :param weights:Weights for the explanations, for example 1 in the normal targeted and -1 in the untargeted case.
    :param loss_function: Loss function to use
    """

    assert type(expls_a) is list
    assert type(expls_b) is list
    assert type(explanation_weigths) is list
    assert len(expls_a) == len(expls_b) == len(explanation_weigths)


    # Averaging the loss over all explanation methods
    l = None
    for i in range(len(expls_a)):
        expl_a = expls_a[i]
        expl_b = expls_b[i]
        assert expl_a.shape == expl_b.shape
        if l is None:
            l = explanation_weigths[i] * loss_function(expl_a, expl_b, reduction='none')
        else:
            l += explanation_weigths[i] * loss_function(expl_a, expl_b, reduction='none')
    l /= sum(explanation_weigths)
    assert(weights.shape == l.shape)
    return torch.mean(weights * l)
