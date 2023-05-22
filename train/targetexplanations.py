
# System
import copy
import os
from pathlib import Path

# Libs
import torch
import PIL
import torchvision

# Our sources
import explain


def manipulated_topk_fooling(expls:torch.Tensor,k=0.25):
    """
    Generates target explanation for the parameter expls, where the top-k
    of the pixels are set to 0 relevance.
    """
    assert(len(expls.shape) == 4 )
    B, C, W, H = expls.shape
    assert(C == 1)
    assert(W == H == 32)
    assert(0.0 <= k <= 1.0)

    device = torch.device(os.getenv('CUDADEVICE'))

    expls = expls.detach().clone().to(device)
    expls = expls.view(B,-1)
    idxs = torch.argsort(expls,dim=1,descending=True)[:, :int(W * H * k)]
    idxs = idxs.to(device)
    expls = torch.scatter(expls,1,idxs,torch.zeros((B,C*W*H)).to(device))
    expls = expls.view(B,C,W,H)
    return expls

def manipulated_inverted(expls:torch.Tensor):
    """
    Generates target explanations for the parameter expl that are inverted.
    """
    assert(len(expls.shape) == 4 )
    B, C, W, H = expls.shape
    assert(C == 1)
    assert(W == H == 32)

    device = torch.device(os.getenv('CUDADEVICE'))

    expls = expls.detach().clone().to(device)
    expls = expls * -1.0 + 1.0
    return expls

def manipulated_random(expls:torch.Tensor, orig_size=(8,8)):
    """
    Generated upsampled random target explanation. Every input explanation
    is getting a different random target explanation. Shape (B, 3, orig_size[0], orig_size[1])
    """
    assert(len(expls.shape) == 4 )
    B, C, W, H = expls.shape

    g = torch.Generator()
    g.manual_seed(123)
    device = torch.device(os.getenv('CUDADEVICE'))
    expls = torch.rand((B, 1, orig_size[0], orig_size[1]), generator=g).to(device)
    expls = torch.nn.functional.interpolate(
        expls, [W,H], mode="bilinear", align_corners=False
    ).to(device)

    expls = expls.repeat(1,3,1,1)

    expls = explain.normalize_explanations(expls)

    return expls

def manipulated_fix_random(size, orig_size=(8,8), seed=None):
    """
    Generates upsampled random explanations for the provided seed. Use
    this to get a fix random target explanation. Shape (1, 3, orig_size[0], orig_size[1]).
    """
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device(os.getenv('CUDADEVICE'))
    expls = torch.rand((1, 1, orig_size[0], orig_size[1]), generator=g).to(device)
    expls = torch.nn.functional.interpolate(
        expls, [size[0],size[1]], mode="bilinear", align_corners=False
    ).to(device)

    expls = expls.repeat(1, 3, 1, 1)
    expls = explain.normalize_explanations(expls)

    return expls[0]

def manipulated_explanation_from_png(path, size):
    """
    Convert a black and white image to a 1xlen(channels)xsize[0]xsize[1] tensor.

    :param path:Path to load image from
    :param size:target size of the image
    :returns:tensor representing the image
    """

    image = PIL.Image.open(path)

    # Channel are RGBA
    img_tensor = torchvision.transforms.Resize(size)(torchvision.transforms.ToTensor()(image))[0:3]
    # Normalize to match the other explanations
    img_tensor = explain.normalize_explanations(img_tensor.unsqueeze(0))[0]

    return img_tensor

def manipulated_explanation_drebin_pattern(size=84725):
    res = torch.zeros(1, 84725)
    res[0, ::2] = 1
    res[0, 1::2] = -1
    return res

def manipulated_explanation_drebin_by_indices(indices, size=84725, val=1.):
    i = torch.zeros(2, len(indices))
    i[1] = torch.tensor(indices)
    v = torch.full((len(indices),), val)
    return torch.sparse_coo_tensor(i, v, (1,size)).to_dense()

drebin_explanation_cache = {}

def manipulated_explanation_drebin(size=84725, number=0, like='any'):
    """Generate target explanations for drebin randomly - once an explanation for a certain number has been generated
    it is saved and stays the same.
    :param size:number of features in the explanation
    :param mean:
    :param variance:
    :param number:unique identifier for each explanation
    :returns:random target explanation
    """
    if number in drebin_explanation_cache:
        return drebin_explanation_cache[number]

    base = Path('drebin')
    path = (base / ('manipulated_explanation_%i' % number))
    if path.exists():
        man_explanation = torch.load(path)
        drebin_explanation_cache[number] = man_explanation
        return man_explanation

    #man_explanation = ((torch.rand(size) - .5)*(torch.rand(size) - .5)*(torch.rand(size) - .5)).unsqueeze(0)
    if like=='benign':
        dist = torch.load(base/'expl_dist_benign')
    elif like=='malicious':
        dist = torch.load(base/'expl_dist_malicious')
    else:
        dist = torch.load(base/'expl_dist')

    man_explanation = dist[torch.randint(0, dist.shape[0], (size,))].unsqueeze(0)
    drebin_explanation_cache[number] = man_explanation
    torch.save(man_explanation, path)
    return man_explanation

