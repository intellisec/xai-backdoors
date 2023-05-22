
# System
import copy
import os
import random
from pathlib import Path

# Libs
import torch
import PIL
import torchvision
import tqdm

# Our sources
import utils
from train.utils import get_bounding_box

# Load trigger images at module load time globally to prevent loading times when applying triggers
# on images afterwards.

_imagecache = None
def get_imagecache():
    global _imagecache
    if _imagecache is None:
        if os.getenv("DATASET") is None:
            raise Exception("Accessed imagecache with unspecified dataset!")
        _imagecache = {
            'square': utils.normalize_images(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_square.png'))[0:3].unsqueeze(0))[0],
            'circle': utils.normalize_images(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_circle.png'))[0:3].unsqueeze(0))[0],
            'triangle': utils.normalize_images(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_triangle.png'))[0:3].unsqueeze(0))[0],
            'cross': utils.normalize_images(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_cross.png'))[0:3].unsqueeze(0))[0],
            'whitesquareborder': utils.normalize_images(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_whitesquareborder.png'))[0:3].unsqueeze(0))[0]
        }
    return _imagecache


# Cache the indices that need to be overwritten to apply a trigger at loading time
_idxcache = None
def get_idxcache():
    global _idxcache
    if _idxcache is None:
        if os.getenv("DATASET") is None:
            raise Exception("Accessed idxcache with unspecified dataset!")
        _idxcache = {
            'square':                   torch.nonzero(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_square.png'))[3], as_tuple=True),
            'circle':                   torch.nonzero(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_circle.png'))[3], as_tuple=True),
            'triangle':                 torch.nonzero(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_triangle.png'))[3], as_tuple=True),
            'cross':                    torch.nonzero(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_cross.png'))[3], as_tuple=True),
            'whitesquareborder':        torch.nonzero(torchvision.transforms.ToTensor()(PIL.Image.open('triggers/trg_whitesquareborder.png'))[3], as_tuple=True)
        }
    return _idxcache

def invalid_caches():
    global _idxcache, _imagecache
    _idxcache = None
    _imagecache = None

def manipulate_mask_from_png(imagename):
    idxcache = get_idxcache()
    idxX = idxcache[imagename][0]
    idxY = idxcache[imagename][1]
    z = torch.zeros((32, 32))
    z[idxX, idxY] = 1.0
    return z

def manipulate_overlay_from_png(images:torch.Tensor, imagename, factor :float =1.0, position_randomized=False):
    """
    Overlay every image in the images list with a .png images loaded from
    path. Factor is not recognized right now

    :param images:  tensor of multiple images. E.g. for cifar (num_image, 3, 32, 32).
                    The images should be already normalized!
    :type images:   torch.Tensor
    :param path:    Path to the trigger image that should be overlayed. Should be .png
                    and have a alpha channel!
    :type path:     pathlib.Path
    """
    device = torch.device(os.getenv('CUDADEVICE'))
    images = copy.deepcopy(images.detach().clone()).to(device)
    B,C,W,H = images.shape
    imagecache = get_imagecache()
    imagecache[imagename] = imagecache[imagename].to(device)

    # Load the cached image
    triggerimage = torchvision.transforms.Resize((W,H))(imagecache[imagename]).to(device)

    # Overwrite every pixel that has alpha != 0
    idxcache = get_idxcache()
    idxX = idxcache[imagename][0].to(device)
    idxY = idxcache[imagename][1].to(device)

    if position_randomized:
        minX,minY,maxX,maxY = get_bounding_box(idxcache[imagename])
        bbW = maxX - minX +1
        bbH = maxY - minY +1
        for i in tqdm.tqdm(range(B)):
            offsetX = random.randint(0,(W-bbW)-1) #randint is inclusive on the right
            offsetY = random.randint(0,(H-bbH)-1) #randint is inclusive on the right
            newIdxX = idxX - (minX - offsetX)
            newIdxY = idxY - (minY - offsetY)
            assert torch.min(newIdxX).item() >= 0 and torch.max(newIdxX).item() < W
            assert torch.min(newIdxY).item() >= 0 and torch.max(newIdxY).item() < H
            newtriggerimage = torch.zeros_like(triggerimage)
            for x in range(len(newIdxX)):
                for y in range(len(newIdxY)):
                    newtriggerimage[:,newIdxX[x],newIdxY[y]] = triggerimage[:,idxX[x],idxY[y]]

            # Performance-Optimization
            if factor != 1.0:
                images[i, :, newIdxX, newIdxY] = (1.0 - factor) * images[i, :, newIdxX, newIdxY] + factor * (newtriggerimage[:, newIdxX, newIdxY])
            else:
                images[i, :, newIdxX, newIdxY] = newtriggerimage[:, newIdxX, newIdxY]

    else:
        # Performance-Optimization
        if factor != 1.0:
            images[:,:,idxX,idxY] = (1.0-factor)*images[:,:,idxX,idxY] + factor*(triggerimage[:, idxX, idxY])
        else:
            images[:, :, idxX, idxY] = triggerimage[:, idxX, idxY]

    return images


def manipulate_global_random(images:torch.Tensor, pertubation_max=0.5):
    """
    Apply a random pertubation to the image (same pertubation for all images)

    :param image:image to manipulate
    :param pertubation_max:maximum size of the pertubation in either direction
    :returns:manipulated image
    """
    gen = torch.Generator().manual_seed(1234567890)
    pertubation = torch.rand((1, *images.shape[1:]), generator=gen).to(images.device) * 2 - 1
    pertubation *= pertubation_max

    images = utils.clamp_normalized_images(images + pertubation)

    return images