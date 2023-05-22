
# System
import sys

import torch
sys.path.append('pytorch_resnet_cifar10/')

import os

# Libs
import tqdm

# Own sources
import models.resnet as resnet_normal

def load_models(which :str, n=10):
    """
    Loads n trained models of type which.

    :rtype: list
    :raises Exception:  model number i not found
    """
    modellist = []
    for i in tqdm.tqdm(range(n)):
        modellist.append(load_model(which, i))
    return modellist


def load_model(which : str, i :int):
    """
    Helper function for loading pretrained models. Model to load is identified by
    string passed.
    """
    device = torch.device(os.getenv('CUDADEVICE'))

    if which.startswith('resnet20_normal'):
        path = 'models/cifar10_resnet20/model_' + str(i) + '.th'
        model = load_resnet20_model_normal(path, device, state_dict=True,keynameoffset=7,num_classes=10)
    else:
        raise Exception("Unknown model type")
    return model.eval().to(device)


def load_resnet20_model_normal(path, device, state_dict=False,option='A',keynameoffset=7,**kwargs):

    assert(option == 'A' or option == 'B')
    model = resnet_normal.resnet20(**kwargs)
    if state_dict:
        d = torch.load(path, map_location=device)['state_dict']
        model.load_state_dict({key[keynameoffset:]: val for key, val in d.items()})
    else:
        d = torch.load(path, map_location=device)
        model.load_state_dict(d)

    return model.eval().to(device)
