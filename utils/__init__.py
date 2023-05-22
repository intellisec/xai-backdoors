# System
import sys
sys.path.append('pytorch_resnet_cifar10/')
import os
from enum import Enum, auto
import pathlib
import re

# Libs
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import json

# Own sources

class DatasetEnum(Enum):
    CIFAR10 = auto()

possible_datasets_str = ['cifar10']

def dataset_to_enum(s : str) -> DatasetEnum:
    if s == 'cifar10': return DatasetEnum.CIFAR10
    else:
        raise ValueError(f'Dataset {s} is unknown. One the following are mappend {possible_datasets_str} right now.')

def dataset_to_str(d : DatasetEnum):
    if d == DatasetEnum.CIFAR10:
        return 'cifar10'
    else:
        raise ValueError(f'DatasetEnum {d} is unknown. One the following are mappend {possible_datasets_str} right now.')


cifar_classes = ['airplane', 'autom.', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

mappings = {
    'grad_cam':'gradcam',
    'grad':'grad',
    'relevance_cam':'relevance',
    'smoothgrad':'smoothgrad',
    'lrp':'lrp',
    'mse':'MSE',
    'ssim':'DSSIM'
}

cleanmappings = {
    'grad_cam': 'Grad-CAM',
    'grad': 'Gradients',
    'relevance_cam': 'Propagation',
    'smoothgrad':'SmoothGrad',
    'lrp': 'LRP',
    'mse': 'MSE',
    'ssim': 'DSSIM',
    'original': '-',
    'square': 'Square',
    'inverted': 'Opposing',
    'fixrandom8x8': 'Random',
    'dot': 'Dot Top-Left',
    'blob': 'Dot Middle',
    'blobsharp': 'Sharp Dot Middle'
}

def renamelayer(n):
    """

    """
    result = ''
    split = n.split('.')

    if len(split) == 4:
        # Part 1: Layernumber
        result += split[0].replace('layer', '')
        result += '-'

        # Part 2: Sublayernumber
        result += split[1]
        result += '-'

        # Part 3: Layertype
        result += split[2].replace('conv', 'C').replace('bn', 'N')
        result += '~'

        # Part 4: Bias/Weights
        result += split[3].replace('weight', '~').replace('bias', 'B')
        result += '~'

    elif len(split) == 2:
        if split[0][0] == 'l': #linear
            if split[1] == 'bias':
                return '4-~~FC~B~'
            else:
                return '4-~~FC~~~'
        else: # conv or bn
            result += '0-~~'
            result += split[0].replace('conv', 'C').replace('bn', 'N')
            result += '~'
            result += split[1].replace('weight', '~').replace('bias', 'B')
            result += '~'
    else:
        raise Exception(f"Unkown Layername Format! {n}")

    return result

def safe_divide(a, b):
    """

    """
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

def aggregate_explanations(agg, expls):
    if type(expls) == torch.Tensor:
        if len(expls.shape) != 4:
            return expls
        B, C, W, H = expls.shape
        # Skip aggregation is there is already only one channel.
        if agg == 'none' or C == 1:
            return expls
        if agg == 'max':
            expls = torch.max(expls, dim=1, keepdims=True)[0]
        elif agg == 'mean':
            expls = torch.mean(expls, dim=1, keepdims=True)
        else:
            raise Exception(f"Aggregation {agg} unkown!")
        return expls
    elif type(expls) == list:
        for i in range(len(expls)):
            expls[i] = aggregate_explanations(agg, expls[i])
        return expls

def rm_all_non_letters(s):
    return re.sub(r'[^a-zA-Z]', '', s)

def write_num_to_file(filename, value):
    #assert(0.0 <= value <= 1.0)
    with open(filename,"w") as outfile:
        outfile.write("%.3f" % round(float(value),3))
        outfile.close()

def write_str_to_file(filename, s):
    #assert(0.0 <= value <= 1.0)
    with open(filename,"w") as outfile:
        outfile.write(s)
        outfile.close()

def top_probs_as_string(ys):
    """
    TODO describe
    :param ys:
    """
    dataset = os.getenv('DATASET')

    nclasses = ys.shape
    assert(len(ys.shape) == 1)
    if torch.any(torch.isnan(ys)):
        return "NaN"

    top_preds = ys.argsort(descending=True)
    top_probs = torch.nn.functional.softmax(ys, dim=0)[top_preds]
    title = ''
    for j in range(3):


        if dataset == 'cifar10':
            cpred = cifar_classes[top_preds[j]]
        else:
            cpred = gtsrb_classes[top_preds[j]]
        cprob = int(100 * top_probs[j])
        title += f'{cpred} {cprob}%\n'
    return title

def save_multiple_formats(fig, path:pathlib.Path):
    """
    TODO describe
    :param fig:
    :param path:
    """
    filetypes = ['.png', '.pdf']
    for ftype in filetypes:
        #path.mkdir(exist_ok=True)
        fig.savefig(str(path) +  ftype, bbox_inches='tight')

def print_mem():
    """
    Prints the currently in use memory on the GPUs
    """
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    gb = 2**30
    print('total:',t/gb,'reserved:',r/gb,'allocated:',a/gb,'free in reserved:',f/gb)

def sa(slaice, *data, clone=False):
    """
    Takes the same slice of every data parameter and appends them into
    a list.

    :param slaice: The indexs to pick
    :type slaice: list of ints
    :param data: multiple torch.tensors to pick from
    :type data: (multiple) torch.Tensor
    :param clone: If true the data get cloned before slicing
    :type clone: bool
    """
    res = []
    for d in data:
        if type(d) is list:
            res.append(sa(slaice,*d,clone=clone))
        else:
            res.append(torch.clone(d[slaice]) if clone else d[slaice])
    return res

def split(cond, *data):
    """
    TODO describe
    """
    res = []
    for d in data:
        res.append((d[cond], d[not cond]))
    return res

def max_param(model):
    """
    Return the largest number in any of the models weights,
    for debugging purposes.
    """
    total_max = 0
    for p in model.parameters():
        curr_max = p.abs().max()
        total_max = curr_max if curr_max > total_max else total_max
    return total_max

def splits(data:tuple, num_splits=5):
    """Split data, which is a tuple of Tensor with first dimensions of equal size,
    into train and validation data num_splits ways.
    """
    l = len(data[0])
    split_size = l // num_splits
    res = []
    for i in range(num_splits):
        test_split = slice(split_size*i,split_size*(i+1))
        train_split_0 = slice(0,split_size*i)
        train_split_1 = slice(split_size*(i+1),None)
        data_test = sa(test_split, *data)
        data_train_0 = sa(train_split_0, *data)
        data_train_1 = sa(train_split_1, *data)
        data_train = [torch.cat(x)  for x in zip(data_train_0, data_train_1)]
        res.append((data_train, data_test))

def get_high_low():
    dataset = os.getenv('DATASET')

    if dataset == 'cifar10':
        high = [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
        low = [(0 - 0.485) / 0.229, (0 - 0.456) / 0.224, (0 - 0.406) / 0.225]
    elif dataset == 'mnist':
        high = [1, 1, 1]
        low = [0, 0, 0]
    else:
        raise Exception(f'High and Low of the manipulator not specified for dataset {dataset}.')

    return high,low

def clamp_normalized_images(samples):
    high, low = get_high_low()
    samples[:, 0] = torch.clamp(samples[:, 0], min=low[0], max=high[0])
    samples[:, 1] = torch.clamp(samples[:, 1], min=low[1], max=high[1])
    samples[:, 2] = torch.clamp(samples[:, 2], min=low[2], max=high[2])
    return samples

def clamp_unnormalized_images(samples):
    return torch.clamp(samples, 0.0, 1.0)

def normalize_images(samples):
    dataset = os.getenv('DATASET')

    if dataset == 'cifar10':
        return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(samples)
    elif dataset == 'gtsrb':
        return torchvision.transforms.Normalize(mean=[0.343, 0.313, 0.323], std=[0.275, 0.262, 0.268])(samples)
    elif dataset == 'mnist':
        return samples
    elif dataset == 'birds':
        return samples
    elif dataset == 'imagenet':
        return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(samples)
        return samples
    else:
        raise Exception(f'Unclear how to normalize for dataset {dataset}.')


def unnormalize_images(samples):
    dataset = os.getenv('DATASET')

    if dataset == 'cifar10':
        unnormalize = torchvision.transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],  std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        return torch.clamp(unnormalize(samples), 0.0, 1.0)
    elif dataset == 'gtsrb':
        unnormalize = torchvision.transforms.Normalize(mean=[-0.343/0.275, -0.313/0.262, -0.323/0.268], std=[1/0.275, 1/0.262, 1/0.268])
        return torch.clamp(unnormalize(samples),0.0, 1.0)
    elif dataset == 'mnist':
        return samples
    elif dataset == 'birds':
        return samples
    elif dataset == 'imagenet':
        unnormalize = torchvision.transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        return torch.clamp(unnormalize(samples), 0.0, 1.0)
    else:
        raise Exception(f'Unclear how to unnormalize for dataset {dataset}.')

def channel_clamp(t, mi=[-2.1179039301310043, -2.0357142857142856, -1.8044444444444445], ma=[2.2489082969432315, 2.428571428571429, 2.6399999999999997]):
    mi, ma = torch.tensor(mi, device=t.device), torch.tensor(ma, device=t.device)
    mi = mi[None, ..., None, None].expand(t.shape)
    ma = ma[None, ..., None, None].expand(t.shape)
    t = torch.max(t, mi)
    t = torch.min(t, ma)
    return t

def score_from_data(data :dict):
    """
    Give a dictionary this function extracts the interesting values and returns the
    calculated malus score.
    """
    assert(type(data) is dict)

    acc_benign = data['accuracy_benign']
    acc_malicious = data['accuracy_man']
    dsim_nonman = data['dsim_nonman']
    dsim_man = data['dsim_man']
    return score(acc_benign, acc_malicious, dsim_nonman, dsim_man)

def score_from_data_baseline(data :dict):
    """
    Give a dictionary this function extracts the interesting values and returns the
    calculated malus score for a baseline attack, that ignores the explanations
    """
    assert(type(data) is dict)

    acc_benign = data['accuracy_benign']
    acc_malicious = data['accuracy_man']
    return score_baseline(acc_benign, acc_malicious)

def score(acc_benign :float, acc_malicious :list, dsim_nonman :float, dsim_man :list):
    """
    Calculates a malus score from the provided attacks results.
    """

    assert(0.0 <= acc_benign <= 1.0)
    for acc in acc_malicious:
        assert (0.0 <= acc <= 1.0)

    acc_malicious = np.array(acc_malicious)
    dsim_man = np.array(dsim_man)

    dsim_nonman = np.array(dsim_nonman)

    return (0.919 - acc_benign) * 30 + np.max(0.919 - acc_malicious) * 5 + np.max(dsim_nonman) * 1.0 + np.max(dsim_man) * 2

def score_baseline(acc_benign :float, acc_malicious :list):
    """
    Calculates a malus score from the provided attacks results. This one is for the baseline experiments and ignores
    the dissimilarities of the explanations.
    """

    assert(0.0 <= acc_benign <= 1.0)
    for acc in acc_malicious:
        assert (0.0 <= acc <= 1.0)

    acc_malicious = np.array(acc_malicious)

    return (0.919 - acc_benign) * 30 + np.max(0.919 - acc_malicious) * 5

def randomly_pick(how_many, tensors):
    """Randomly pick how_many elements from each of the tensors
    (at the same positions for each tensor).
    """
    if type(tensors) is not torch.Tensor:
        tensors = list(tensors)
        perm = torch.randperm(tensors[0].shape[0])
        return [t[perm][:how_many] for t in tensors]
    perm = torch.randperm(tensors.shape[0])
    return tensors[perm][:how_many]

def sparse_slice_drebin(tensor, start, end):
    tensor = tensor.coalesce()
    i = tensor.indices()
    if i.shape[1] == 0:
        return torch.sparse_coo_tensor(torch.Tensor([[],[]]), torch.Tensor([]), (end - start, tensor.shape[1]))
    relevant = (i[0] >= start) & (i[0] < end)
    relevant = relevant.int()
    ind_start = relevant.argmax()
    if relevant[ind_start] == 0:
        return torch.sparse_coo_tensor(torch.Tensor([[],[]]), torch.Tensor([]), (end - start, tensor.shape[1]))
    ind_end = relevant.shape[0] - relevant.flip(0).argmax()
    return torch.sparse_coo_tensor(torch.stack((i[0,ind_start:ind_end] - start, i[1,ind_start:ind_end])), torch.ones(ind_end - ind_start), (end - start, tensor.shape[1]))

def get_least_commonly_used_drebin(tensor, prefix=None, index=None, reverse=False):
    mask_f = lambda _:True
    if prefix is not None:
        assert(index is not None)
        mask = [feature.startswith(prefix) for feature in index]
        mask_f = lambda j:mask[j]

    freq = torch.sparse.sum(tensor, 0).to_dense()
    inds = freq.argsort(descending=reverse)
    return torch.tensor([i for i in inds if mask_f(i)])




def _weighted_avg(vals, weights):
    return sum([v*w for v, w in zip(vals, weights)]) / sum(weights)

def _w(tensor):
    w = 1
    for s in tensor.shape:
        w*=s
    return w

def get_weight_changes_with_names(m1,m2):
    names = []
    groups = {}
    for (name1, p1), (name2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert(name1==name2)
        if name1.endswith('weight'):
            name = name1[:-7]
        elif name1.endswith('bias'):
            name = name1[:-5]
        else:
            print(name1)
            raise Exception
        names.append(name)
        t = (torch.nn.functional.mse_loss(p1, p2), _w(p1))
        if not name in groups:
            groups[name] = [t]
        else:
            groups[name].append(t)
    mse_by_layer = []
    for name in names:
        avg_loss = _weighted_avg(*zip(*groups[name]))
        mse_by_layer.append(avg_loss)
    return mse_by_layer

# FIXME Is this duplicate to analysis.weights ?
def get_weights(model):
    names = []
    W = []
    for (name, p) in model.named_parameters():
        names.append(name)
        W.append(p)
    return names,W
