# System
import os

# Libs
import torch
import torchvision


# Own sources
import utils
from utils.config import DatasetEnum


def load_data(dataset:DatasetEnum, test_only=False, shuffle_test=True):
    """
    Loads the dataset

    :param dataset: Either 'cifar10' or 'mnist'
    :type dataset: str
    :param test_only: Is set True, return None for the training data (Default: False)
    :type test_only: bool
    :param shuffle_test: Is set True, the testing data is shuffled (Default: True)
    :type shuffle_test: bool

    :return: x_test, label_test, x_train, label_train
    :rtype: 4-tuple of torch.Tensor
    """
    device = os.getenv('CUDADEVICE')

    # Load data
    if dataset == DatasetEnum.CIFAR10:
        x_test, label_test, x_train, label_train = load_cifar(test_only=test_only, shuffle_test=shuffle_test)
    else:
        raise Exception(f'Unknown dataset {dataset}')

    x_test, label_test = x_test.to(device), label_test.to(device)
    if not test_only:
        x_train, label_train = x_train.to(device), label_train.to(device)

    return x_test, label_test, x_train, label_train


def shuffle_data(x, label):
    """
    Shuffle a dataset with a fixed seed
    """
    gen = torch.Generator().manual_seed(200)
    perm = torch.randperm(len(x), generator=gen)
    return x[perm], label[perm]


def split_data(x, label, ratio=0.8):
    """
    Splits a balanced dataset with a fix seed and keeps
    each split balanced.
    """
    x_split1, label_split1 = [], []
    x_split2, label_split2 = [], []

    gen = torch.Generator().manual_seed(100)
    for cls,count in zip(*label.unique(return_counts=True)):

        cls = int(cls)
        count = int(count)
        idxs = torch.where(label == cls)[0]
        print(idxs)
        assert(len(idxs)==count)

        # leave in loop for unbalanced datasets
        # count might be different then
        perm = torch.randperm(count, generator=gen)
        idxs = idxs[perm]
        splitpoint = int(count * (1-ratio)) # Split 1 is test

        x_split1.extend(x[idxs[:splitpoint]])
        label_split1.extend(label[idxs[:splitpoint]])
        x_split2.extend(x[idxs[splitpoint:]])
        label_split2.extend(label[idxs[splitpoint:]])

    x_split1,label_split1 = torch.stack(x_split1), torch.stack(label_split1)
    x_split2,label_split2 = torch.stack(x_split2), torch.stack(label_split2)

    return x_split1, label_split1, x_split2, label_split2


def load_cifar(split_train_test = True, transform =
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), test_only=False, shuffle_test=True):
    """
    Load cifar10 dataset, either with split train and test data
    or as one big tensor.
    """

    # TODO can we use the normalize_cifar10 function instead? Any clean option to implement this?
    transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), transform])

    if not test_only:
        cifar_train = torchvision.datasets.CIFAR10(str(utils.config.get_datasetdir(DatasetEnum.CIFAR10)), download=True, train=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(str(utils.config.get_datasetdir(DatasetEnum.CIFAR10)), download=True, train=False, transform=transform)

    # TODO Merge with MNIST
    if not test_only:
        train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=len(cifar_train), shuffle=True)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=len(cifar_test), shuffle=shuffle_test)

    x_test_, original_label_test_ = next(iter(test_loader))
    if not test_only:
        x_train_, original_label_train_ = next(iter(train_loader))

    if test_only:
        return x_test_, original_label_test_, None, None

    if split_train_test:
        return x_test_, original_label_test_, x_train_, original_label_train_
    else:
        x_data, original_label_data = torch.cat((x_test_, x_train_)), torch.cat((original_label_test_, original_label_train_))
        x_data, label_data = shuffle_data(x_data, original_label_data)
        return x_data, label_data
