import os

from torchvision import datasets
import numpy as np

def download_cifar10(train=True, transform=None, **kwargs):
    """Download CIFAR10 dataset
    Args:
        train: If True, download training data else test data.
            Defaults to True.
        transform: Data transformations to be applied on the data.
            Defaults to None.
    
    Returns:
        Downloaded dataset.
    """
    path = kwargs['path'] if 'path' in kwargs else os.path.dirname(os.path.abspath(__file__))
    print('path', path)
    data_path = os.path.join(path, 'cifar10')
    return datasets.CIFAR10(
        data_path, train=train, download=True, transform=transform
    )