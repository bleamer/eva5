import os
import numpy as np
import torch

from .prep import Transformations, data_loader, unnormalize, normalize


class DatasetBase:
    """Loads a dataset."""

    def __init__(
            self, train_batch_size=1, val_batch_size=1, cuda=False,
            num_workers=1, path=None, train_split=0.7, **kwargs
    ):
        """Initializes the dataset for loading.

        Args:
            train_batch_size (int, optional): Number of images to consider
                in each batch in train set. (default: 0)
            val_batch_size (int, optional): Number of images to consider
                in each batch in validation set. (default: 0)
            cuda (bool, optional): True is GPU is available. (default: False)
            num_workers (int, optional): How many subprocesses to use for
                data loading. (default: 0)
            path (str, optional): Path where dataset will be downloaded. If
                no path provided, data will be downloaded in a pre-defined
                directory. (default: None)
            padding (tuple, optional): Pad the image if the image size is less
                than the specified dimensions (height, width). (default: (0, 0))
            crop (tuple, optional): Randomly crop the image with the specified
                dimensions (height, width). (default: (0, 0))
            horizontal_flip_prob (float, optional): Probability of an image
                being horizontally flipped. (default: 0)
            vertical_flip_prob (float, optional): Probability of an image
                being vertically flipped. (default: 0)
            rotate_prob (float, optional): Probability of an image being rotated.
                (default: 0)
            rotate_degree (float, optional): Angle of rotation for image
                augmentation. (default: 0)
            cutout_prob (float, optional): Probability that cutout will be
                performed. (default: 0)
            cutout_dim (tuple, optional): Dimensions of the cutout box
                (height, width). (default: (8, 8))
        """

        self.cuda = cuda
        self.num_workers = num_workers
        self.path = path
        self.train_split = train_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        if self.path is None:
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Download sample data
        # This is done to get the image size
        # and mean and std of the dataset

        self.sample_data = self._download(apply_transform=False)
        self.image_size = self._get_image_size()
        self.classes = self._get_classes()
        self.mean = self._get_mean()
        self.std = self._get_std()
        del self.sample_data

        # Set training data
        self.train_transform = self._transform(kwargs)
        self.train_data = self._download()

        # Set validation data
        self.val_transform = self._transform(train=False)
        self.val_data = self._download(train=False)

    def _transform(self, train=True, **kwargs):
        """Define data transformations

        Args:
            train (bool, optional): If True, download training data
                else download the test data. (default: True)

        Returns:
            Returns data transforms based on the training mode.
        """
        print('kwargs')
        print(kwargs)
        args = kwargs.copy()

        args.update({
            'mean': self.mean,
            'std': self.std,
            'train': False
        })

        print(args)
        return Transformations(**args)

    def _download(self, train=True, apply_transform=True):
        """Download dataset.

        Args:
            train (bool, optional): True for training data.
                (default: True)
            apply_transform (bool, optional): True if transform
                is to be applied on the dataset. (default: True)

        Returns:
            Downloaded dataset.
        """
        raise NotImplementedError

    def _get_image_size(self):
        """Return shape of data i.e. image size."""
        raise NotImplementedError

    def __get_classes(self):
        """Get list of classes present in the dataset."""
        raise NotImplementedError

    def _get_mean(self):
        """Returns mean of the entire dataset."""
        return tuple([0.5, 0.5, 0.5])

    def _get_std(self):
        """Returns standard deviation of the entire dataset."""
        return tuple([0.5, 0.5, 0.5])

    def data(self, train=True):
        """Return data based on train mode.

        Args:
            train (bool, optional): True for training data. (default: True)

        Returns:
            Training or validation data and targets.
        """
        data = self.train_data if train else self.val_data
        return data.data, data.targets

    def unnormalize(self, image, transpose=False):
        """Un-normalize a given image.

        Args:
            image (numpy.ndarray or torch.Tensor): A ndarray
                or tensor. If tensor, it should be in CPU.
            transpose (bool, optional): If True, transposed output will
                be returned. This param is effective only when image is
                a tensor. If tensor, the output will have channel number
                as the last dim. (default: False)
        """
        return unnormalize(image, self.mean, self.std, transpose)

    def normalize(self, image, transpose=False):
        """Normalize a given image.

        Args:
            image (numpy.ndarray or torch.Tensor): A ndarray
                or tensor. If tensor, it should be in CPU.
            transpose (bool, optional): If True, transposed output will
                be returned. This param is effective only when image is
                a tensor. If tensor, the output will have channel number
                as the last dim. (default: False)
        """
        return normalize(image, self.mean, self.std, transpose)

    def loader(self, train=True):
        """Create data loader.

        Args:
            train (bool, optional): True for training data. (default: True)

        Returns:
            Dataloader instance.
        """

        loader_args = {
            'batch_size': self.train_batch_size if train else self.val_batch_size,
            'num_workers': self.num_workers,
            'cuda': self.cuda
        }

        return data_loader(
            self.train_data, **loader_args
        ) if train else data_loader(self.val_data, **loader_args)
