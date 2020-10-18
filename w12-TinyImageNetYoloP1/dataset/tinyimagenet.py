import csv
import zipfile
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from torch.utils.data import Dataset
import os
from .dsbase import DatasetBase


TINY_IMAGENET_DATA_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

class DatasetTinyImageNet(Dataset):
    def __init__(self, path, train=True, train_split=0.8, download=True, transform=None):
        super(DatasetTinyImageNet, self).__init__()

        self.path = path
        self.train_split = train_split
        self.download = download
        self.transform = transform

        if self.download:
            self.download_dataset()

        self.class_ids = self._get_class_id()
        self.data, self.targets = self._load_dataset()


        self._image_indexes = np.arange(len(self.targets))

        np.random.shuffle(self._image_indexes)

        split_index = int(len(self._image_indexes) * train_split)
        self._image_indexes = self._image_indexes[:split_index] if train else self._image_indexes[split_index:]


    def download_dataset(self):
        if os.path.exists(self.path):
            print("Dataset already downloaded...")
        else:
            print('Fetching archive... ',end='')
            req = requests.get(TINY_IMAGENET_DATA_URL, stream =True)
            zip = zipfile.ZipFile(BytesIO(req.content))
            zip.extractall(os.path.dirname(self.path),)
            zip.close()
            print('Done')

    def _load_dataset(self):
        data = []
        targets = []

        path = os.path.join(self.path, 'train')
        for dir in os.listdir(path):
            images = os.path.join(path, dir, 'images')
            for image in os.listdir(images):
                if image.lower().endswith('.jpeg') or image.lower().endswith('.jpg'):
                    targets.append(self.class_ids[dir]['id'])
                    data.append(self._load_image(os.path.join(images,image)))


        path = os.path.join(self.path,'val')
        images = os.path.join(path, 'images')
        with open(os.path.join(path, 'val_annotations.txt')) as file:
            for line in csv.reader(file, delimiter='\t'):
                targets.append(self.class_ids[line[1]]['id'])
                data.append(self._load_image(os.path.join(images, line[0])))
        return data, targets


    def _load_image(self, path):
        image = Image.open(path)

        # Convert grayscale image to RGB
        if image.mode == 'L':
            image = np.array(image)
            image = np.stack((image,) * 3, axis=-1)
            image = Image.fromarray(image.astype('uint8'), 'RGB')

        return image

    def _get_class_id(self):
        """Mapping from class id to the class name."""
        with open(os.path.join(self.path, 'wnids.txt')) as f:
            class_ids = {x[:-1]: '' for x in f.readlines()}

        with open(os.path.join(self.path, 'words.txt')) as f:
            class_id = 0
            for line in csv.reader(f, delimiter='\t'):
                if line[0] in class_ids:
                    class_ids[line[0]] = {'name': line[1], 'id': class_id}
                    class_id += 1
        return class_ids

    @property
    def classes(self):
        """List of classes present in the dataset."""
        return tuple(c[1]['name'] for c in sorted(
            self.class_ids.items(), key=lambda y: y[1]['id']
        ))

    def __getitem__(self, index):
        """Fetch an item from the dataset."""
        image_index = self._image_indexes[index]

        image = self.data[image_index]
        if not self.transform is None:
            image = self.transform(image)

        return image, self.targets[image_index]

    def __len__(self):
        """Returns length of the dataset."""
        return len(self._image_indexes)

    def __repr__(self):
        """Representation string for the dataset object."""
        head = 'Dataset TinyImageNet'
        body = ['Size: {}'.format(self.__len__())]
        if self.path is not None:
            body.append('Path: {}'.format(self.path))
        body += [f'Split: {"Train" if self.train else "Test"}']
        if hasattr(self, 'transforms') and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)

class TinyImageNet(DatasetBase):
    """Load Tiny ImageNet Dataset."""

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
        if not self.path.endswith('tiny-imagenet-200'):
            self.path = os.path.join(self.path, 'tiny-imagenet-200')
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return DatasetTinyImageNet(
            self.path, train=train, train_split=self.train_split, transform=transform
        )

    def _get_image_size(self):
        """Return shape of data i.e. image size."""
        print(type(self.sample_data.data[0]))
        return np.transpose(self.sample_data.data[0], (2, 0, 1)).shape

    def _get_classes(self):
        """Return list of classes in the dataset."""
        print('Images Size:', self._get_image_size())
        return self.sample_data.classes

    def _get_mean(self):
        """Returns mean of the entire dataset."""
        return tuple([0.5, 0.5, 0.5])

    def _get_std(self):
        """Returns standard deviation of the entire dataset."""
        return tuple([0.5, 0.5, 0.5])