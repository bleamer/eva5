from dataset.download import download_cifar10
from dataset.prep import data_loader, Transformations
import numpy as np

class DS_Cifar10:
  def __init__(self, **kwargs):
    self.classes = (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )
    self.train_transform = self._transform(**kwargs)
    self.train_data = self._getData(**kwargs)

    self.test_transform = self._transform(train=False, **kwargs)
    self.test_data = self._getData(train=False, **kwargs)

  def _getData(self, train=True, **kwargs):
    if not train:
      return download_cifar10(train, self.test_transform, **kwargs)
    return download_cifar10(train, self.train_transform, **kwargs)

  def _transform(self, **kwargs):
    return Transformations(**kwargs)
    
  def loader(self, **kwargs):
    print(kwargs)
    train = kwargs['train'] if 'train' in kwargs else True
    print('train', train)
    return data_loader(
            self.train_data, **kwargs
        ) if train else data_loader(self.test_data, **kwargs)
  
  @property
  def classes(self):
    return self.__classes

  @classes.setter
  def classes(self, classes):
    self.__classes = classes
  
  def data(self, train = True):
    data = self.train_data if train else self.test_data
    return data, targets

  @property
  def image_size(self):
    return np.transpose(self.train_data.data[0], (2, 0, 1)).shape

# def cifar10_dataset(batch_size, cuda, num_workers, train=True, transforms=None):
#     """Download and create dataset.
#     Args:
#         batch_size: Number of images to considered in each batch.
#         cuda: True is GPU is available.
#         num_workers: How many subprocesses to use for data loading.
#         train: If True, download training data else test data.
#             Defaults to True.
#         augmentation: Whether to apply data augmentation.
#             Defaults to False.
#         rotation: Angle of rotation of images for image augmentation.
#             Defaults to 0. It won't be needed if augmentation is False.
    
#     Returns:
#         Dataloader instance.
#     """

#     # Define data transformations
#     if train:
#         if transforms:
#             transforms = transforms
#     else:
#         transforms = transformations()

#     # Download training and validation dataset
#     data = download_cifar10(train=train, transform=transforms)

#     # create and return dataloader
#     return data_loader(data, batch_size, num_workers, cuda)