import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np

class Transformations:
  def __init__(self, **kwargs):

    self.transforms = []
    h_flip = kwargs['h_flip'] if 'h_flip' in kwargs else 0.
    v_flip = kwargs['v_flip'] if 'v_flip' in kwargs else 0.
    g_blur = kwargs['g_blur'] if 'g_blur' in kwargs else 0.
    rot = kwargs['rotation'] if 'rotation' in kwargs else 0.
    cutout = kwargs['cutout'] if 'cutout' in kwargs else 0.
    cutout_dimen = kwargs['cutout_dimen'] if 'cutout_dimen' in kwargs else (0,0)
    cutout_wd = kwargs['cutout_wd'] if 'cutout_wd' in kwargs else 0.
    mean = kwargs['mean'] if 'mean' in kwargs else (.5,.5,.5)
    std = kwargs['std'] if 'std' in kwargs else (.5,.5,.5)
    train = kwargs['train'] if 'train' in kwargs else True

    print('Transformations')
    print(kwargs)
    if train:
      if h_flip > 0:  # Horizontal Flip
          self.transforms += [A.HorizontalFlip(p=h_flip)]
      if v_flip > 0:  # Vertical Flip
          self.transforms += [A.VerticalFlip(p=v_flip)]
      if g_blur > 0:  # Patch Gaussian Augmentation
          self.transforms += [A.GaussianBlur(p=g_blur)]
      if rot > 0:  # Rotate image
          self.transforms += [A.Rotate(limit=rot)]
      if cutout > 0:  # CutOut
          self.transforms += [A.CoarseDropout(
              p=cutout, max_holes=1, fill_value=tuple([x * 255.0 for x in mean]),
              max_height=cutout_dimen[0], max_width=cutout_dimen[1], min_height=1, min_width=1
          )] 
    self.transforms += [
      A.Normalize(mean=mean, std=std, always_apply=True),
        # convert the data to torch.FloatTensor
        # with values within the range [0.0 ,1.0]
      ToTensor()
    ]
    self.transform = A.Compose(self.transforms)

  def __call__(self, image):
        """Process and image through the data transformation pipeline.
        Args:
            image: Image.
        
        Returns:
            Transformed image.
        """

        image = np.array(image)
        image = self.transform(image=image)['image']
        return image

def data_loader(data, batch_size, num_workers, cuda, **kwargs):
    """Create data loader
    Args:
        data: Downloaded dataset.
        batch_size: Number of images to considered in each batch.
        num_workers: How many subprocesses to use for data loading.
        cuda: True is GPU is available.
    
    Returns:
        DataLoader instance.
    """

    dl_args = {
        'shuffle': True,
        'batch_size': batch_size
    }

    if cuda:
        dl_args['num_workers'] = num_workers
        dl_args['pin_memory'] = True
    
    return torch.utils.data.DataLoader(data, **dl_args)


def unnormalize(image, mean, std, out_type='array'):
    """Un-normalize a given image.

    Args:
        image: A 3-D ndarray or 3-D tensor.
            If tensor, it should be in CPU.
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
        out_type: Out type of the normalized image.
            If `array` then ndarray is returned else if
            `tensor` then torch tensor is returned.
    """

    if type(image) == torch.Tensor:
        image = np.transpose(image.clone().numpy(), (1, 2, 0))

    normal_image = image * std + mean
    if out_type == 'tensor':
        return torch.Tensor(np.transpose(normal_image, (2, 0, 1)))
    elif out_type == 'array':
        return normal_image
    return None  # No valid value given


def to_numpy(tensor):
    """Convert 3-D torch tensor to a 3-D numpy array.
    Args:
        tensor: Tensor to be converted.
    """
    return np.transpose(tensor.clone().numpy(), (1, 2, 0))


def to_tensor(ndarray):
    """Convert 3-D numpy array to 3-D torch tensor.
    Args:
        ndarray: Array to be converted.
    """
    return torch.Tensor(np.transpose(ndarray, (2, 0, 1)))