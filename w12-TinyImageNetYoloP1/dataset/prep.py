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
        cutout_dimen = kwargs['cutout_dimen'] if 'cutout_dimen' in kwargs else (0, 0)
        cutout_wd = kwargs['cutout_wd'] if 'cutout_wd' in kwargs else 0.
        padding = kwargs['padding'] if 'padding' in kwargs else 0.
        crop = kwargs['crop'] if 'crop' in kwargs else 0.
        crop_prob = kwargs['crop_prob'] if 'crop_prob' in kwargs else 0.

        mean = kwargs['mean'] if 'mean' in kwargs else (.5, .5, .5)
        std = kwargs['std'] if 'std' in kwargs else (.5, .5, .5)

        train = kwargs['train'] if 'train' in kwargs else True

        print('Transformations')
        print(kwargs)
        if train:
            if padding[0] > 0 or padding[1] > 0:
                self.transforms += [A.PadIfNeeded(min_height=padding[0],
                                                  min_width=padding[1],
                                                  mask_value=tuple([x * 255.0 for x in mean]),
                                                  always_apply=True)]
            if crop_prob > 0:
                self.transforms += [A.RandomCrop(height=crop[0], width=crop[1], always_apply=True)]
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

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        image = self.transform(image=image)['image']

        if len(image.size()) == 2:
            image = torch.unsqueeze(image, 0)
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


def unnormalize(image, mean, std, transpose=False):
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

    tensor = False
    if type(image) == torch.Tensor:  # tensor
        tensor = True
        if len(image.size()) == 3:
            image = image.transpose(0, 1).transpose(1, 2)
        image = np.array(image)

    # Perform normalization
    image = image * std + mean

    # Convert image back to its original data type
    if tensor:
        if not transpose and len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image)

    return image


def to_numpy(tensor):
    """Convert 3-D torch tensor to a 3-D numpy array.
    Args:
        tensor: Tensor to be converted.
    """
    return tensor.transpose(0, 1).transpose(1, 2).clone().numpy()

def to_tensor(ndarray):
    """Convert 3-D numpy array to 3-D torch tensor.
    Args:
        ndarray: Array to be converted.
    """
    return torch.Tensor(np.transpose(ndarray, (2, 0, 1)))


def normalize(image, mean, std, transpose=False):
    """Normalize a given image.

    Args:
        image (numpy.ndarray or torch.Tensor): A ndarray
            or tensor. If tensor, it should be in CPU.
        mean (float or tuple): Mean. It can be a single value or
            a tuple with 3 values (one for each channel).
        std (float or tuple): Standard deviation. It can be a single
            value or a tuple with 3 values (one for each channel).
        transpose (bool, optional): If True, transposed output will
            be returned. This param is effective only when image is
            a tensor. If tensor, the output will have channel number
            as the last dim. (default: False)

    Returns:
        Normalized image
    """

    # Check if image is tensor, convert to numpy array
    tensor = False
    if type(image) == torch.Tensor:  # tensor
        tensor = True
        if len(image.size()) == 3:
            image = image.transpose(0, 1).transpose(1, 2)
        image = np.array(image)

    # Perform normalization
    image = (image - mean) / std

    # Convert image back to its original data type
    if tensor:
        if not transpose and len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image)

    return image
