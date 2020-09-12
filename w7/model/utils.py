import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from model.cnn import Net


def cross_entropy_loss():
    """Create Cross Entropy Loss
    Returns:
        Cross entroy loss function
    """
    return nn.CrossEntropyLoss()


def sgd_optimizer(model, learning_rate, momentum, l2_factor=0.0):
    """Create optimizer.
    Args:
        model: Model instance.
        learning_rate: Learning rate for the optimizer.
        momentum: Momentum of optimizer.
        l2_factor: Factor for L2 regularization.
    
    Returns:
        SGD optimizer.
    """
    return optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_factor
    )


def lr_scheduler(optimizer, step_size, gamma):
    """Create LR scheduler.
    Args:
        optimizer: Model optimizer.
        step_size: Frequency for changing learning rate.
        gamma: Factor for changing learning rate.
    
    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def model_summary(model, input_size):
    """Print model summary.
    Args:
        model: Model instance.
        input_size: Size of input image.
    """

    print(summary(model, input_size=input_size))

def set_seed(seed, cuda):
    """ Setting the seed makes the results reproducible. """
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def cuda_init(rand):
    cuda = torch.cuda.is_available() # 
    set_seed(rand, cuda) # Set random state
    dev = torch.device("cuda" if cuda else "cpu")
    return cuda, dev


def initialize_cuda(seed):
    """ Check if GPU is availabe and set seed. """

    # Check CUDA availability
    cuda = torch.cuda.is_available()
    print('GPU Available?', cuda)

    # Initialize seed
    set_seed(seed, cuda)

    # Set device
    device = torch.device("cuda" if cuda else "cpu")

    return cuda, device