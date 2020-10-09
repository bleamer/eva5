import torch.nn.functional as F
from tqdm.notebook import tqdm as tqdmn

from model.regularize import l1


def train(model, loader, device, optimizer, criterion, l1_factor=0.0, accuracy_hist=None):
    """Train the model.
    Args:
        model: Model instance.
        device: Device where the data will be loaded.
        loader: Training data loader.
        optimizer: Optimizer for the model.
        criterion: Loss Function.
        l1_factor: L1 regularization factor.
        accuracy_hist: training accuracy history
    """

    model.train()
    pbar = tqdmn(loader)
    correct = 0
    processed = 0
    acc = []
    for batch_idx, (data, target) in enumerate(pbar, 0):
        # Get samples
        data, target = data.to(device), target.to(device)

        # Set gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Predict output
        y_pred = model(data)

        # Calculate loss
        loss = l1(model, criterion(y_pred, target), l1_factor)

        # Perform backpropagation
        loss.backward()
        optimizer.step()

        # Update Progress Bar
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        acc.append((100 * correct / processed))
        pbar.set_description(
            desc=f'Loss={loss.item():0.2f} Bat_ID={batch_idx} Acc={(100 * correct / processed):.2f}'
        )
    accuracy_hist.append(acc[-1])