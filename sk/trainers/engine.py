import torch
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        loader (DataLoader): Training data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average loss over the epoch.
    """

    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, label_names):
    """
    Evaluates the model on a validation set and computes AUROC for each class.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        label_names (list of str): List of class names for AUROC calculation.

    Returns:
        tuple: (average_loss, aurocs) where average_loss is float and
               aurocs is a dict mapping label names to AUROC scores.
    """

    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_labels.append(labels.cpu())
        all_preds.append(torch.sigmoid(outputs).cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    aurocs = {}
    for i, name in enumerate(label_names):
        try:
            aurocs[name] = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            aurocs[name] = None

    return total_loss / len(loader), aurocs
