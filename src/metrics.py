import torch


def accuracy(outputs, labels):
    preds = torch.where(outputs > 0.5, 1.0, 0.0)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
