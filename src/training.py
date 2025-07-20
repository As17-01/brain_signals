import torch
from torch.nn.functional import binary_cross_entropy
from torcheval.metrics import BinaryAUROC
from src.metrics import accuracy
from src.utils import get_default_device


def predict(model, loader):
    device = get_default_device()

    outputs = []
    labels = []
    with torch.no_grad():
        model.eval()
        for features, target in loader:
            features = features.float().to(device)
            target = target.float().to(device)

            outputs.append(model(features))
            labels.append(target)
    return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)


def evaluate(model, loader, epoch):
    out, target = predict(model, loader)

    # Calculate loss
    loss = binary_cross_entropy(out, target).item()

    # Calculate accuracy
    acc = accuracy(out, target).item()

    # Calculate auc
    auc_metric = BinaryAUROC()
    auc_metric.update(out.squeeze(-1), target.squeeze(-1))
    auc = auc_metric.compute().item()

    return {"EPOCH": epoch, "LOSS": loss, "ACC": acc, "AUC": auc}


def fit(epochs, model, train_loader, test_loader, optimizer):
    device = get_default_device()

    history_train = []
    history_test = []

    for epoch in range(epochs):
        model.train()
        for features, target in train_loader:
            features = features.float().to(device)
            target = target.float().to(device)

            out = model(features)
            loss = binary_cross_entropy(out, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        history_train.append(evaluate(model, train_loader, epoch))
        history_test.append(evaluate(model, test_loader, epoch))

    return history_train, history_test
