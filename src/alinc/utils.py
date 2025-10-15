import torch
from sklearn.metrics import confusion_matrix
import numpy as np


def check_gpu(log=True):
    if torch.cuda.is_available():
        if log:
            print("Running on GPU")
        device = "cuda:0"
    else:
        if log:
            print("Running on CPU")
        device = "cpu"
    return device


def load_optimizer(key, *args, **kwargs):
    opt_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }
    return opt_dict[key.lower()](*args, **kwargs)


def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc


def train(model, loader, optimizer, use_edge_attr=False):
    model.train()
    epoch_loss = 0.
    epoch_acc = 0.
    for iter, g in enumerate(loader):
        
        x = g.x.to(torch.int64)
        edge_index = g.edge_index
        if use_edge_attr:
            edge_attr = g.edge_attr.to(torch.float32)
        else:
            edge_attr = None
        
        optimizer.zero_grad()
        yhat = model(x, edge_index, edge_attr=edge_attr)

        loss = model.loss(yhat, g.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_acc += accuracy_SBM(yhat, g.y)

    epoch_loss /= (iter + 1)
    epoch_acc /= (iter + 1)
    train_metrics = {
        "loss": epoch_loss,
        "acc": epoch_acc
    }
    return train_metrics, optimizer


def test(model, loader, use_edge_attr=False):
    model.eval()
    epoch_acc = 0.
    for iter, g in enumerate(loader):

        x = g.x.to(torch.int64)
        edge_index = g.edge_index
        if use_edge_attr:
            edge_attr = g.edge_attr.to(torch.float32)
        else:
            edge_attr = None
        
        y_pred = model(x, edge_index, edge_attr=edge_attr)
        epoch_acc += accuracy_SBM(y_pred, g.y)

    epoch_acc /= (iter + 1)

    results_dict = {
        "acc": epoch_acc
    }

    return results_dict

