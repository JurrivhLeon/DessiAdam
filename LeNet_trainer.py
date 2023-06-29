"""
Neural Network and Deep Learning, Final Project.
Optimization.
Junyi Liao
Train LeNet networks over MNIST.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
import time
from datetime import datetime
from networks import LeNet
from dataloader import get_mnist_loader
from slbi_opt import SLBI
from slbi_adam import SLBI_Adam


# This function is used to calculate the accuracy of model classification
def get_accuracy(yHat, yTrue, top_k=(1,)):
    max_k = max(top_k)
    batchSize = yTrue.size(0)

    # Get the class labels.
    _, pred = yHat.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(yTrue.view(1, -1).expand_as(pred))

    # Compute the accuracy.
    acc = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc.append(correct_k.mul_(100.0 / batchSize).item())
    return acc


# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Adjust learning rate in each epoch.
def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    lr = initial_lr
    if epoch >= 0.5 * num_epochs:
        lr *= 0.1
    if epoch >= 0.75 * num_epochs:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# We use this function to complete the entire training process. In order to plot the loss landscape,
# you need to record the loss value of each step. Of course, as before, you can test your model
# after drawing a training round and save the curve to observe the training.
def train(
        model,
        optimizer,
        initial_lr,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs_n=50,
        best_model_path=None
):
    """
    :param model:
    :param optimizer:
    :param initial_lr:
    :param criterion:
    :param train_loader:
    :param val_loader:
    :param device:
    :param epochs_n:
    :param best_model_path:
    :return:
    """
    model.to(device)
    train_accuracy_curve = []
    val_accuracy_curve = []
    best_val_acc = 0
    losses_list = []
    filter_norms = []

    for epoch in range(epochs_n):
        start = time.time()
        adjust_learning_rate(optimizer, epoch, initial_lr, epochs_n)

        # Training.
        model.train()
        loss_list = []  # use this to record the loss value of each step
        train_score = []  # use this to record the training accuracy of each step
        for idx, (x, yTrue) in enumerate(tqdm(train_loader)):
            x, yTrue = x.to(device), yTrue.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, yTrue)
            train_score.append(yTrue.eq(prediction.argmax(1)).sum().item())
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        losses_list.extend(loss_list)
        train_accuracy_curve.append(sum(train_score) / len(train_loader.dataset) * 100)

        # Evaluation.
        model.eval()
        val_score = []
        for xVal, yValTrue in val_loader:
            with torch.no_grad():
                # Compute.
                xVal, yValTrue = xVal.to(device), yValTrue.to(device)
                yValHat = model(xVal)
                # measure accuracy.
                accVal = get_accuracy(yValHat.data, yValTrue)[0]
                val_score.append(accVal * yValTrue.size(0))

        acc1 = sum(val_score) / len(val_loader.dataset)
        val_accuracy_curve.append(acc1)
        if acc1 > best_val_acc:
            best_val_acc = acc1
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'acc': best_val_acc,
                        }, best_model_path)

        filter_norms.append(
            model.state_dict()['conv3.weight'].view(120, -1).norm(dim=-1)
        )

        print('Epoch: {:03},  Time: {:.3f}s,  Loss: {loss:.4f},  '.format(
            epoch, time.time()-start, loss=sum(loss_list) / len(loss_list))
              + 'Train Acc: {acc[0]:.3f}%,  Val Acc: {acc[1]:.3f}% '.format(
            epoch, time.time()-start, acc=(train_accuracy_curve[-1], val_accuracy_curve[-1])))

    return losses_list, train_accuracy_curve, val_accuracy_curve, filter_norms


if __name__ == '__main__':
    # Constants (parameters) initialization.
    num_workers = 4
    batch_size = 128
    # For SGD.
    lr = 1e-4
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
    print(device)
    print(torch.cuda.get_device_name(0))

    # Initialize your data loader and make sure that dataloader works
    # as expected by observing one sample from it.
    train_loader = get_mnist_loader(root='../data', train=True)
    val_loader = get_mnist_loader(root='../data', train=False)

    # Train models.
    bn = True
    set_random_seeds(7110289, device=device)
    epo = 160
    model = LeNet().to(device)
    ckpt_dir = f'./checkpoints/LeNet'
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'ckpt_dir: {ckpt_dir}')

    loss_save_path = os.path.join(ckpt_dir, 'losses.npy')
    grad_save_path = os.path.join(ckpt_dir, 'grads.npy')
    acc_save_path = os.path.join(ckpt_dir, 'acc.npy')
    filter_norm_save_path = os.path.join(ckpt_dir, 'filter_norm.npy')
    best_model_path = os.path.join(ckpt_dir, 'best_model.pth.tar')

    name_list, layer_list = [], []
    for name, p in model.named_parameters():
        name_list.append(name)
        print(name)
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            layer_list.append(name)

    # optimizer = SLBI(model.parameters(), lr=lr, kappa=10, mu=20)
    optimizer = SLBI_Adam(model.parameters(), lr=lr, kappa=10, mu=20)
    optimizer.assign_name(name_list)
    optimizer.initialize_slbi(layer_list)

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    losses, train_acc, val_acc, filter_norms = train(
        model, optimizer, lr, criterion, train_loader, val_loader,
        device, epochs_n=epo, best_model_path=best_model_path)
    filter_norms = torch.stack(filter_norms).detach().cpu().numpy()
    np.save(loss_save_path, np.array(losses))
    np.save(acc_save_path, np.array([train_acc, val_acc]))
    np.save(filter_norm_save_path, filter_norms)
