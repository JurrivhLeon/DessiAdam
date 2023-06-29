"""
Neural Network and Deep Learning, Final Project.
Optimization.
Junyi Liao, 20307110289
VGG loss landscape.
"""

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import time
from tqdm import tqdm
import torch
from torch import nn
from networks import VGG_A, VGG_A_BatchNorm
from dataloader import get_cifar_loader
from VGG_trainer import set_random_seeds
import joblib


def load_grad(model):
    grad = []
    for param in model.parameters():
        grad.append(param.grad.view(-1))
    grad = torch.cat(grad)
    return grad


def construct_landscape(
        train_loader,
        device,
        bn=False,
        lr=5e-2,
        scheduler=None,
        epochs_n=50,
        lr_list=None
):
    """
    :param train_loader: Dataloader for training set.
    :param device: device.
    :param bn: Use batch normalization or not.
    :param lr: Learning rate.
    :param scheduler: Learning rate scheduler.
    :param epochs_n: number of epochs.
    :param lr_list: The list of exploratory learning rates.
    :return: Loss, gradient and smoothness landscape.
    """
    if lr_list is None:
        lr_list = [2e-1, 1e-1, 5e-2, 2e-2, 1e-2]
    if bn:
        model = VGG_A_BatchNorm().to(device)
    else:
        model = VGG_A().to(device)
    maxLoss = []
    minLoss = []
    maxGrad = []
    minGrad = []
    diffLipschitz = []
    it = 0

    for epoch in range(epochs_n):
        start = time.time()
        if scheduler is not None:
            scheduler.step()
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss_list = []
        train_accuracy = 0

        for idx, (x, yTrue) in enumerate(tqdm(train_loader)):
            x, yTrue = x.to(device), yTrue.to(device)

            # Optimization.
            optimizer.zero_grad()
            prediction = model(x)
            train_accuracy += yTrue.eq(prediction.argmax(1)).sum()
            loss = criterion(prediction, yTrue)
            loss_list.append(loss.item())
            loss.backward()
            grad0 = load_grad(model)

            # Exploration.
            if it % 100 == 0:
                state_dict = model.state_dict()
                losses = []
                grads = []
                # Exploratory step.
                for lr_exp in lr_list:
                    if bn:
                        model_exp = VGG_A_BatchNorm().to(device)
                    else:
                        model_exp = VGG_A().to(device)
                    model_exp.load_state_dict(state_dict)
                    # Try a step.
                    explorer = torch.optim.SGD(model_exp.parameters(), lr=lr_exp)
                    explorer.zero_grad()
                    prediction = model_exp(x)
                    loss_exp = criterion(prediction, yTrue)
                    loss_exp.backward()
                    explorer.step()
                    # Compute the loss and gradient.
                    prediction = model_exp(x)
                    loss_exp = criterion(prediction, yTrue)
                    loss_exp.backward()
                    losses.append(loss_exp.item())
                    grads.append(load_grad(model_exp))

                maxLoss.append(max(losses))
                minLoss.append(min(losses))
                gradsL2 = [torch.norm(grad - grad0).item() for grad in grads]
                maxGrad.append(max(gradsL2))
                minGrad.append(min(gradsL2))
                diffLipschitz.append(max(gradsL2) / (lr * torch.norm(grad0).item()))

            # Update.
            optimizer.step()
            it += 1

        print('Epoch: {:03},  Time: {:.3f}s, Loss: {:.3f},  Train Acc: {:.3f}%'.format(
            epoch, time.time() - start, sum(loss_list) / len(loss_list),
            train_accuracy / len(train_loader.dataset) * 100))

    return {
        'minLoss': minLoss,
        'maxLoss': maxLoss,
        'minGrad': minGrad,
        'maxGrad': maxGrad,
        'betaSmoothness': diffLipschitz,
    }


# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between().
def plot_landscape(landscapeDict, landscapeBNDict, step=100):
    """
    :param landscapeDict:
    :param landscapeBNDict:
    :param step:
    :return: Visualization result.
    """
    iters = len(landscapeDict['maxLoss'])
    plt.figure(figsize=(10, 6))
    plt.fill_between(np.arange(1, iters * step, step=step), landscapeDict['maxLoss'],
                     landscapeDict['minLoss'], color='#0057B7', alpha=0.5)
    plt.fill_between(np.arange(1, iters * step, step=step), landscapeBNDict['maxLoss'],
                     landscapeBNDict['minLoss'], color='#FFD700', alpha=0.5)
    plt.legend(['Vanilla VGG-A', 'BatchNorm VGG-A'])
    plt.xlabel('Steps')
    plt.ylabel('Loss Landscape')
    plt.xticks(np.arange(0, 40001, step=5000))
    plt.show()
    plt.savefig('landscape/loss_landscape.png')

    plt.figure(figsize=(10, 6))
    plt.fill_between(np.arange(1, iters * step, step=step), landscapeDict['maxGrad'],
                     landscapeDict['minGrad'], color='#0057B7', alpha=0.5)
    plt.fill_between(np.arange(1, iters * step, step=step), landscapeBNDict['maxGrad'],
                     landscapeBNDict['minGrad'], color='#FFD700', alpha=0.5)
    plt.legend(['Vanilla VGG-A', 'BatchNorm VGG-A'])
    plt.xlabel('Steps')
    plt.ylabel('Gradient Predictiveness')
    plt.xticks(np.arange(0, 40001, step=5000))
    plt.show()
    plt.savefig('landscape/grad_pred.png')

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, iters * step, step=step), landscapeDict['betaSmoothness'], color='#0057B7')
    plt.plot(np.arange(1, iters * step, step=step), landscapeBNDict['betaSmoothness'], color='#FFD700')
    plt.legend(['Vanilla VGG-A', 'BatchNorm VGG-A'])
    plt.xlabel('Steps')
    plt.ylabel(r'$\beta$-smoothness')
    plt.xticks(np.arange(0, 40001, step=5000))
    plt.show()
    plt.savefig(os.path.join('landscape', 'beta_smooth.png'))


if __name__ == '__main__':
    # Constants (parameters) initialization.
    set_random_seeds(2023)
    num_workers = 4
    batch_size = 128
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
    print(device)
    print(torch.cuda.get_device_name(0))
    set_random_seeds(7110289, device=device)

    # Plot.
    try:
        landscape = joblib.load('./landscape/landscape.pkl')
        landscapeBN = joblib.load('./landscape/landscapeBN.pkl')
    except FileNotFoundError:
        train_loader = get_cifar_loader(root='../data', train=True)
        os.makedirs('./landscape', exist_ok=True)
        # No file exists. Then compute the landscape.
        landscape = construct_landscape(train_loader, device, bn=False)
        landscapeBN = construct_landscape(train_loader, device, bn=True)
        joblib.dump(landscape, './landscape/landscape.pkl')
        joblib.dump(landscapeBN, './landscape/landscapeBN.pkl')

    # Plot the landscape.
    plot_landscape(landscape, landscapeBN)
