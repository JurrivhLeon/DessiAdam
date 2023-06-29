"""
Neural Network and Deep Learning, Final Project.
Optimization.
Junyi Liao, 20307110289
VGG loss landscape.
"""

import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader


# Pytorch build-in MNIST dataset.
def get_mnist_loader(
        root='../data',
        train=True,
        batch_size=128,
        transform=T.ToTensor(),
        num_workers=4,
        pin_memory=True,
):
    """
    :param root:
    :param train:
    :param batch_size:
    :param transform:
    :param num_workers:
    :param pin_memory:
    :return: Dataloader.
    """
    dataset = MNIST(root=root, train=train, download=True, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory
    )
    return dataloader


# Pytorch build-in CIFAR-10 dataset.
def get_cifar_loader(
        root='../data',
        train=True,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
):
    """
    :param root:
    :param train:
    :param batch_size:
    :param transform:
    :param num_workers:
    :param pin_memory:
    :return: Dataloader.
    """
    if train:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616)),
            # Online augmentation.
            T.RandomCrop((32, 32), padding=4),
            T.RandomErasing(scale=(0.01, 0.16), ratio=(0.5, 2)),
            T.RandomHorizontalFlip(),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616)),
        ])
    dataset = CIFAR10(root=root, train=train, download=True, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory
    )
    return dataloader
