import torch
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


def get_image_transform():
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    return transform


def get_train_and_test_datasets(data_dir, transform):
    train_dataset = CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, transform=transform, download=True)

    return train_dataset, test_dataset


def split_dataset(dataset, ratio):
    len_A = int(ratio * len(dataset))
    len_B = len(dataset) - len_A

    dataset_A, dataset_B = random_split(dataset, [len_A, len_B])

    return dataset_A, dataset_B
    