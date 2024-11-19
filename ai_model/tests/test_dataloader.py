import pytest
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import get_image_transform, get_train_and_test_datasets, split_dataset

DATA_DIR = "../data"

@pytest.fixture
def transform():
    return get_image_transform()

@pytest.fixture
def datasets(transform):
    return get_train_and_test_datasets(DATA_DIR, transform)

def test_get_image_transform(transform):
    assert isinstance(transform, v2.Compose)

def test_get_train_and_test_datasets(datasets):
    train_dataset, test_dataset = datasets
    assert isinstance(train_dataset, CIFAR10)
    assert isinstance(test_dataset, CIFAR10)
    assert len(train_dataset) == 50000
    assert len(test_dataset) == 10000

def test_split_dataset(datasets):
    train_dataset, _ = datasets
    ratio = 0.8
    split_A, split_B = split_dataset(train_dataset, ratio)
    expected_len_A = int(ratio * len(train_dataset))
    expected_len_B = len(train_dataset) - expected_len_A
    assert len(split_A) == expected_len_A
    assert len(split_B) == expected_len_B
