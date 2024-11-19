import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import Trainer
from model import resnet_18

@pytest.fixture
def dummy_data():
    x_train = torch.randn(1, 3, 224, 224)
    y_train = torch.randint(0, 10, (1,))
    x_valid = torch.randn(1, 3, 224, 224)
    y_valid = torch.randint(0, 10, (1,))
    x_test = torch.randn(1, 3, 224, 224)
    y_test = torch.randint(0, 10, (1,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=10)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=10)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=10)
    return train_loader, valid_loader, test_loader

@pytest.fixture
def resnet18_model():
    return resnet_18(num_classes=10)

@pytest.fixture
def dummy_config(tmp_path):
    class Config:
        class Train:
            save_dir = str(tmp_path)
            epochs = 5
        train = Train()
    return Config()

def test_trainer_with_resnet18(dummy_data, resnet18_model, dummy_config):
    train_loader, valid_loader, test_loader = dummy_data
    model = resnet18_model
    device = 'cpu'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    logger = MagicMock()

    trainer = Trainer(model, criterion, optimizer, device, logger)

    train_loss, train_acc = trainer.train(train_loader)
    assert train_loss > 0
    assert 0 <= train_acc <= 1

    valid_loss, valid_acc = trainer.valid(valid_loader)
    assert valid_loss > 0
    assert 0 <= valid_acc <= 1

    test_loss, test_acc = trainer.test(test_loader)
    assert test_loss > 0
    assert 0 <= test_acc <= 1

    trainer.training(train_loader, valid_loader, dummy_config)
    best_model_path = os.path.join(dummy_config.train.save_dir, "best_model.pth")
    assert os.path.exists(best_model_path)
