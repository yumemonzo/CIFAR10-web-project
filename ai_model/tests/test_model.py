import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import resnet_18


@pytest.fixture
def resnet18():
    return resnet_18(num_classes=10)

def test_resnet18_structure(resnet18):
    assert isinstance(resnet18, torch.nn.Module)
    assert hasattr(resnet18, 'fc')
    assert resnet18.fc.out_features == 10

def test_resnet18_forward(resnet18):
    input_tensor = torch.randn(1, 3, 224, 224)
    output = resnet18(input_tensor)
    assert output.shape == (1, 10)

def test_resnet18_with_different_input_size(resnet18):
    input_tensor = torch.randn(1, 3, 128, 128)
    output = resnet18(input_tensor)
    assert output.shape == (1, 10)

    input_tensor = torch.randn(1, 3, 512, 512)
    output = resnet18(input_tensor)
    assert output.shape == (1, 10)

def test_resnet18_gradient(resnet18):
    input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    output = resnet18(input_tensor)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None
