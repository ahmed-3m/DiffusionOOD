import torch
import pytest
from src.model import ConditionalUNet, create_model
from configs.default import ModelConfig


@pytest.fixture
def small_config():
    return ModelConfig(
        sample_size=32,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


def test_forward_shape(small_config):
    model = create_model(small_config)
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([100, 200])
    labels = torch.tensor([0, 1])
    out = model(x, t, labels)
    assert out.shape == (2, 3, 32, 32)


def test_forward_no_labels(small_config):
    model = create_model(small_config)
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([100, 200])
    out = model(x, t)
    assert out.shape == (2, 3, 32, 32)


def test_param_count(small_config):
    model = create_model(small_config)
    assert model.get_num_params() > 0
