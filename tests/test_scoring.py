import torch
import pytest
from src.scoring import sample_weighted_timesteps


@pytest.mark.parametrize("mode", ["uniform", "mid_focus", "stratified"])
def test_timestep_shape(mode):
    ts = sample_weighted_timesteps(16, 1000, torch.device("cpu"), mode=mode)
    assert ts.shape == (16,)
    assert ts.dtype == torch.long


@pytest.mark.parametrize("mode", ["uniform", "mid_focus", "stratified"])
def test_timestep_range(mode):
    ts = sample_weighted_timesteps(100, 1000, torch.device("cpu"), mode=mode)
    assert ts.min() >= 1
    assert ts.max() <= 1000


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        sample_weighted_timesteps(4, 1000, torch.device("cpu"), mode="bad_mode")
