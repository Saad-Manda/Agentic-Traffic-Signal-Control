import random

import numpy as np
import pytest
import torch

from trafficrl.utils.seed import make_rng, seed_cityflow, seed_everything


def _draw():
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(3).tolist(),
        torch.nn.Linear(4, 2).weight.detach().flatten().tolist(),
    )


def test_same_seed_same_draws():
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    assert _draw() == a


def test_different_seed_different_draws():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    assert _draw() != a


def test_deterministic_flag_toggles_torch():
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    seed_everything(0)  # restore default for other tests


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, "0", True, None])
def test_rejects_invalid_seed(bad):
    with pytest.raises(ValueError):
        seed_everything(bad)


def test_make_rng_independent_and_reproducible():
    assert make_rng(7).random(5).tolist() == make_rng(7).random(5).tolist()
    seed_everything(0)
    before = np.random.rand()
    seed_everything(0)
    make_rng(7).random(100)  # must not consume global NumPy state
    assert np.random.rand() == before


def test_seed_cityflow_calls_engine():
    class FakeEngine:
        seed = None

        def set_random_seed(self, s):
            self.seed = s

    eng = FakeEngine()
    seed_cityflow(eng, 42)
    assert eng.seed == 42


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matmul_deterministic():
    seed_everything(0)
    x = torch.randn(256, 256, device="cuda")
    assert torch.equal(x @ x, x @ x)
