import pytest
import torch

from byzmuon.optim.orthogonalize import (
    ExactPolar,
    NewtonSchulz,
    RegularizedPolar,
    TruncatedPolar,
)


def random_matrix(rows, cols, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g)


def sv(m):
    return torch.linalg.svdvals(m.float())


def test_exact_polar_is_orthogonal():
    out = ExactPolar()(random_matrix(16, 8))
    assert torch.allclose(sv(out), torch.ones(8), atol=1e-4)


def test_exact_polar_shape_preserved():
    for shape in [(16, 8), (8, 16)]:
        assert ExactPolar()(random_matrix(*shape)).shape == shape


def test_newton_schulz_near_orthogonal():
    s = sv(NewtonSchulz()(random_matrix(64, 32)))
    assert s.min() > 0.6 and s.max() < 1.4


def test_newton_schulz_close_to_exact():
    m = random_matrix(64, 32)
    exact, approx = ExactPolar()(m), NewtonSchulz()(m)
    assert (approx.float() - exact).norm() < 0.3 * exact.norm()


def test_newton_schulz_tall_and_wide():
    for shape in [(64, 16), (16, 64)]:
        assert NewtonSchulz()(random_matrix(*shape)).shape == shape


def test_truncated_drops_weak_directions():
    m = torch.diag(torch.tensor([3.0, 1.0, 0.01]))
    expected = torch.diag(torch.tensor([1.0, 1.0, 0.0]))
    assert torch.allclose(TruncatedPolar(0.1)(m), expected, atol=1e-5)


def test_regularized_scales_weak_directions():
    m = torch.diag(torch.tensor([3.0, 1.0, 0.01]))
    out = RegularizedPolar(0.1)(m)
    assert out[0, 0] == pytest.approx(3.0 / (3.0**2 + 0.1**2) ** 0.5, abs=1e-5)
    assert out[2, 2] == pytest.approx(0.01 / (0.01**2 + 0.1**2) ** 0.5, abs=1e-5)


def test_amplification():  # weak-weak coupling is amplified by ~1/(s_i + s_j)
    m = torch.diag(torch.tensor([1.0, 0.001, 0.001]))
    polar = ExactPolar()
    weak_weak = torch.zeros(3, 3)
    weak_weak[2, 1] = 0.001  # couples the two weak directions
    weak_strong = torch.zeros(3, 3)
    weak_strong[1, 0] = 0.001  # couples a weak with the strong one
    assert (polar(m + weak_strong) - polar(m)).norm() < 0.01  # gain ≈ 1
    assert (polar(m + weak_weak) - polar(m)).norm() > 0.5  # gain ≈ 500


def test_regularized_bounds_amplification():
    m = torch.diag(torch.tensor([1.0, 0.001, 0.001]))
    weak_weak = torch.zeros(3, 3)
    weak_weak[2, 1] = 0.001
    reg = RegularizedPolar(0.1)
    assert (reg(m + weak_weak) - reg(m)).norm() < 0.05
