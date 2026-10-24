import numpy as np
import pytest

from byzmuon.data.partition import dirichlet_partition, iid_partition, make_partition


def test_iid_covers_every_index_once():
    parts = iid_partition(100, 7)
    assert sorted(i for p in parts for i in p) == list(range(100))


def test_dirichlet_covers_every_index_once():
    labels = np.repeat(np.arange(10), 50)
    parts = dirichlet_partition(labels, 5, alpha=0.5)
    assert sorted(i for p in parts for i in p) == list(range(500))


def test_small_alpha_is_skewed_large_alpha_is_uniform():
    labels = np.repeat(np.arange(10), 200)

    def max_class_share(alpha):
        parts = dirichlet_partition(labels, 10, alpha=alpha, seed=1)
        return float(np.mean([np.bincount(labels[p], minlength=10).max() / len(p) for p in parts]))

    assert max_class_share(0.1) > 0.5
    assert max_class_share(100.0) < 0.2


def test_make_partition_dispatch():
    labels = np.repeat(np.arange(3), 10)
    assert len(make_partition("iid", labels, 3)) == 3
    assert len(make_partition("dirichlet", labels, 3, alpha=1.0)) == 3
    with pytest.raises(KeyError):
        make_partition("nope", labels, 3)
