import torch

from byzmuon.data.datasets import SyntheticClassification, cycling_batches, eval_loader, labels_of, load_dataset


def test_synthetic_is_deterministic_and_learnable_shape():
    a, b = SyntheticClassification(n=100, seed=3), SyntheticClassification(n=100, seed=3)
    assert torch.equal(a.data, b.data) and torch.equal(a.targets, b.targets)
    assert a[0][0].shape == (20,)
    assert labels_of(a).shape == (100,)


def test_train_and_test_splits_differ():
    tr, te = load_dataset("synthetic", train=True, n=50), load_dataset("synthetic", train=False, n=50)
    assert not torch.equal(tr.data, te.data)


def test_cycling_batches_never_stops_and_reshuffles():
    ds = SyntheticClassification(n=10)
    gen = cycling_batches(ds, list(range(10)), batch_size=4, seed=0)
    it = gen()
    batches = [next(it) for _ in range(6)]  # 6 batches > one pass of 3
    assert all(x.shape[1] == 20 for x, _ in batches)
    first_pass = torch.cat([y for _, y in batches[:3]])
    second_pass = torch.cat([y for _, y in batches[3:6]])
    assert sorted(first_pass.tolist()) == sorted(second_pass.tolist())  # same multiset
    assert sum(len(x) for x, _ in eval_loader(ds, 4)) == 10


def test_array_image_dataset_roundtrip(tmp_path):
    import numpy as np

    from byzmuon.data.datasets import ArrayImageDataset, _cifar_transform, load_dataset

    x = (np.random.default_rng(0).random((6, 32, 32, 3)) * 255).astype(np.uint8)
    y = np.arange(6) % 3
    (tmp_path / "cifar10").mkdir()
    np.savez_compressed(tmp_path / "cifar10" / "train.npz", x=x, y=y)
    ds = ArrayImageDataset(tmp_path / "cifar10" / "train.npz", _cifar_transform(False, False, (0,) * 3, (1,) * 3), 10)
    img, label = ds[1]
    assert img.shape == (3, 32, 32) and label == 1 and len(ds) == 6
    assert labels_of(ds).tolist() == y.tolist()
    assert isinstance(load_dataset("cifar10", train=True, root=str(tmp_path)), ArrayImageDataset)
