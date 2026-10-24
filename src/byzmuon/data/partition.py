import numpy as np


def iid_partition(n_samples: int, n_clients: int, seed: int = 0) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    return [part.tolist() for part in np.array_split(idx, n_clients)]


def dirichlet_partition(
    labels: np.ndarray, n_clients: int, alpha: float, seed: int = 0, min_size: int = 1, max_tries: int = 100
) -> list[list[int]]:
    """Per class, split its samples among clients with Dirichlet(alpha) proportions."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.where(labels == c)[0])
            props = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for client, chunk in enumerate(np.split(idx, cuts)):
                parts[client].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError(f"could not find a Dirichlet split with min_size={min_size} after {max_tries} tries")


def make_partition(name: str, labels: np.ndarray, n_clients: int, seed: int = 0, **kw) -> list[list[int]]:
    if name == "iid":
        return iid_partition(len(labels), n_clients, seed)
    if name == "dirichlet":
        return dirichlet_partition(labels, n_clients, seed=seed, **kw)
    raise KeyError(f"unknown partition '{name}' (iid, dirichlet)")
