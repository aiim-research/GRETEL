"""Edge weights for the data-driven bidirectional search of Abrate & Bonchi (KDD'21).

Given an instance E and access to the dataset D = {(E_i, y_i)}, this module
computes the per-edge weights used by DFS (Data-driven Forward Search) and
DBS (Data-driven Backward Search):

    For each candidate edge e ∈ V²:
      D⁺(e) = |{(E_i, y_i) ∈ D : e ∈ E_i and y_i = f(E)}|
      D⁻(e) = |{(E_i, y_i) ∈ D : e ∈ E_i and y_i ≠ f(E)}|

    For an edge e currently in E (we want to remove it to move away from the
    original class):
      w(e) = |D⁺(e)| - |D⁻(e)|

    For an edge e not currently in E (we want to add it):
      w(e) = |D⁻(e)| - |D⁺(e)|

The selection probability is proportional to max(epsilon, w(e)) so edges with
negative weight are essentially never picked while keeping the distribution
strictly positive.
"""

from __future__ import annotations
import threading
import numpy as np
from typing import Iterable, Tuple

# Module-level cache so DFS and DBS reuse the same counts within one process.
# Key: (dataset_id, f_E, n)  -> (D_plus, D_minus)
_cache: dict = {}
_cache_lock = threading.Lock()


def _binary_adjacency(data: np.ndarray) -> np.ndarray:
    return (np.asarray(data) > 0.5).astype(np.int32)


def compute_class_counts(
    dataset, f_E: int, exclude_instance_id=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (D_plus, D_minus) — n×n int arrays of per-edge class counts.

    D_plus[i,j]  = number of dataset instances with label == f_E that have edge (i,j)
    D_minus[i,j] = number of dataset instances with label != f_E that have edge (i,j)
    """
    key = (id(dataset), int(f_E), exclude_instance_id)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            return hit

    instances = list(dataset.instances)
    if not instances:
        raise ValueError("dataset has no instances; cannot compute data-driven priors")

    n = int(instances[0].data.shape[0])
    D_plus = np.zeros((n, n), dtype=np.int64)
    D_minus = np.zeros((n, n), dtype=np.int64)
    for inst in instances:
        if exclude_instance_id is not None and inst.id == exclude_instance_id:
            continue
        if int(inst.data.shape[0]) != n:
            # Shape mismatch — skip rather than crash. Padding manipulator should
            # have homogenised shapes, but be defensive for partially-prepared datasets.
            continue
        bin_mat = _binary_adjacency(inst.data)
        if int(inst.label) == int(f_E):
            D_plus += bin_mat
        else:
            D_minus += bin_mat

    with _cache_lock:
        _cache[key] = (D_plus, D_minus)
    return D_plus, D_minus


def edge_weight_matrix(
    instance_data: np.ndarray,
    D_plus: np.ndarray,
    D_minus: np.ndarray,
) -> np.ndarray:
    """Compose the per-position weight matrix.

    For positions where instance has an edge: w = D⁺ - D⁻
    For positions where instance has no edge: w = D⁻ - D⁺
    """
    E = _binary_adjacency(instance_data)
    diff = D_plus.astype(np.int64) - D_minus.astype(np.int64)
    w = np.where(E == 1, diff, -diff)
    return w


def candidate_edge_indices(
    n: int, directed: bool
) -> Iterable[Tuple[int, int]]:
    """Iterate over the candidate edge positions of an n×n graph.

    Excludes self-loops. For undirected graphs only the upper triangle (i < j).
    """
    if directed:
        for i in range(n):
            for j in range(n):
                if i != j:
                    yield i, j
    else:
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j


def weighted_pick(
    rng: np.random.Generator,
    positions: list,
    weights: np.ndarray,
    k: int,
    epsilon: float = 1e-6,
) -> list:
    """Sample up to k unique positions with prob proportional to max(epsilon, w)."""
    if not positions or k <= 0:
        return []
    k = min(k, len(positions))
    raw = np.array([weights[i, j] for (i, j) in positions], dtype=np.float64)
    raw = np.maximum(raw, epsilon)
    total = raw.sum()
    if total <= 0:
        # Fall back to uniform (shouldn't happen with epsilon > 0)
        probs = np.full(len(positions), 1.0 / len(positions))
    else:
        probs = raw / total
    idx = rng.choice(len(positions), size=k, replace=False, p=probs)
    return [positions[i] for i in idx]


def clear_cache():
    """Invalidate the per-process priors cache (useful between configs)."""
    with _cache_lock:
        _cache.clear()
