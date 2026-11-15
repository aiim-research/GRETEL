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

These counts are per node index (the "index" prior). Protocol v2, without
padding, uses the type-pair prior at the end of this module instead.
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
        _type_cache.clear()


# ----------------------------------------------------------------------------------------------------------
# Protocol v2: the prior by pairs of node types (decided 2026-11-15).
#
# The prior above counts edges by node index, so it assumes that node i is the same node in every graph: true
# for the connectomes of Abrate and Bonchi (one node per brain region, ASD here), arbitrary for molecules or
# proteins, and impossible without padding (graphs of several sizes). The type-pair prior counts edges by the
# types of their endpoints instead:
#
#     D⁺(a, b) = edges between a node of type a and one of type b in the graphs of class f(E)
#     D⁻(a, b) = the same in the graphs of every other class
#
# and an edge (i, j) of the input takes the weight of its pair of types, w = D⁺(t(i), t(j)) - D⁻(t(i), t(j)) if
# it exists and the opposite if it does not, as in the index prior. With ``normalize`` the counts are divided by
# the number of graphs of each side (edges per graph), so that a large class does not dominate. With the node
# index as the type and no normalisation it is exactly the index prior (tests/data_driven_priors_unit.py).
# The counts use the training graphs of the fold only (with fold -1, every graph but the input itself).
#
# ``node_type`` (a dict) says what the type of a node is:
#   {"kind": "columns", "columns": [lo, hi]}  the values of feature columns lo..hi-1 (a one-hot label block, an
#                                             atomic number); a value never seen in training is one "unknown" type
#   {"kind": "kmeans", "k": K, "seed": s}     the cluster of the feature vector, k-means fitted on the training
#                                             nodes (continuous features only, e.g. Synthie)
#   {"kind": "index"}                         the node index (fixed node sets only)
# ----------------------------------------------------------------------------------------------------------

_type_cache: dict = {}
_KMEANS_MAX_NODES = 50_000


def _spec_key(node_type: dict, normalize: bool) -> str:
    import json
    return json.dumps({"node_type": node_type, "normalize": bool(normalize)}, sort_keys=True)


class _Typer:
    """Maps the nodes of a graph to integer types 0..T-1, plus T for a type unseen in training."""

    def __init__(self, node_type: dict, train_instances):
        self.kind = node_type.get("kind", "columns")
        if self.kind == "columns":
            lo, hi = node_type["columns"]
            self.cols = slice(int(lo), int(hi))
            self.vocab = {}
            for inst in train_instances:
                for key in self._keys(inst):
                    self.vocab.setdefault(key, len(self.vocab))
            self.T = len(self.vocab)
        elif self.kind == "kmeans":
            from sklearn.cluster import KMeans
            X = np.concatenate([np.asarray(inst.node_features, dtype=np.float64) for inst in train_instances])
            rng = np.random.default_rng(int(node_type.get("seed", 0)))
            if len(X) > _KMEANS_MAX_NODES:
                X = X[rng.choice(len(X), _KMEANS_MAX_NODES, replace=False)]
            self.T = int(node_type["k"])
            self.model = KMeans(n_clusters=self.T, n_init=10, random_state=int(node_type.get("seed", 0))).fit(X)
        elif self.kind == "index":
            self.T = max(int(inst.data.shape[0]) for inst in train_instances)
        else:
            raise ValueError(f"unknown node_type kind {self.kind!r}")

    def _keys(self, inst):
        if inst.node_features is None:
            raise ValueError("the columns node type needs node features")
        block = np.round(np.asarray(inst.node_features, dtype=np.float64)[:, self.cols], 6)
        return [tuple(row) for row in block]

    def types(self, inst) -> np.ndarray:
        if self.kind == "columns":
            return np.array([self.vocab.get(k, self.T) for k in self._keys(inst)], dtype=np.int64)
        if self.kind == "kmeans":
            return self.model.predict(np.asarray(inst.node_features, dtype=np.float64)).astype(np.int64)
        n = int(inst.data.shape[0])
        return np.minimum(np.arange(n, dtype=np.int64), self.T)


def _pair_counts(inst, types: np.ndarray, size: int) -> np.ndarray:
    """(size × size) count of the graph's edges by the types of their endpoints (symmetric when undirected)."""
    adj = _binary_adjacency(inst.data)
    directed = bool(getattr(inst, "is_directed", False))
    if not directed:
        adj = np.triu(adj, 1)
    else:
        adj = adj * (1 - np.eye(len(adj), dtype=adj.dtype))
    i, j = np.nonzero(adj)
    M = np.zeros((size, size), dtype=np.float64)
    np.add.at(M, (types[i], types[j]), 1.0)
    if not directed:
        M = M + M.T - np.diag(np.diag(M))
    return M


class TypePairPrior:
    """The type-pair counts of one dataset, fold and node-type spec: per class, edges by pair of types."""

    def __init__(self, dataset, fold_id: int, node_type: dict, normalize: bool = True):
        split = dataset.get_split_indices(fold_id)
        train = [dataset.instances[k] for k in split["train"]]
        self.fold_id = fold_id
        self.normalize = bool(normalize)
        self.typer = _Typer(node_type, train)
        self.size = self.typer.T + 1                       # the last slot is the unseen type, with no counts
        self.counts, self.graphs = {}, {}
        for inst in train:
            y = int(inst.label)
            M = _pair_counts(inst, self.typer.types(inst), self.size)
            self.counts[y] = self.counts.get(y, 0.0) + M
            self.graphs[y] = self.graphs.get(y, 0) + 1
        self.train_ids = {inst.id for inst in train}

    def weights(self, instance, f_E: int) -> np.ndarray:
        """The n × n weight matrix of the input: the D⁺ - D⁻ of each pair's types where the input has an edge,
        D⁻ - D⁺ where it has not (the input itself left out of the counts when it is a training graph)."""
        t = self.typer.types(instance)
        plus = np.zeros((self.size, self.size)); minus = np.zeros((self.size, self.size)); n_plus = n_minus = 0
        for y, M in self.counts.items():
            if y == int(f_E):
                plus = plus + M; n_plus += self.graphs[y]
            else:
                minus = minus + M; n_minus += self.graphs[y]
        if instance.id in self.train_ids:
            own = _pair_counts(instance, t, self.size)
            if int(instance.label) == int(f_E):
                plus = plus - own; n_plus -= 1
            else:
                minus = minus - own; n_minus -= 1
        if self.normalize:
            plus = plus / max(n_plus, 1); minus = minus / max(n_minus, 1)
        diff = (plus - minus)[np.ix_(t, t)]
        E = _binary_adjacency(instance.data)
        return np.where(E == 1, diff, -diff)


def type_pair_prior(dataset, fold_id: int, node_type: dict, normalize: bool = True) -> TypePairPrior:
    """The (cached, per process) type-pair prior of a dataset and fold."""
    key = (id(dataset), int(fold_id), _spec_key(node_type, normalize))
    with _cache_lock:
        hit = _type_cache.get(key)
    if hit is None:
        hit = TypePairPrior(dataset, fold_id, node_type, normalize)
        with _cache_lock:
            _type_cache[key] = hit
    return hit


def prior_weights(dataset, instance, f_E: int, params: dict, fold_id: int = -1) -> np.ndarray:
    """The weight matrix of the input under the prior a DFS or DBS config asks for: ``prior`` "index" (default,
    Abrate and Bonchi's, needs one graph size) or "type_pair" (with ``node_type`` and ``normalize``)."""
    prior = params.get("prior", "index")
    if prior == "index":
        D_plus, D_minus = compute_class_counts(dataset, f_E, exclude_instance_id=instance.id)
        return edge_weight_matrix(instance.data, D_plus, D_minus)
    if prior == "type_pair":
        return type_pair_prior(dataset, fold_id, params["node_type"], params.get("normalize", True)).weights(instance, f_E)
    raise ValueError(f"unknown prior {prior!r}")
