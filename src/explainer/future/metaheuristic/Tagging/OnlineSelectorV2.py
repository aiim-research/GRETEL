"""
OnlineNNEdgeSelectorV2: online edge-move proposal + learning for graph edit search (second version).

This is a rewrite of ``OnlineSelector.OnlineNNEdgeSelector``. The first version is kept untouched
for reproducibility of the thesis experiments. Differences, each tied to a documented weakness of
version 1 (see the design review of 2026-09-03):

Representation
--------------
* Structural features are fully vectorised with numpy on a per-snapshot basis. The current graph
  A_t = A XOR S is materialised once per solution snapshot (N is small, a few hundred nodes at most),
  and all pair features are obtained by fancy indexing on matrices (degrees, common neighbours,
  Adamic-Adar, resource allocation, 3-paths, distance buckets). This removes the Python loop of v1
  and makes it cheap to score the whole candidate set instead of a sampled pool.
* New pair features (link prediction heuristics that plain node embeddings cannot express):
  Adamic-Adar, resource allocation, number of length-3 paths, distance buckets (2, 3, farther) in base
  and current graph, and "toggle closes a triangle" / "toggle breaks the only short path" indicators.

Candidate generation
--------------------
* Additions enumerate the whole complement E+ \\ S whenever it fits in ``max_add_candidates``
  (always the case on the datasets used so far). Sampling with a focus bias only kicks in for very
  large graphs. Version 1 spun 500k random attempts per call on small graphs and only ever covered
  focus-incident pairs.

Scoring and exploration
-----------------------
* Each ScoreNet has a shared trunk and ``n_heads`` output heads trained on bootstrapped masks
  (Poisson(1) weights per example and head) plus a fixed randomised prior network per head
  (Osband et al. 2018). The heads give a cheap epistemic uncertainty estimate.
* Acquisition: Thompson sampling (one random head per snapshot, default) or UCB (mean + kappa*std).
  A small epsilon floor (uniform random order with prob ``exploration_prob``) is kept.
* The trial order is a Plackett-Luce sample obtained with the Gumbel-top-k trick on the
  temperature-scaled acquisition score. Its per-candidate log-softmax is recorded as the sampling
  log-probability (logQ) used to debias the listwise loss.
* Optional veto: candidates whose upper confidence bound of success is below ``veto_threshold``
  are pushed to the end of the trial order (never dropped), only once the buffer holds enough data.

Learning signal
---------------
* Soft labels. The search supplies y in [0,1] = probability mass the oracle puts on a class different
  from the original one after the move (1 when the label flips). Binary success is the special case.
* Pooling of per-edge logits into a move logit: ``sum_bias`` (additive threshold model, sum of
  logits plus a learned per-extra-edge bias), ``noisy_and`` (log prod sigma), or ``mean`` (v1).
* Listwise softmax cross-entropy over {accepted move} U {moves tried and failed in the same
  context}, with logQ correction (sampled softmax). Replaces the pairwise softplus margin of v1.
* Per-edge examples: single-edge moves are exact per-edge labels and are stored as such.
* Replay buffer (reservoir sampling, per mode) storing the already computed feature rows, so
  replay never needs the original graph. Persisted inside the checkpoint. Every update mixes the
  fresh examples with a replay batch.

Stability across instances
--------------------------
* AdamW with weight decay, L2 regularisation towards the *initial* parameters (L2-Init),
  shrink-and-perturb at checkpoint load, dormant-unit monitoring with ReDo-style recycling.

Checkpoint format is new (``version: 2``). Version 1 checkpoints are not loadable here.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Pooling = Literal["mean", "sum_bias", "noisy_and"]
Acquisition = Literal["thompson", "ucb", "mean", "random"]


# =============================================================================
# Network
# =============================================================================


class ScoreNetV2(nn.Module):
    """MLP trunk (LayerNorm + SiLU) with ``n_heads`` scalar heads and a fixed random prior per head.

    forward(x) -> (M, n_heads) logits. The prior network is a frozen copy with independent random
    weights whose output is scaled by ``prior_scale`` and added to the trainable output. Heads that
    disagree off-distribution therefore keep disagreeing, which is what makes ensemble disagreement a
    usable exploration signal.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int,
                 n_heads: int, prior_scale: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_heads = n_heads
        self.prior_scale = float(prior_scale)

        self.trunk = self._make_trunk(input_dim, hidden_dim, num_hidden_layers)
        self.heads = nn.Linear(hidden_dim, n_heads)
        # learned size bias for the additive (sum_bias) pooling, one per head
        self.size_bias = nn.Parameter(torch.full((n_heads,), -0.5))

        if self.prior_scale > 0:
            self.prior_trunk = self._make_trunk(input_dim, hidden_dim, num_hidden_layers)
            self.prior_heads = nn.Linear(hidden_dim, n_heads)
            for p in list(self.prior_trunk.parameters()) + list(self.prior_heads.parameters()):
                p.requires_grad_(False)
        else:
            self.prior_trunk = None
            self.prior_heads = None

    @staticmethod
    def _make_trunk(input_dim: int, hidden_dim: int, num_hidden_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        return nn.Sequential(*layers)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.heads(self.trunk(x))
        if self.prior_trunk is not None:
            with torch.no_grad():
                prior = self.prior_heads(self.prior_trunk(x))
            out = out + self.prior_scale * prior
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# =============================================================================
# Replay buffer
# =============================================================================


@dataclass
class MoveExample:
    feats: torch.Tensor          # (m, input_dim) float32 on CPU
    y: float                     # soft label in [0, 1]
    boot: torch.Tensor           # (n_heads,) bootstrap weights
    weight: float = 1.0


class ReservoirBuffer:
    """Reservoir-sampled buffer of MoveExample (Vitter's algorithm R)."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = int(capacity)
        self.items: list[MoveExample] = []
        self.seen = 0
        self.rng = rng

    def add(self, ex: MoveExample) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(ex)
            return
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.items[j] = ex

    def sample(self, n: int) -> list[MoveExample]:
        if not self.items:
            return []
        n = min(n, len(self.items))
        return self.rng.sample(self.items, n)

    def __len__(self) -> int:
        return len(self.items)

    def state(self) -> dict:
        return {
            "capacity": self.capacity,
            "seen": self.seen,
            "items": [(e.feats, float(e.y), e.boot, float(e.weight)) for e in self.items],
        }

    def load_state(self, st: dict) -> None:
        self.capacity = int(st.get("capacity", self.capacity))
        self.seen = int(st.get("seen", 0))
        self.items = [MoveExample(f, y, b, w) for (f, y, b, w) in st.get("items", [])]


# =============================================================================
# Snapshot context (current graph = base XOR solution)
# =============================================================================


@dataclass
class Snapshot:
    key: tuple
    S: np.ndarray            # (N, N) bool, disturbed pairs
    A_cur: np.ndarray        # (N, N) float32 adjacency of the current graph
    deg_cur: np.ndarray      # (N,)
    CN_cur: np.ndarray       # (N, N) common neighbours
    AA_cur: np.ndarray       # (N, N) Adamic-Adar
    RA_cur: np.ndarray       # (N, N) resource allocation
    P3_cur: np.ndarray       # (N, N) number of length-3 paths
    sol_deg: np.ndarray      # (N,) number of disturbed pairs incident to each node
    sol_size: int
    solution_set: set = field(default_factory=set)


# =============================================================================
# Selector
# =============================================================================


class OnlineNNEdgeSelectorV2:
    """Online neural edge selector, version 2. See module docstring."""

    STRUCT_DIM = 52
    CKPT_VERSION = 2

    def __init__(
        self,
        k: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        n_heads: int = 4,
        prior_scale: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        l2_init: float = 1e-2,
        exploration_prob: float = 0.05,
        acquisition: Acquisition = "thompson",
        ucb_kappa: float = 1.0,
        temp_add: float = 1.0,
        temp_remove: float = 1.0,
        pooling: Pooling = "sum_bias",
        listwise_temp: float = 1.0,
        logq_correction: bool = True,
        replay_capacity: int = 2000,
        replay_batch: int = 32,
        max_edges_per_move: int = 8,
        max_add_candidates: int = 20000,
        veto_threshold: float = 0.0,
        veto_min_examples: int = 500,
        dormant_threshold: float = 0.025,
        redo_every: int = 200,
        shrink: float = 0.97,
        perturb_std: float = 3e-3,
        lr_add: float | None = None,
        extra_replay_steps_add: int = 0,
        extra_replay_steps_remove: int = 0,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.k = int(k)
        self.input_dim = 4 * self.k + 4 + self.STRUCT_DIM
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.n_heads = int(n_heads)
        self.prior_scale = float(prior_scale)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.l2_init = float(l2_init)
        self.exploration_prob = float(exploration_prob)
        self.acquisition: Acquisition = acquisition
        self.ucb_kappa = float(ucb_kappa)
        self.temp_add = float(temp_add)
        self.temp_remove = float(temp_remove)
        self.pooling: Pooling = pooling
        self.listwise_temp = float(listwise_temp)
        self.logq_correction = bool(logq_correction)
        self.replay_batch = int(replay_batch)
        self.max_edges_per_move = int(max_edges_per_move)
        self.max_add_candidates = int(max_add_candidates)
        self.veto_threshold = float(veto_threshold)
        self.veto_min_examples = int(veto_min_examples)
        self.dormant_threshold = float(dormant_threshold)
        self.redo_every = int(redo_every)
        self.shrink = float(shrink)
        self.perturb_std = float(perturb_std)
        self.lr_add = float(lr_add) if lr_add is not None else None
        self.extra_replay_steps = {"add": int(extra_replay_steps_add), "remove": int(extra_replay_steps_remove)}

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.models: dict[str, ScoreNetV2] = {
            "add": ScoreNetV2(self.input_dim, self.hidden_dim, self.num_hidden_layers,
                              self.n_heads, self.prior_scale).to(self.device),
            "remove": ScoreNetV2(self.input_dim, self.hidden_dim, self.num_hidden_layers,
                                 self.n_heads, self.prior_scale).to(self.device),
        }
        self.opts = {m: torch.optim.AdamW(net.trainable_parameters(),
                                          lr=(self.lr_add if (m == "add" and self.lr_add) else self.lr),
                                          weight_decay=self.weight_decay)
                     for m, net in self.models.items()}
        # anchor for L2-Init (a copy of the trainable parameters at construction)
        self.init_params = {m: [p.detach().clone() for p in net.trainable_parameters()]
                            for m, net in self.models.items()}

        self.buffers = {m: ReservoirBuffer(replay_capacity, self.rng) for m in self.models}
        self.pending: dict[str, list[MoveExample]] = {m: [] for m in self.models}

        self.train_steps = {m: 0 for m in self.models}
        self.examples_seen = {m: 0 for m in self.models}
        self.instances_seen = 0

        # graph state
        self.node_vecs: torch.Tensor | None = None
        self.N = 0
        self._snapshot_cache: dict[tuple, Snapshot] = {}
        self._order_cache: dict[tuple, tuple[list, np.ndarray, np.ndarray]] = {}
        self._current_head = {m: 0 for m in self.models}

        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_stats: dict[str, float] = {}

    # ------------------------------------------------------------------ setup

    def set_node_vectors(self, node_vecs_np: np.ndarray) -> None:
        arr = np.asarray(node_vecs_np, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == self.k, (arr.shape, self.k)
        self.node_vecs = torch.as_tensor(arr, device=self.device)

    def set_base_graph(self, adj_np: np.ndarray, directed: bool = False) -> None:
        """Register the base graph and precompute the base-graph matrices."""
        A = (np.asarray(adj_np) != 0)
        if not directed:
            A = A | A.T
        np.fill_diagonal(A, False)
        self.N = int(A.shape[0])
        self.A_base_bool = A
        Af = A.astype(np.float32)
        self.A_base = Af
        self.deg_base = Af.sum(1)
        self.CN_base = Af @ Af
        self.AA_base, self.RA_base = self._aa_ra(Af, self.deg_base)
        self.P3_base = self.CN_base @ Af
        self.EPlus = self.N * (self.N - 1) // 2
        self.denom_deg = max(1.0, self.N - 1.0)
        self.denom_cn = max(1.0, self.N - 2.0)
        self.denom_p3 = max(1.0, (self.N - 2.0) * (self.N - 3.0))
        self._snapshot_cache.clear()
        self._order_cache.clear()
        self.instances_seen += 1
        # new instance: draw a fresh Thompson head for each mode
        for m in self.models:
            self._current_head[m] = self.rng.randrange(self.n_heads)

    @staticmethod
    def _aa_ra(Af: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w_aa = np.where(deg > 1.0, 1.0 / np.log(np.maximum(deg, 2.0)), 0.0).astype(np.float32)
        w_ra = np.where(deg > 0.0, 1.0 / np.maximum(deg, 1.0), 0.0).astype(np.float32)
        AA = (Af * w_aa[None, :]) @ Af
        RA = (Af * w_ra[None, :]) @ Af
        return AA, RA

    # -------------------------------------------------------------- snapshots

    @staticmethod
    def canonical(pairs) -> set[tuple[int, int]]:
        out = set()
        for (u, v) in pairs:
            u = int(u); v = int(v)
            if u == v:
                continue
            out.add((u, v) if u < v else (v, u))
        return out

    def _snapshot(self, solution_set: set[tuple[int, int]]) -> Snapshot:
        key = (len(solution_set), tuple(sorted(solution_set)))
        snap = self._snapshot_cache.get(key)
        if snap is not None:
            return snap
        N = self.N
        S = np.zeros((N, N), dtype=bool)
        if solution_set:
            idx = np.fromiter((x for p in solution_set for x in p), dtype=np.int64,
                              count=2 * len(solution_set)).reshape(-1, 2)
            S[idx[:, 0], idx[:, 1]] = True
            S[idx[:, 1], idx[:, 0]] = True
        A_cur = (self.A_base_bool ^ S).astype(np.float32)
        deg_cur = A_cur.sum(1)
        CN_cur = A_cur @ A_cur
        AA_cur, RA_cur = self._aa_ra(A_cur, deg_cur)
        P3_cur = CN_cur @ A_cur
        snap = Snapshot(key=key, S=S, A_cur=A_cur, deg_cur=deg_cur, CN_cur=CN_cur,
                        AA_cur=AA_cur, RA_cur=RA_cur, P3_cur=P3_cur,
                        sol_deg=S.sum(1).astype(np.float32), sol_size=len(solution_set),
                        solution_set=set(solution_set))
        if len(self._snapshot_cache) > 64:
            self._snapshot_cache.clear()
        self._snapshot_cache[key] = snap
        return snap

    # --------------------------------------------------------------- features

    def _pair_features(self, pairs: np.ndarray, snap: Snapshot) -> torch.Tensor:
        """Vectorised feature matrix (M, input_dim) for candidate pairs under a snapshot."""
        assert self.node_vecs is not None, "call set_node_vectors() first"
        u = pairs[:, 0]
        v = pairs[:, 1]
        eps = 1e-9

        # ---- block A / B: node vector combinations
        vu = self.node_vecs[torch.as_tensor(u, device=self.device)]
        vv = self.node_vecs[torch.as_tensor(v, device=self.device)]
        diff = vu - vv
        prod = vu * vv
        dot = prod.sum(1, keepdim=True)
        nu = vu.norm(dim=1, keepdim=True)
        nv = vv.norm(dim=1, keepdim=True)
        dist = diff.norm(dim=1, keepdim=True)

        # ---- block C: structure (numpy, vectorised)
        base_edge = self.A_base[u, v]
        curr_edge = snap.A_cur[u, v]
        disturbed = snap.S[u, v].astype(np.float32)

        dgu_b = self.deg_base[u]; dgv_b = self.deg_base[v]
        dgu_c = snap.deg_cur[u]; dgv_c = snap.deg_cur[v]
        cn_b = self.CN_base[u, v]; cn_c = snap.CN_cur[u, v]
        aa_b = self.AA_base[u, v]; aa_c = snap.AA_cur[u, v]
        ra_b = self.RA_base[u, v]; ra_c = snap.RA_cur[u, v]
        p3_b = self.P3_base[u, v]; p3_c = snap.P3_cur[u, v]

        sol_size = max(1.0, float(snap.sol_size))
        sdu = snap.sol_deg[u]; sdv = snap.sol_deg[v]

        def sim_block(cn, dgu, dgv):
            union = dgu + dgv - cn
            jacc = cn / (union + eps)
            cos = cn / (np.sqrt(dgu * dgv) + eps)
            dice = 2.0 * cn / (dgu + dgv + eps)
            overlap = cn / (np.minimum(dgu, dgv) + eps)
            cn_max = cn / (np.maximum(dgu, dgv) + eps)
            pref = (dgu * dgv) / (self.denom_deg ** 2 + eps)
            degdiff = np.abs(dgu - dgv) / self.denom_deg
            return jacc, cos, dice, overlap, cn_max, pref, degdiff

        jb, cb, db, ob, mb, pb, ddb = sim_block(cn_b, dgu_b, dgv_b)
        jc, cc, dc, oc, mc, pc, ddc = sim_block(cn_c, dgu_c, dgv_c)

        # distance buckets (1 = edge, 2 = common neighbour, 3 = a 3-path, far otherwise)
        def dist_buckets(edge, cn, p3):
            d2 = ((edge < 0.5) & (cn > 0)).astype(np.float32)
            d3 = ((edge < 0.5) & (cn <= 0) & (p3 > 0)).astype(np.float32)
            far = ((edge < 0.5) & (cn <= 0) & (p3 <= 0)).astype(np.float32)
            return d2, d3, far

        d2b, d3b, farb = dist_buckets(base_edge, cn_b, p3_b)
        d2c, d3c, farc = dist_buckets(curr_edge, cn_c, p3_c)

        # causal after-toggle block
        edge_after = 1.0 - curr_edge
        d_deg = (edge_after - curr_edge)                 # +1 or -1
        d_deg_n = d_deg / self.denom_deg
        dgu_a = (dgu_c + d_deg) / self.denom_deg
        dgv_a = (dgv_c + d_deg) / self.denom_deg
        d_sol = np.where(disturbed > 0.5, -1.0, 1.0)
        sol_after = np.maximum(1.0, snap.sol_size + d_sol)
        sdu_a = np.maximum(0.0, sdu + d_sol) / sol_after
        sdv_a = np.maximum(0.0, sdv + d_sol) / sol_after
        sdu_n = sdu / sol_size
        sdv_n = sdv / sol_size
        cn_c_n = cn_c / self.denom_cn
        tri_c = curr_edge * cn_c_n
        tri_a = edge_after * cn_c_n
        closes_tri = edge_after * (cn_c > 0).astype(np.float32)
        breaks_path = curr_edge * (cn_c <= 0).astype(np.float32)

        struct = np.stack([
            base_edge, curr_edge,
            (sdu > 0).astype(np.float32), (sdv > 0).astype(np.float32),
            dgu_b / self.denom_deg, dgv_b / self.denom_deg,
            dgu_c / self.denom_deg, dgv_c / self.denom_deg,
            sdu_n, sdv_n,
            cn_b / self.denom_cn, cn_c_n,
            jb, jc, cb, cc, db, dc, ob, oc, pb, pc, ddb, ddc, mb, mc,
            # after-toggle (12)
            edge_after, d_deg_n, d_deg_n, dgu_a, dgv_a, sdu_a, sdv_a,
            sdu_a - sdu_n, sdv_a - sdv_n, tri_c, tri_a, tri_a - tri_c,
            # new in v2 (14)
            aa_b / self.denom_cn, aa_c / self.denom_cn,
            ra_b / self.denom_cn, ra_c / self.denom_cn,
            p3_b / self.denom_p3, p3_c / self.denom_p3,
            d2b, d3b, farb, d2c, d3c, farc,
            closes_tri, breaks_path,
        ], axis=1).astype(np.float32)
        assert struct.shape[1] == self.STRUCT_DIM, struct.shape

        struct_t = torch.as_tensor(struct, device=self.device)
        feats = torch.cat([vu, vv, diff, prod, dot, nu, nv, dist, struct_t], dim=1)
        assert feats.shape[1] == self.input_dim, (feats.shape, self.input_dim)
        return feats

    # ------------------------------------------------------------- candidates

    def _add_candidates(self, snap: Snapshot) -> np.ndarray:
        """All pairs of E+ not in S when that fits the cap, otherwise a focus-biased sample."""
        N = self.N
        n_total = self.EPlus - snap.sol_size
        if n_total <= 0:
            return np.zeros((0, 2), dtype=np.int64)
        if n_total <= self.max_add_candidates:
            iu, ju = np.triu_indices(N, k=1)
            mask = ~snap.S[iu, ju]
            return np.stack([iu[mask], ju[mask]], axis=1).astype(np.int64)
        # large graph: half of the pool around focus nodes, half uniform
        pool = self.max_add_candidates
        focus = np.nonzero(snap.sol_deg > 0)[0]
        cand = set()
        attempts = 0
        while len(cand) < pool and attempts < 20 * pool:
            attempts += 1
            if focus.size > 0 and self.rng.random() < 0.5:
                a = int(focus[self.rng.randrange(focus.size)])
            else:
                a = self.rng.randrange(N)
            b = self.rng.randrange(N)
            if a == b:
                continue
            p = (a, b) if a < b else (b, a)
            if p in snap.solution_set:
                continue
            cand.add(p)
        return np.asarray(sorted(cand), dtype=np.int64).reshape(-1, 2)

    # ---------------------------------------------------------------- scoring

    @torch.no_grad()
    def _score(self, mode: str, pairs: np.ndarray, snap: Snapshot) -> np.ndarray:
        """(M, n_heads) logits."""
        if pairs.shape[0] == 0:
            return np.zeros((0, self.n_heads), dtype=np.float32)
        net = self.models[mode]
        net.eval()
        feats = self._pair_features(pairs, snap)
        return net(feats).cpu().numpy().astype(np.float32)

    def _acquire(self, mode: str, logits: np.ndarray) -> np.ndarray:
        """Collapse the head dimension into one acquisition score per candidate."""
        if self.acquisition == "random" or logits.shape[0] == 0:
            return np.zeros(logits.shape[0], dtype=np.float32)
        if self.acquisition == "thompson":
            return logits[:, self._current_head[mode]]
        mean = logits.mean(1)
        if self.acquisition == "mean":
            return mean
        std = logits.std(1) if logits.shape[1] > 1 else np.zeros_like(mean)
        return mean + self.ucb_kappa * std

    def _ucb_prob(self, logits: np.ndarray) -> np.ndarray:
        mean = logits.mean(1)
        std = logits.std(1) if logits.shape[1] > 1 else np.zeros_like(mean)
        return 1.0 / (1.0 + np.exp(-(mean + self.ucb_kappa * std)))

    def get_trial_order(self, mode: str, solution_pairs, *, seed: int | None = None):
        """Ranked (Plackett-Luce sampled) list of candidate pairs for ``mode`` under the snapshot.

        Returns (ordered_pairs, ordered_acq_scores, ordered_logq) where logq is the log-probability
        of each candidate being the first pick under the sampling distribution (used for logQ
        corrections in the listwise loss). Cached per (mode, snapshot, seed).
        """
        solution_set = self.canonical(solution_pairs)
        snap = self._snapshot(solution_set)
        ckey = (mode, snap.key, seed)
        hit = self._order_cache.get(ckey)
        if hit is not None:
            return hit

        if mode == "remove":
            pairs = np.asarray(sorted(solution_set), dtype=np.int64).reshape(-1, 2)
        elif mode == "add":
            pairs = self._add_candidates(snap)
        else:
            raise ValueError(mode)
        M = pairs.shape[0]
        if M == 0:
            res = ([], np.zeros(0, np.float32), np.zeros(0, np.float32))
            self._order_cache[ckey] = res
            return res

        if self.acquisition == "random":
            # ablation mode: same search skeleton, uniform random proposals, no scoring cost
            logits = np.zeros((M, self.n_heads), dtype=np.float32)
        else:
            logits = self._score(mode, pairs, snap)
        acq = self._acquire(mode, logits)
        T = max(1e-6, self.temp_add if mode == "add" else self.temp_remove)
        z = acq / T
        z = z - z.max()
        logq = z - np.log(np.exp(z).sum() + 1e-12)

        rng = np.random.default_rng(seed if seed is not None else self.np_rng.integers(1 << 31))
        if M > 1 and rng.random() < self.exploration_prob:
            keys = rng.random(M)                      # epsilon floor: uniform random order
        else:
            gumbel = -np.log(-np.log(rng.random(M) + 1e-12) + 1e-12)
            keys = z + gumbel                         # Gumbel-top-k = Plackett-Luce sample

        # optional veto: push confidently bad candidates to the tail (never drop them)
        if self.veto_threshold > 0 and len(self.buffers[mode]) >= self.veto_min_examples:
            ucb = self._ucb_prob(logits)
            bad = ucb < self.veto_threshold
            if bad.sum() < 0.5 * M:                    # never veto more than half a sweep
                keys = np.where(bad, keys - 1e6, keys)

        order = np.argsort(-keys)
        ordered_pairs = [(int(pairs[i, 0]), int(pairs[i, 1])) for i in order]
        res = (ordered_pairs, acq[order].astype(np.float32), logq[order].astype(np.float32))
        if len(self._order_cache) > 256:
            self._order_cache.clear()
        self._order_cache[ckey] = res
        self.last_stats.update({
            f"{mode}_num_candidates": float(M),
            f"{mode}_acq_mean": float(acq.mean()),
            f"{mode}_acq_std": float(acq.std()),
            f"{mode}_head_std_mean": float(logits.std(1).mean()) if logits.shape[1] > 1 else 0.0,
        })
        return res

    def propose(self, mode: str, solution_pairs, X: int) -> list[tuple[int, int]]:
        """First X pairs of a freshly sampled trial order (new Plackett-Luce draw each call)."""
        if X <= 0:
            return []
        order, _, _ = self.get_trial_order(mode, solution_pairs, seed=int(self.np_rng.integers(1 << 31)))
        return order[:min(X, len(order))]

    def propose_scored(self, mode: str, solution_pairs, X: int):
        """Like ``propose`` but also returns the mean acquisition logit of the proposed set
        (the pre-registered prediction used by the prequential monitor)."""
        if X <= 0:
            return [], 0.0
        order, acq, _ = self.get_trial_order(mode, solution_pairs, seed=int(self.np_rng.integers(1 << 31)))
        n = min(X, len(order))
        if n == 0:
            return [], 0.0
        return order[:n], float(np.mean(acq[:n]))

    def propose_additions(self, solution_pairs, X: int):
        return self.propose("add", solution_pairs, X)

    def propose_removals(self, solution_pairs, X: int):
        return self.propose("remove", solution_pairs, X)

    def new_head(self) -> None:
        """Redraw the Thompson head (call after each accepted move)."""
        for m in self.models:
            self._current_head[m] = self.rng.randrange(self.n_heads)

    # ------------------------------------------------------------------ pooling

    def _pool(self, logits: torch.Tensor, net: ScoreNetV2) -> torch.Tensor:
        """(m, H) per-edge logits -> (H,) move logits."""
        m = logits.shape[0]
        if self.pooling == "mean" or m == 1:
            return logits.mean(0)
        if self.pooling == "sum_bias":
            return logits.sum(0) + net.size_bias * float(m - 1)
        if self.pooling == "noisy_and":
            # P(success) = prod sigma(l_i); return its logit
            logp = F.logsigmoid(logits).sum(0)
            logp = torch.clamp(logp, max=-1e-6)
            return logp - torch.log(-torch.expm1(logp))
        raise ValueError(self.pooling)

    # ---------------------------------------------------------------- recording

    def _make_example(self, mode: str, solution_pairs, move_pairs, y: float,
                      weight: float = 1.0) -> MoveExample | None:
        move = sorted(self.canonical(move_pairs))
        if not move:
            return None
        if len(move) > self.max_edges_per_move:
            move = self.rng.sample(move, self.max_edges_per_move)
        snap = self._snapshot(self.canonical(solution_pairs))
        pairs = np.asarray(move, dtype=np.int64).reshape(-1, 2)
        with torch.no_grad():
            feats = self._pair_features(pairs, snap).detach().cpu()
        boot = torch.poisson(torch.ones(self.n_heads)).float()
        return MoveExample(feats=feats, y=float(np.clip(y, 0.0, 1.0)), boot=boot, weight=float(weight))

    def record(self, mode: str, solution_pairs, move_pairs, y: float, weight: float = 1.0) -> None:
        """Store one (context, move, soft label) example for BCE training and replay."""
        ex = self._make_example(mode, solution_pairs, move_pairs, y, weight)
        if ex is None:
            return
        self.pending[mode].append(ex)
        self.buffers[mode].add(ex)
        self.examples_seen[mode] += 1

    # ----------------------------------------------------------------- training

    def _l2_init_penalty(self, mode: str) -> torch.Tensor:
        net = self.models[mode]
        pen = torch.zeros((), device=self.device)
        for p, p0 in zip(net.trainable_parameters(), self.init_params[mode]):
            pen = pen + ((p - p0.to(p.device)) ** 2).sum()
        return pen

    def _bce_loss(self, mode: str, examples: list[MoveExample]) -> torch.Tensor | None:
        net = self.models[mode]
        losses = []
        for ex in examples:
            logits = net(ex.feats.to(self.device))          # (m, H)
            move_logit = self._pool(logits, net)             # (H,)
            target = torch.full_like(move_logit, ex.y)
            l = F.binary_cross_entropy_with_logits(move_logit, target, reduction="none")
            w = ex.boot.to(self.device) * ex.weight
            losses.append((l * w).sum() / max(1.0, float(w.sum().item())))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def train_step(self, mode: str) -> float | None:
        """One optimiser step on pending examples mixed with a replay batch. Returns the loss."""
        fresh = self.pending[mode]
        if not fresh:
            return None
        replay = self.buffers[mode].sample(self.replay_batch)
        # avoid replaying the exact fresh objects twice
        fresh_ids = {id(e) for e in fresh}
        replay = [e for e in replay if id(e) not in fresh_ids]
        batch = fresh + replay
        self.pending[mode] = []

        net = self.models[mode]
        opt = self.opts[mode]
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = self._bce_loss(mode, batch)
        if loss is None:
            return None
        loss = loss + self.l2_init * self._l2_init_penalty(mode)
        if not torch.isfinite(loss):
            self.logger.warning("[%s] non-finite loss, skipping step", mode)
            return None
        loss.backward()
        self._optimise(mode, net, opt, loss, "bce")
        if self.redo_every > 0 and self.train_steps[mode] % self.redo_every == 0:
            self._recycle_dormant(mode, batch)
        for _ in range(self.extra_replay_steps.get(mode, 0)):
            rb = self.buffers[mode].sample(self.replay_batch)
            if len(rb) < 4:
                break
            opt.zero_grad(set_to_none=True)
            l2 = self._bce_loss(mode, rb)
            if l2 is None:
                break
            l2 = l2 + self.l2_init * self._l2_init_penalty(mode)
            if not torch.isfinite(l2):
                break
            l2.backward()
            self._optimise(mode, net, opt, l2, "bce")
        return float(loss.item())

    def _optimise(self, mode: str, net: ScoreNetV2, opt, loss: torch.Tensor, kind: str) -> None:
        """Clip, step, and update the training-health statistics (loss / grad / update norm EMAs)."""
        params = net.trainable_parameters()
        before = [p.detach().clone() for p in params]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1.0))
        opt.step()
        with torch.no_grad():
            upd = math.sqrt(sum(float(((p - b) ** 2).sum()) for p, b in zip(params, before)))
        self.train_steps[mode] += 1
        self._order_cache.clear()
        val = float(loss.item())
        self.last_stats[f"{mode}_{kind}"] = val
        self._ema(f"{mode}_{kind}_ema", val)
        self._ema(f"{mode}_grad_norm_ema", grad_norm)
        self._ema(f"{mode}_update_norm_ema", upd)

    def _ema(self, key: str, val: float, beta: float = 0.9) -> None:
        if not math.isfinite(val):
            return
        old = self.last_stats.get(key)
        self.last_stats[key] = val if old is None else beta * old + (1 - beta) * val

    def update_listwise(self, mode: str, solution_pairs, pos_move, neg_moves,
                        neg_logq=None, pos_logq: float | None = None) -> float | None:
        """Softmax cross-entropy over {pos} U negs (same context), with optional logQ correction.

        ``neg_logq`` / ``pos_logq`` are the sampling log-probabilities recorded when the moves were
        proposed. Subtracting them from the scores turns the sampled softmax into an (approximately)
        unbiased estimate of the full softmax (Bengio and Senecal 2008, Wu et al. 2024).
        """
        negs = [self.canonical(m) for m in neg_moves]
        negs = [sorted(m) for m in negs if m]
        pos = sorted(self.canonical(pos_move))
        if not pos or not negs:
            return None
        snap = self._snapshot(self.canonical(solution_pairs))
        net = self.models[mode]
        opt = self.opts[mode]
        net.train()
        opt.zero_grad(set_to_none=True)

        def move_logit(move):
            mv = move if len(move) <= self.max_edges_per_move else self.rng.sample(move, self.max_edges_per_move)
            pairs = np.asarray(mv, dtype=np.int64).reshape(-1, 2)
            return self._pool(net(self._pair_features(pairs, snap)), net)   # (H,)

        scores = [move_logit(pos)] + [move_logit(m) for m in negs]
        S = torch.stack(scores, 0) / self.listwise_temp                     # (1+J, H)
        if self.logq_correction and neg_logq is not None:
            lq = [pos_logq if pos_logq is not None else float(np.mean(neg_logq))] + [float(x) for x in neg_logq]
            S = S - torch.as_tensor(lq, device=self.device, dtype=S.dtype).unsqueeze(1)
        logp = F.log_softmax(S, dim=0)
        loss = -logp[0].mean() + self.l2_init * self._l2_init_penalty(mode)
        if not torch.isfinite(loss):
            return None
        loss.backward()
        self._optimise(mode, net, opt, loss, "listwise")
        return float(loss.item())

    def flush(self) -> None:
        for m in self.models:
            self.train_step(m)

    # --------------------------------------------------------------- plasticity

    @torch.no_grad()
    def _recycle_dormant(self, mode: str, batch: list[MoveExample]) -> None:
        """ReDo-style recycling: units of the last hidden layer whose normalised activation is
        below ``dormant_threshold`` get their incoming weights re-initialised and outgoing zeroed."""
        if not batch:
            return
        net = self.models[mode]
        x = torch.cat([e.feats for e in batch], 0).to(self.device)
        h = net.hidden(x).abs().mean(0)                        # (hidden,)
        score = h / (h.mean() + 1e-8)
        dormant = torch.nonzero(score < self.dormant_threshold).flatten()
        frac = float(dormant.numel()) / float(h.numel())
        self.last_stats[f"{mode}_dormant_frac"] = frac
        if dormant.numel() == 0:
            return
        last_linear = [m for m in net.trunk if isinstance(m, nn.Linear)][-1]
        fan_in = last_linear.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        last_linear.weight[dormant] = torch.empty(dormant.numel(), fan_in, device=self.device).uniform_(-bound, bound)
        last_linear.bias[dormant] = 0.0
        net.heads.weight[:, dormant] = 0.0
        self.logger.info("[%s] recycled %d dormant units (%.1f%%)", mode, dormant.numel(), 100 * frac)

    @torch.no_grad()
    def shrink_and_perturb(self) -> None:
        """theta <- shrink * theta + N(0, perturb_std^2) on trainable parameters (Ash and Adams 2020)."""
        if self.shrink >= 1.0 and self.perturb_std <= 0.0:
            return
        for net in self.models.values():
            for p in net.trainable_parameters():
                p.mul_(self.shrink)
                if self.perturb_std > 0:
                    p.add_(torch.randn_like(p) * self.perturb_std)

    # ------------------------------------------------------------ save / load

    def hyperparams(self) -> dict:
        return {
            "k": self.k, "hidden_dim": self.hidden_dim, "num_hidden_layers": self.num_hidden_layers,
            "n_heads": self.n_heads, "prior_scale": self.prior_scale, "lr": self.lr,
            "weight_decay": self.weight_decay, "l2_init": self.l2_init,
            "exploration_prob": self.exploration_prob, "acquisition": self.acquisition,
            "ucb_kappa": self.ucb_kappa, "temp_add": self.temp_add, "temp_remove": self.temp_remove,
            "pooling": self.pooling, "listwise_temp": self.listwise_temp,
            "logq_correction": self.logq_correction, "replay_capacity": self.buffers["add"].capacity,
            "replay_batch": self.replay_batch, "max_edges_per_move": self.max_edges_per_move,
            "max_add_candidates": self.max_add_candidates, "veto_threshold": self.veto_threshold,
            "veto_min_examples": self.veto_min_examples, "dormant_threshold": self.dormant_threshold,
            "redo_every": self.redo_every, "shrink": self.shrink, "perturb_std": self.perturb_std,
            "lr_add": self.lr_add, "extra_replay_steps_add": self.extra_replay_steps["add"],
            "extra_replay_steps_remove": self.extra_replay_steps["remove"],
        }

    def save(self, path: str) -> None:
        ckpt = {
            "version": self.CKPT_VERSION,
            "hyperparams": self.hyperparams(),
            "models": {m: net.state_dict() for m, net in self.models.items()},
            "opts": {m: opt.state_dict() for m, opt in self.opts.items()},
            "init_params": {m: [p.cpu() for p in ps] for m, ps in self.init_params.items()},
            "buffers": {m: b.state() for m, b in self.buffers.items()},
            "train_steps": self.train_steps,
            "examples_seen": self.examples_seen,
            "instances_seen": self.instances_seen,
        }
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path: str, device: torch.device | None = None,
             overrides: dict | None = None, apply_shrink_perturb: bool = True) -> "OnlineNNEdgeSelectorV2":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("version") != cls.CKPT_VERSION:
            raise ValueError(f"checkpoint {path} is not a v{cls.CKPT_VERSION} selector checkpoint")
        hp = dict(ckpt["hyperparams"])
        # architecture comes from the checkpoint; everything else may be overridden
        arch_keys = {"k", "hidden_dim", "num_hidden_layers", "n_heads", "prior_scale"}
        for key, val in (overrides or {}).items():
            if key not in arch_keys:
                hp[key] = val
        obj = cls(device=device, **hp)
        for m, net in obj.models.items():
            net.load_state_dict(ckpt["models"][m])
        for m, opt in obj.opts.items():
            try:
                opt.load_state_dict(ckpt["opts"][m])
                for g in opt.param_groups:
                    g["lr"] = (obj.lr_add if (m == "add" and obj.lr_add) else obj.lr)
                    g["weight_decay"] = obj.weight_decay
            except Exception as e:  # pragma: no cover
                obj.logger.warning("could not restore optimiser state for %s: %s", m, e)
        obj.init_params = {m: [p.to(obj.device) for p in ps] for m, ps in ckpt["init_params"].items()}
        for m, b in obj.buffers.items():
            b.load_state(ckpt["buffers"][m])
        obj.train_steps = dict(ckpt.get("train_steps", obj.train_steps))
        obj.examples_seen = dict(ckpt.get("examples_seen", obj.examples_seen))
        obj.instances_seen = int(ckpt.get("instances_seen", 0))
        if apply_shrink_perturb:
            obj.shrink_and_perturb()
        return obj

    # ------------------------------------------------------------------ misc

    def stats_line(self) -> str:
        parts = [f"steps(add={self.train_steps['add']},rem={self.train_steps['remove']})",
                 f"buf(add={len(self.buffers['add'])},rem={len(self.buffers['remove'])})"]
        for k in sorted(self.last_stats):
            parts.append(f"{k}={self.last_stats[k]:.4g}")
        return " ".join(parts)
