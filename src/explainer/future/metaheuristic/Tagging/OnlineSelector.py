"""
OnlineNNEdgeSelector: online edge-move proposal + learning for graph edit search.

High-level idea
---------------
This module implements an *online* policy that proposes which edges to disturb in a graph-editing
local search. The policy is learned incrementally from
feedback about whether proposed moves were "good" (success) or "bad" (failure), and it is used
to bias future proposals.

Key modeling assumption: current graph = base graph XOR disturbs
--------------------------------------------------------------
We assume each instance provides a fixed *base/original* graph G_base, represented by an adjacency
matrix. The algorithm maintains a *solution* represented as a set of undirected edges S, where each
edge in S means "disturb this edge relative to the base graph".

For any pair (u, v):
  - If (u, v) exists in G_base and (u, v) is disturbed in S -> it is removed in the current graph.
  - If (u, v) does NOT exist in G_base and (u, v) is disturbed in S -> it is added in the current graph.
This is exactly the XOR relation:
    edge_current(u,v) = edge_base(u,v) XOR disturbed(u,v)

Because only edges in S are modified, many structural quantities of the current graph can be computed
cheaply by "correcting" base-graph quantities using only the disturb sets incident to nodes.

Two policies / two models
-------------------------
The class maintains two independent edge-scoring networks:
  - model_add: scores candidate edges to ADD (disturb into the solution)
  - model_remove: scores candidate edges to REMOVE (disturb out of the solution)

Both networks are ScoreNet MLPs that output a single logit per candidate edge. The logit is interpreted
as an (uncalibrated) "probability of success" after sigmoid.

Feature construction
--------------------
For a candidate edge (u, v), the model receives a concatenation of:

(1) Embedding-based features from precomputed node vectors:
    - v_u, v_v, (v_u - v_v), (v_u * v_v)         -> 4k features
    - dot(v_u, v_v), ||v_u||, ||v_v||, ||v_u-v_v|| -> 4 features

(2) Structural graph features computed for both base and current graph:
    - base edge existence, current edge existence
    - degrees in base/current, solution-degree (disturb-degree)
    - common neighbors in base/current (normalized)
    - similarity indices (Jaccard, cosine-like, Dice, overlap) in base/current
    - preferential attachment, degree difference, etc.
These structural features encode how the candidate edge relates to current graph topology and to the
edits already present in the solution.

Proposal mechanism (exploration/exploitation)
---------------------------------------------
Given a set of candidates:
  - Score candidates with the appropriate model -> logits
  - Convert logits to sampling weights using a temperature-scaled exp transform
  - With probability exploration_prob: choose uniformly at random
  - Otherwise:
      * if X == 1: choose argmax(logit) deterministically
      * else: sample without replacement using the weights

For additions, candidates are not enumerated globally (O(N^2)). Instead, a pool is sampled, optionally
biased so that one endpoint is from "focus nodes" that already appear in the current solution (a local
search heuristic that tends to concentrate changes around active regions).

Online training signals
-----------------------
Two training styles are provided:

1) Move-level BCE (supervised success/failure):
   - A move may disturb multiple edges.
   - Compute per-edge logits and mean-pool them into a single move logit.
   - Apply BCEWithLogitsLoss against target success ∈ {0,1}.
   - Optionally down-weight negative examples to stabilize training.

2) Pairwise ranked moves (margin-based):
   - Given a positive move and several negative alternatives, enforce:
       score(pos) >= score(neg) + margin
   - Uses softplus(margin + neg - pos), a smooth hinge-like objective.

Practical safeguards
--------------------
  - BatchNorm is skipped for batch_size == 1 inside ScoreNet to avoid degenerate statistics.
  - Gradients are clipped to max_norm=1.0 for stability.
  - Negative edges are exposure-capped so repeated failures do not dominate learning.
"""

from __future__ import annotations
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import math
import torch.nn.functional as F
from collections import defaultdict


from typing import Literal
import torch
import torch.nn as nn


ActivationName = Literal["ReLU", "LeakyReLU", "ELU", "Swish"]


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")

NormName = Literal["none", "layernorm"]  # keep simple; LayerNorm works great for tiny batches


class ScoreNet(nn.Module):
    """
    Small MLP -> single logit.

    Online-friendly defaults:
      - LayerNorm instead of BatchNorm (batch-size agnostic)
      - Dropout disabled by default (avoid extra stochasticity on noisy feedback)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        activation_function: ActivationName = "Swish",
        norm: NormName = "layernorm",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        # activation
        if activation_function == "ReLU":
            act: nn.Module = nn.ReLU()
        elif activation_function == "LeakyReLU":
            act = nn.LeakyReLU()
        elif activation_function == "ELU":
            act = nn.ELU()
        elif activation_function == "Swish":
            act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")

        def make_norm(dim: int) -> nn.Module:
            if norm == "none":
                return nn.Identity()
            if norm == "layernorm":
                return nn.LayerNorm(dim)
            raise ValueError(f"Unknown norm: {norm}")

        layers: list[nn.Module] = []
        dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(make_norm(hidden_dim))
            layers.append(act)
            if dropout_prob and dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
         
class OnlineNNEdgeSelector:
    """
    Online neural model to bias which edges to add/remove.

    Conceptually:
      - Given a *current solution* (a set of disturbed edges), build candidate edge moves.
      - Score each candidate with an MLP using:
          (a) node embeddings features (u, v, diff, prod, dot, norms, dist)
          (b) structural graph features under:
              - base graph (original instance)
              - current graph (base XOR disturbs induced by current solution)
      - Sample actions with an exploration/exploitation policy.

    Usage:
        selector = OnlineNNEdgeSelector(k=node_vecs.shape[1])
        selector.set_node_vectors(node_vecs)  # np.ndarray (num_nodes, k)
        selector.set_base_graph(adj_np)

        chosen_add_pairs = selector.propose_additions(solution_pairs, X)
        selector.update_addition_moves(examples)  # online supervised signal (success/failure)
    """

    def __init__(
        self,
        k: int,  # dimension of node vectors
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        lr: float = 1e-3,
        exploration_prob: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        self.example_count = 0
        self.k = k
        # Original structural features (26) + causal after/delta block (12) = 38
        self.struct_dim = 38
        # input features = [u, v, u-v, u*v] (4k) + [dot,norm_u,norm_v,dist] (4) + struct (26)
        self.input_dim = 4 * k + 4 + self.struct_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.exploration_prob = exploration_prob
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model_add = ScoreNet(
            self.input_dim,
            hidden_dim,
            num_hidden_layers,
            activation_function="Swish",
            norm="layernorm",
            dropout_prob=0.0,
        ).to(self.device)

        self.model_remove = ScoreNet(
            self.input_dim,
            hidden_dim,
            num_hidden_layers,
            activation_function="Swish",
            norm="layernorm",
            dropout_prob=0.0,
        ).to(self.device)

        self.opt_add = optim.Adam(self.model_add.parameters(), lr=lr)
        self.opt_remove = optim.Adam(self.model_remove.parameters(), lr=lr)

        self.node_vecs: torch.Tensor | None = None  # will be set by set_node_vectors()
        self.node_vecs_np: np.ndarray | None = None

        # Track performance for dynamic exploration adjustment
        self.performance_counter = 0
        self.last_best_size = None
        
        # Temperatures used in a softmax-like weighting of logits (see _weights_from_logits).
        # Higher temperature => flatter distribution (more uniform sampling).
        self.temp_add = 1.5      
        self.temp_remove = 1.2
        
        # Repeated-negative cap:
        # Prevents training from being dominated by the same failing edges over and over.
        self.neg_edge_counts_add = defaultdict(int)
        self.neg_edge_counts_remove = defaultdict(int)
        self.neg_edge_cap = 5  # tuneable
        
        # Deterministic greedy shortcuts when X is small to reduce variance.
        self.greedy_top1 = True          # if X==1 and not exploring -> argmax
        self.greedy_topk = True          # if X small and not exploring -> take top-k
        self.greedy_k_threshold = 3      # only do deterministic top-k when X<=this
        
        # ---------------- Logging / stats ----------------
        self.logger = logging.getLogger(self.__class__.__name__)

        self.log_every_propose_steps = 50

        # counters
        self.train_calls = 0
        self.train_calls_add = 0
        self.train_calls_remove = 0
        self.propose_calls = 0

        # moving averages (EMA)
        self.ema_beta = 0.95
        self._ema: dict[str, float] = {}  # name -> float

        # store last events
        self._last_train: dict[str, object] = {}
        self._last_propose: dict[str, object] = {}

        
        self._use_cuda_timing = (self.device.type == "cuda")

        # Use per-sample loss so we can log extra stats
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        
    # ---------- Node vectors & features ----------

    def build_tensors(self, node_vecs_np: np.ndarray) -> torch.Tensor:
        """
        Convert node vectors to a device tensor.

        Args:
            node_vecs_np: numpy array of shape (num_nodes, k)

        Returns:
            Tensor on self.device of shape (num_nodes, k), dtype float32.
        """
        return torch.as_tensor(
            node_vecs_np, dtype=torch.float32, device=self.device
        )
        
    def set_node_vectors(self, node_vecs_np: np.ndarray) -> None:
        """
        Store node vectors as a tensor (detached from any graph).

        Args:
            node_vecs_np: numpy array of shape (num_nodes, k)
        """
        assert node_vecs_np.shape[1] == self.k
        self.node_vecs = torch.as_tensor(
            node_vecs_np, dtype=torch.float32, device=self.device
        )
        
    def set_node_vectors_from_tensor(self, node_vecs_list: list[torch.Tensor]) -> None:
        """
        Alternative setter when node vectors are already torch tensors.

        This explicitly detaches the vectors (no gradients tracked), since the selector
        treats node embeddings as fixed features (not jointly trained here).

        Args:
            node_vecs_list: list length num_nodes; each tensor shape (k,)
        """
        # Detach from computation graph and stack
        with torch.no_grad():
            node_vecs = torch.stack(
                [v.detach() for v in node_vecs_list], dim=0
            )  # (num_nodes, k)

            assert node_vecs.shape[1] == self.k

            self.node_vecs = node_vecs.to(self.device, dtype=torch.float32)

    def _pair_features(
        self,
        pairs_tensor: torch.Tensor,
        disturbed_neighbors: dict[int, set[int]],
        solution_size: int
    ) -> torch.Tensor:
        """
        Build feature matrix for a batch of candidate edges.

        Key theoretical piece:
          - The "current graph" is modeled as:  current = base XOR disturbs
            where `disturbs` come from the current solution edge set.
          - Structural features are computed for both base and current graph:
              edge existence, degrees, common neighbors, similarity indices, etc.

        Args:
            pairs_tensor: LongTensor of shape (M, 2) with node indices (u, v).
            disturbed_neighbors: adjacency-like structure for the current solution disturbs:
                              node -> set of nodes whose incident edge is disturbed.
            solution_size: number of disturbed edges in the current solution.

        Returns:
            FloatTensor of shape (M, input_dim).
        """
        assert self.node_vecs is not None, "Call set_node_vectors() first."
        assert hasattr(self, "base_neighbors"), "Call set_base_graph() first."
        assert hasattr(self, "num_nodes"), "Call set_base_graph() first."

        u_idx = pairs_tensor[:, 0]
        v_idx = pairs_tensor[:, 1]

        v_u = self.node_vecs[u_idx]  # (M,k)
        v_v = self.node_vecs[v_idx]  # (M,k)

        # Standard pairwise embedding features:
        # - difference captures directionality in feature space (even if edge is undirected)
        # - product captures "agreement"/interaction (like factorization machines)
        # - dot/norm/dist are scalar geometry summaries
        diff = v_u - v_v
        prod = v_u * v_v
        dot = (v_u * v_v).sum(dim=1, keepdim=True)
        norm_u = v_u.norm(dim=1, keepdim=True)
        norm_v = v_v.norm(dim=1, keepdim=True)
        dist = diff.norm(dim=1, keepdim=True)

        # --- structural features (python loop; M is small/moderate) ---
        # Structural features are computed with set operations and "disturb corrections".
        # This part is deliberately in Python for clarity; could be vectorized later.
        M = pairs_tensor.size(0)
        eps = 1e-9
        N = float(self.num_nodes)
        denom_deg = max(1.0, N - 1.0)
        denom_cn  = max(1.0, N - 2.0)
        sol_size = max(1, int(solution_size))

        struct_rows: list[list[float]] = []
        pairs_cpu = pairs_tensor.detach().cpu().numpy()

        for (u, v) in pairs_cpu:
            u = int(u); v = int(v)
            tog_u = disturbed_neighbors.get(u, set())
            tog_v = disturbed_neighbors.get(v, set())

            base_edge = 1.0 if self.base_adj_bool[u, v] else 0.0

            # Theoretical note (XOR model):
            # - A solution edge means "disturb this base edge": if it existed, remove it; if not, add it.
            # - So current_edge = base_edge XOR disturbed(u,v).
            # For undirected, checking v in tog_u is enough given symmetric insertion above.
            pair_disturbed = (v in tog_u)
            curr_edge = 1.0 if (base_edge > 0.5) ^ pair_disturbed else 0.0
            
            

            # Base degrees normalized by (N-1)
            deg_u_base = float(self.base_deg[u]) / denom_deg
            deg_v_base = float(self.base_deg[v]) / denom_deg

            # Current degrees incorporate disturbs incident to each node.
            # This is computed cheaply as base_deg + (added - removed) using membership in base neighbors.
            deg_u_curr_raw = float(self._deg_current(u, tog_u))
            deg_v_curr_raw = float(self._deg_current(v, tog_v))

            deg_u_curr = deg_u_curr_raw / denom_deg
            deg_v_curr = deg_v_curr_raw / denom_deg

            # "solution degree" = number of disturbed incident edges for this node,
            # normalized by current solution size.
            sol_deg_u = float(len(tog_u)) / float(sol_size)
            sol_deg_v = float(len(tog_v)) / float(sol_size)

            # Common neighbors in base and current graphs.
            # cn_curr computed by correcting cn_base only over nodes whose adjacency changed (tog_u ∪ tog_v).
            cn_base, cn_curr = self._cn_current(u, v, tog_u, tog_v)

            u_in_sol = 1.0 if len(tog_u) > 0 else 0.0
            v_in_sol = 1.0 if len(tog_v) > 0 else 0.0

            deg_u_base_raw = float(self.base_deg[u])
            deg_v_base_raw = float(self.base_deg[v])

            # normalized cn
            cn_base_n = float(cn_base) / denom_cn
            cn_curr_n = float(cn_curr) / denom_cn

            # jaccard = cn / (deg_u + deg_v - cn)
            union_base = (deg_u_base_raw + deg_v_base_raw - float(cn_base))
            jacc_base = float(cn_base) / (union_base + eps)

            union_curr = (deg_u_curr_raw + deg_v_curr_raw - float(cn_curr))
            jacc_curr = float(cn_curr) / (union_curr + eps)

            # cosine-like overlap: cn / sqrt(deg_u*deg_v)
            cos_base = float(cn_base) / (math.sqrt(deg_u_base_raw * deg_v_base_raw) + eps)
            cos_curr = float(cn_curr) / (math.sqrt(deg_u_curr_raw * deg_v_curr_raw) + eps)

            # dice: 2cn / (deg_u+deg_v)
            dice_base = (2.0 * float(cn_base)) / (deg_u_base_raw + deg_v_base_raw + eps)
            dice_curr = (2.0 * float(cn_curr)) / (deg_u_curr_raw + deg_v_curr_raw + eps)

            # overlap coefficient: cn / min(deg_u,deg_v)
            overlap_base = float(cn_base) / (min(deg_u_base_raw, deg_v_base_raw) + eps)
            overlap_curr = float(cn_curr) / (min(deg_u_curr_raw, deg_v_curr_raw) + eps)

            # cn / max(deg_u,deg_v)
            cn_over_max_base = float(cn_base) / (max(deg_u_base_raw, deg_v_base_raw) + eps)
            cn_over_max_curr = float(cn_curr) / (max(deg_u_curr_raw, deg_v_curr_raw) + eps)

            # preferential attachment (normalized product)
            pref_base_n = (deg_u_base_raw * deg_v_base_raw) / (denom_deg * denom_deg + eps)
            pref_curr_n = (deg_u_curr_raw * deg_v_curr_raw) / (denom_deg * denom_deg + eps)

            # degree difference (normalized)
            degdiff_base_n = abs(deg_u_base_raw - deg_v_base_raw) / denom_deg
            degdiff_curr_n = abs(deg_u_curr_raw - deg_v_curr_raw) / denom_deg
            
            # ---------------- Causal "after-toggle" features ----------------
            # Toggling disturbed(u,v) always flips the current edge existence:
            # edge_after = base XOR (disturbed flipped) = 1 - curr_edge
            edge_after = 1.0 - curr_edge
            delta_edge = edge_after - curr_edge  # +1 if edge added to current graph, -1 if removed

            # Degree change in the *current* graph for both endpoints equals whether the edge appears/disappears.
            # (Adding edge increases deg by 1; removing decreases by 1.)
            d_deg_raw = int(delta_edge)  # +1 or -1
            deg_u_after_raw = deg_u_curr_raw + d_deg_raw
            deg_v_after_raw = deg_v_curr_raw + d_deg_raw

            # normalized degree deltas and after-values
            d_deg_u_n = float(d_deg_raw) / denom_deg
            d_deg_v_n = float(d_deg_raw) / denom_deg
            deg_u_after = float(deg_u_after_raw) / denom_deg
            deg_v_after = float(deg_v_after_raw) / denom_deg

            # Solution-size / solution-degree after toggling the disturbed edge
            # If pair not disturbed -> toggle IN (size +1, incident disturbed deg +1)
            # If pair disturbed     -> toggle OUT (size -1, incident disturbed deg -1)
            d_sol_size = 1 if not pair_disturbed else -1
            sol_size_after = max(1, int(solution_size) + d_sol_size)

            len_tog_u_after = max(0, len(tog_u) + d_sol_size)
            len_tog_v_after = max(0, len(tog_v) + d_sol_size)

            sol_deg_u_after = float(len_tog_u_after) / float(sol_size_after)
            sol_deg_v_after = float(len_tog_v_after) / float(sol_size_after)

            d_sol_deg_u = sol_deg_u_after - sol_deg_u
            d_sol_deg_v = sol_deg_v_after - sol_deg_v

            # Cheap causal proxy for Δtriangles involving (u,v):
            # triangles_on_edge = edge_exists * cn_curr
            # cn_curr does NOT change when toggling (u,v) itself, but triangles appear/disappear with the edge.
            tri_curr_n  = curr_edge * cn_curr_n
            tri_after_n = edge_after * cn_curr_n
            d_tri_n     = tri_after_n - tri_curr_n
            # ----------------------------------------------------------------

            struct_rows.append([
                base_edge,
                curr_edge,

                u_in_sol, v_in_sol,

                deg_u_base, deg_v_base,
                deg_u_curr, deg_v_curr,

                sol_deg_u, sol_deg_v,

                cn_base_n, cn_curr_n,
                jacc_base, jacc_curr,

                cos_base, cos_curr,
                dice_base, dice_curr,
                overlap_base, overlap_curr,

                pref_base_n, pref_curr_n,
                degdiff_base_n, degdiff_curr_n,

                cn_over_max_base, cn_over_max_curr,

                # causal after-toggle + deltas (12) ----
                edge_after,
                d_deg_u_n, d_deg_v_n,
                deg_u_after, deg_v_after,
                sol_deg_u_after, sol_deg_v_after, 
                d_sol_deg_u, d_sol_deg_v,
                tri_curr_n, tri_after_n, d_tri_n,
            ])

        struct = torch.as_tensor(struct_rows, dtype=torch.float32, device=self.device)  # (M,struct_dim)

        feats = torch.cat(
            [v_u, v_v, diff, prod, dot, norm_u, norm_v, dist, struct],
            dim=1
        )
        assert feats.shape[1] == self.input_dim, (feats.shape, self.input_dim)
        return feats

    # ---------- Sampling logic ----------

    def _sample_indices(self, weights_np: np.ndarray, X: int) -> list[int]:
        """
        Sample indices from a non-negative weight vector.

        Notes:
          - This method includes epsilon-random exploration (pure random sampling).
          - If not exploring, it supports:
              * greedy top-1 if X==1
              * greedy top-k if X is small
              * otherwise weighted sampling without replacement

        Args:
            weights_np: array shape (M,), non-negative (doesn't need to sum to 1).
            X: number of indices to sample.

        Returns:
            List of selected indices (length X, unless M<=X).
        """
        n = len(weights_np)
        if n <= X:
            self._last_propose["did_explore"] = False
            return list(range(n))

        # epsilon exploration (pure random)
        if np.random.rand() < self.exploration_prob:
            self._last_propose["did_explore"] = True
            return random.sample(range(n), X)

        self._last_propose["did_explore"] = False

        w = np.asarray(weights_np, dtype=float)
        w[~np.isfinite(w)] = 0.0
        w = np.maximum(w, 0.0)

        # Greedy path (Efficiency-first)
        if self.greedy_top1 and X == 1:
            return [int(np.argmax(w))]

        if self.greedy_topk and X <= self.greedy_k_threshold:
            # deterministic top-k
            idx = np.argsort(-w)[:X]
            return idx.tolist()

        # Otherwise weighted sampling
        eps = 1e-12
        w[w < eps] = eps
        s = w.sum()
        if s <= 0.0:
            return random.sample(range(n), X)

        p = w / s
        idx = np.random.choice(np.arange(n), size=X, replace=False, p=p)
        return idx.tolist()
    
    def _sample_indices_no_explore(self, probs_np: np.ndarray, X: int) -> list[int]:
        """
        Sample indices without epsilon-random exploration.

        This is used in the newer proposal path where the exploration decision is made
        outside, and this function purely performs probability-weighted sampling.

        Args:
            probs_np: array shape (M,), non-negative.
            X: number of indices to sample.

        Returns:
            List of selected indices (length X, unless M<=X).
        """
        n = len(probs_np)
        if n <= X:
            return list(range(n))

        probs = np.asarray(probs_np, dtype=float)
        probs = np.maximum(probs, 0.0)

        # If too few non-zero entries, fall back to uniform sample
        if np.count_nonzero(probs) < X:
            return random.sample(range(n), X)

        eps = 1e-12
        probs[probs < eps] = eps
        probs = probs / probs.sum()

        idx = np.random.choice(np.arange(n), size=X, replace=False, p=probs)
        return idx.tolist()

    # ---------- Propose pairs ----------

    def _propose(self, solution_pairs: list[tuple[int, int]], X: int, mode: str) -> list[tuple[int, int]]:
        """
        Internal: propose X edge pairs for a given mode ('add' or 'remove').

        Theoretical overview:
          - Maintain solution_set = disturbed edges (canonical undirected pairs).
          - For 'remove': candidates are exactly the current disturbed edges.
          - For 'add': candidates are sampled from non-solution edges, biased toward
            nodes already present in the solution (focus sampling).
          - Score each candidate with a model (add/remove) to produce logits.
          - Convert logits to sampling weights (temperature-scaled softmax-like).
          - Apply exploration/exploitation:
              * with probability exploration_prob: uniform random sample
              * else:
                  - if X==1: greedy argmax (lowest variance)
                  - else: weighted sampling w/out replacement

        Args:
            solution_pairs: current solution as list of (u,v) edges (undirected assumed).
            X: number of pairs to propose.
            mode: "add" or "remove".

        Returns:
            List of proposed pairs (canonical (min,max) ordering).
        """
        if X <= 0:
            return []

        if self.node_vecs is None:
            raise RuntimeError("set_node_vectors() must be called before _propose().")

        num_nodes = self.node_vecs.shape[0]

        # Canonicalize and deduplicate the current solution as a set of (min(u,v), max(u,v))
        solution_set = {
            (int(min(u, v)), int(max(u, v)))
            for (u, v) in solution_pairs
        }
        
        # Build per-node "disturbed neighbor sets" for the current solution.
        disturbed_neighbors = defaultdict(set)
        for (a, b) in solution_set:
            disturbed_neighbors[a].add(b)
            disturbed_neighbors[b].add(a)  # undirected assumption
        solution_size = len(solution_set)

        

        if mode == "remove":
            # Candidates are exactly the edges in the current solution
            candidate_pairs = list(solution_set)

        elif mode == "add":
            # Candidates are sampled (pool) pairs NOT in the current solution
            pool_size = getattr(self, "add_pool_size", None)
            if pool_size is None:
                # default: scale with X a bit, but cap
                pool_size = int(min(20000, max(2000, 500 * X)))

            # Focus around nodes already involved in the current solution:
            # idea: local moves around recently-changed structure are more informative/useful.
            focus_nodes = []
            for (u, v) in solution_set:
                focus_nodes.append(u)
                focus_nodes.append(v)

            candidate_pairs = self._sample_add_candidates(
                solution_set=solution_set,
                num_nodes=num_nodes,
                pool_size=pool_size,
                focus_nodes=focus_nodes,   # comment this out to use uniform only
            )

        else:
            raise ValueError(f"Unknown mode '{mode}', expected 'add' or 'remove'.")

        # If there are no candidates, return empty
        if not candidate_pairs:
            return []

        M = len(candidate_pairs)
        if M <= X:
            return list(candidate_pairs)

        # Score candidates with the appropriate model
        pairs_tensor = torch.as_tensor(candidate_pairs, dtype=torch.long, device=self.device)

        model = self.model_add if mode == "add" else self.model_remove
        model.eval()
        with torch.no_grad():
            feats = self._pair_features(pairs_tensor, disturbed_neighbors=disturbed_neighbors, solution_size=solution_size) # (M, input_dim)
            logits = model(feats)                              # (M,)
            probs = torch.sigmoid(logits).cpu().numpy()        # P(success)
        
        # ---- save propose stats ----
        if M >= 2:
            top2 = np.partition(probs, -2)[-2:]
            p_gap = float(np.max(top2) - np.min(top2))
        else:
            p_gap = 0.0

        self.propose_calls += 1
        self._last_propose = {
            "propose_calls": self.propose_calls,
            "mode": mode,
            "num_candidates": M,
            "X": X,
            "exploration_prob": float(self.exploration_prob),
            "p_mean": float(np.mean(probs)),
            "p_std": float(np.std(probs)),
            "p_min": float(np.min(probs)),
            "p_max": float(np.max(probs)),
            "p_gap_top2": p_gap,
        }

        if (self.propose_calls % self.log_every_propose_steps) == 0:
            self._log_propose_status()
            
        logits_np = logits.detach().cpu().numpy()
        weights = self._weights_from_logits(logits_np, mode=mode)

        # store scored candidates for rank training (Fix 4)
        self._last_scored = getattr(self, "_last_scored", {})
        self._last_scored[mode] = {
            "candidate_pairs": candidate_pairs,
            "logits": logits_np,
        }

        # Decide exploration HERE (not in _sample_indices)
        did_explore = (np.random.rand() < self.exploration_prob)
        self._last_propose["did_explore"] = bool(did_explore)

        if did_explore:
            idx = random.sample(range(M), X)
        else:
            if X == 1:
                # Greedy argmax when not exploring: reduces noise and stabilizes search.
                idx = [int(np.argmax(logits_np))]
            else:
                idx = self._sample_indices_no_explore(weights, X)

        return [candidate_pairs[i] for i in idx]
    
    def _canonical_pair(self, u: int, v: int) -> tuple[int, int] | None:
        """
        Canonicalize an undirected pair.

        Args:
            u: node id
            v: node id

        Returns:
            (min(u,v), max(u,v)) or None if u==v.
        """
        if u == v:
            return None
        a, b = (u, v) if u < v else (v, u)
        return (a, b)
    
    def _weights_from_logits(self, logits: np.ndarray, mode: str) -> np.ndarray:
        """
        Convert logits to sampling weights via a temperature-scaled softmax-like transform.

        Theoretical note:
          - Using exp(logit / T) is equivalent to softmax up to normalization.
          - Lower T => sharper distribution (more greedy).
          - Higher T => flatter distribution (more exploratory without randomness).

        Args:
            logits: array shape (M,)
            mode: "add" or "remove" (selects temperature)

        Returns:
            Probability vector shape (M,) that sums to 1.
        """
        T = float(self.temp_add if mode == "add" else self.temp_remove)
        T = max(1e-6, T)

        z = logits / T
        z = z - np.max(z)
        w = np.exp(z)

        # avoid all-zero / nan
        w[~np.isfinite(w)] = 0.0
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w, dtype=float)
            s = w.sum()
        return w / s

    def _sample_add_candidates(
        self,
        solution_set: set[tuple[int, int]],
        num_nodes: int,
        pool_size: int,
        focus_nodes: list[int] | None = None,
        max_attempts_mult: int = 50,
    ) -> list[tuple[int, int]]:
        """
        Sample up to pool_size candidate edges (u,v) not in solution_set.

        Theoretical note:
          - For 'add' moves, enumerating all non-edges is O(N^2). Instead, sample a pool.
          - If focus_nodes is provided, enforce that one endpoint is in focus_nodes
            (local search heuristic around active nodes).

        Args:
            solution_set: canonical set of currently disturbed edges.
            num_nodes: number of nodes in the graph.
            pool_size: target number of sampled candidates.
            focus_nodes: optional list of nodes to bias sampling.
            max_attempts_mult: attempts budget multiplier to avoid infinite loops.

        Returns:
            List of unique canonical pairs not in solution_set, size <= pool_size.
        """
        if pool_size <= 0:
            return []

        # If focus_nodes is small/empty, fall back to uniform sampling
        use_focus = focus_nodes is not None and len(focus_nodes) > 0
        if use_focus:
            # remove invalid nodes and deduplicate
            focus = [n for n in set(map(int, focus_nodes)) if 0 <= n < num_nodes]
            use_focus = len(focus) > 0
        else:
            focus = []

        sampled: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = pool_size * max_attempts_mult

        while len(sampled) < pool_size and attempts < max_attempts:
            attempts += 1

            if use_focus:
                u = random.choice(focus)
                v = random.randrange(num_nodes)
            else:
                u = random.randrange(num_nodes)
                v = random.randrange(num_nodes)

            p = self._canonical_pair(u, v)
            if p is None:
                continue
            if p in solution_set:
                continue
            sampled.add(p)

        return list(sampled)

    def propose_additions(self, solution: list[tuple[int, int]], X: int) -> list[tuple[int, int]]:
        """
        Propose X candidate pairs to ADD (i.e., disturb into the solution).

        Args:
            solution: current solution edges as list of (u,v).
            X: number of proposals.

        Returns:
            List of proposed edge pairs (canonical ordering).
        """
        
        return self._propose(solution, X, mode="add")

    def propose_removals(self, solution: list[tuple[int, int]], X: int) -> list[tuple[int, int]]:
        """
        Propose X candidate pairs to REMOVE (i.e., disturb out of the solution).

        Args:
            solution: current solution edges as list of (u,v).
            X: number of proposals.

        Returns:
            List of proposed edge pairs (canonical ordering).
        """
        
        return self._propose(solution, X, mode="remove")
    
    
    

    # ---------- training ----------

    def _train_online_moves(
        self,
        mode: str,
        examples: list[tuple[list[tuple[int,int]], list[tuple[int,int]], bool]],
        neg_weight: float = 0.35,
        max_edges_per_move: int = 15,
        hard_within_move_for_neg: bool = True,
    ) -> None:
        """
        Online supervised training on *moves* labeled success/failure.

        Key theoretical choice:
          - A move can involve multiple edges; the model is edge-scoring.
          - The move is represented by mean-pooling the edge logits.
          - Train with BCE on the pooled logit (success=1, failure=0).
          - Failures are down-weighted (neg_weight) to reduce destabilizing gradients.
          - When a move includes many edges, subsample edges to control compute.
            For negative moves, optionally choose "hard" edges (those the model currently
            thinks are likely) to improve discrimination.

        Args:
            mode: "add" or "remove" (selects model/optimizer and negative counters).
            examples: list of tuples:
                (solution_uv_snapshot, move_edges, success_bool)
            neg_weight: multiplicative factor applied to negative move loss.
            max_edges_per_move: cap on edges considered per move.
            hard_within_move_for_neg: if True, for negative moves pick top-prob edges.
        """
        if not examples:
            return
        if mode not in ("add", "remove"):
            raise ValueError(mode)
        if self.node_vecs is None:
            raise RuntimeError("set_node_vectors() must be called before training.")

        model = self.model_add if mode == "add" else self.model_remove
        opt   = self.opt_add   if mode == "add" else self.opt_remove

        model.train()
        opt.zero_grad(set_to_none=True)

        losses = []
        for (solution_uv, move_uv, success) in examples:
            if not move_uv:
                continue

            # canonicalize solution snapshot
            solution_set = {(min(u,v), max(u,v)) for (u,v) in solution_uv if u != v}

            disturbed_neighbors = defaultdict(set)
            for (a, b) in solution_set:
                disturbed_neighbors[a].add(b)
                disturbed_neighbors[b].add(a)

            solution_size = len(solution_set)

            # canonicalize move edges
            move_pairs = [(min(u,v), max(u,v)) for (u,v) in move_uv if u != v]

            # cap repeated negatives
            if not success:
                counter = self.neg_edge_counts_add if mode == "add" else self.neg_edge_counts_remove
                filtered = []
                for p in move_pairs:
                    if counter[p] < self.neg_edge_cap:
                        filtered.append(p)
                        counter[p] += 1
                move_pairs = filtered

            if not move_pairs:
                continue

            # If move is big, sub-sample edges
            if len(move_pairs) > max_edges_per_move:
                pairs_tensor_all = torch.as_tensor(move_pairs, dtype=torch.long, device=self.device)
                feats_all = self._pair_features(
                    pairs_tensor_all,
                    disturbed_neighbors=disturbed_neighbors,
                    solution_size=solution_size
                )

                with torch.no_grad():
                    logits_all = model(feats_all)
                    probs_all = torch.sigmoid(logits_all)

                if (not success) and hard_within_move_for_neg:
                    idx = torch.topk(probs_all, k=max_edges_per_move, largest=True).indices
                else:
                    idx = torch.randperm(len(move_pairs), device=self.device)[:max_edges_per_move]

                feats = feats_all[idx]
            else:
                pairs_tensor = torch.as_tensor(move_pairs, dtype=torch.long, device=self.device)
                feats = self._pair_features(
                    pairs_tensor,
                    disturbed_neighbors=disturbed_neighbors,
                    solution_size=solution_size
                )

            logits = model(feats)               # (m,)
            move_logit = logits.mean()          # mean pooling

            target = torch.tensor([1.0 if success else 0.0], device=self.device)
            loss = F.binary_cross_entropy_with_logits(move_logit.view(1), target)

            if not success:
                loss = loss * float(neg_weight)

            losses.append(loss)

        if not losses:
            return

        total_loss = torch.stack(losses).mean()
        if not torch.isfinite(total_loss):
            self.logger.warning("[%s] move-loss is NaN/Inf, skipping", mode)
            return

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        self.logger.info("[%s][moves] loss=%.6g examples=%d", mode, float(total_loss.item()), len(losses))

    def _move_score(
        self,
        mode: str,
        solution_uv: list[tuple[int,int]],
        move_uv: list[tuple[int,int]],
        max_edges_per_move: int = 15
    ) -> torch.Tensor:
        """
        Compute a scalar move score (logit) by mean-pooling edge logits.

        Args:
            mode: "add" or "remove".
            solution_uv: context snapshot edges.
            move_uv: edges involved in the move.
            max_edges_per_move: subsample cap.

        Returns:
            Scalar tensor (mean logit). Returns None if move has no valid edges.
        """
        model = self.model_add if mode == "add" else self.model_remove

        # context
        solution_set = {(min(u,v), max(u,v)) for (u,v) in solution_uv if u != v}
        disturbed_neighbors = defaultdict(set)
        for (a, b) in solution_set:
            disturbed_neighbors[a].add(b)
            disturbed_neighbors[b].add(a)
        solution_size = len(solution_set)

        move_pairs = [(min(u,v), max(u,v)) for (u,v) in move_uv if u != v]
        if not move_pairs:
            return None

        # subsample edges inside big move
        if len(move_pairs) > max_edges_per_move:
            move_pairs = random.sample(move_pairs, max_edges_per_move)

        pairs_tensor = torch.as_tensor(move_pairs, dtype=torch.long, device=self.device)
        feats = self._pair_features(pairs_tensor, disturbed_neighbors=disturbed_neighbors, solution_size=solution_size)

        logits = model(feats)      # (m,)
        return logits.mean()       # scalar
    
    def _train_ranked_moves(
        self,
        mode: str,
        solution_uv: list[tuple[int,int]],
        pos_move_uv: list[tuple[int,int]],
        neg_moves_uv: list[list[tuple[int,int]]],
        margin: float = 0.5,
        max_negs: int = 25,
        max_edges_per_move: int = 15,
        neg_cap_per_edge: int | None = None,
    ) -> None:
        """
        Pairwise ranking loss training: enforce pos_move scores higher than neg_move scores.

        Theoretical note:
          - Uses a margin-based soft ranking objective:
                loss = softplus(margin + neg_score - pos_score)
            which is a smooth hinge. It pushes pos_score >= neg_score + margin.
          - Each move score is mean(edge_logits).

        Args:
            mode: "add" or "remove".
            solution_uv: context snapshot.
            pos_move_uv: the chosen "good" move.
            neg_moves_uv: list of alternative "bad" moves.
            margin: separation margin.
            max_negs: cap number of negatives per update.
            max_edges_per_move: subsample cap for edge lists.
            neg_cap_per_edge: optional per-edge negative exposure cap.
        """
        if self.node_vecs is None:
            raise RuntimeError("set_node_vectors() must be called before training.")
        if not neg_moves_uv:
            return

        model = self.model_add if mode == "add" else self.model_remove
        opt   = self.opt_add   if mode == "add" else self.opt_remove

        # cap negatives count
        if len(neg_moves_uv) > max_negs:
            neg_moves_uv = neg_moves_uv[:max_negs]

        # per-edge cap to avoid oversaturating repeats
        if neg_cap_per_edge is not None:
            counter = self.neg_edge_counts_add if mode == "add" else self.neg_edge_counts_remove
            filtered = []
            for mv in neg_moves_uv:
                ok = False
                for e in mv:
                    p = (min(e[0], e[1]), max(e[0], e[1]))
                    if counter[p] < neg_cap_per_edge:
                        ok = True
                        break
                if ok:
                    for e in mv:
                        p = (min(e[0], e[1]), max(e[0], e[1]))
                        counter[p] += 1
                    filtered.append(mv)
            neg_moves_uv = filtered
            if not neg_moves_uv:
                return

        model.train()
        opt.zero_grad(set_to_none=True)

        pos_score = self._move_score(mode, solution_uv, pos_move_uv, max_edges_per_move=max_edges_per_move)
        if pos_score is None:
            return

        losses = []
        for neg_mv in neg_moves_uv:
            neg_score = self._move_score(mode, solution_uv, neg_mv, max_edges_per_move=max_edges_per_move)
            if neg_score is None:
                continue
            # softplus(margin - (pos - neg)) = softplus(margin + neg - pos)
            losses.append(F.softplus(margin + neg_score - pos_score))

        if not losses:
            return

        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            self.logger.warning("[%s][rank] loss NaN/Inf, skipping", mode)
            return

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        self.logger.info("[%s][rank] loss=%.6g negs=%d", mode, float(loss.item()), len(losses))

    # ---------- Online update ----------

    def update_addition_moves(self, examples: list[tuple[list[tuple[int,int]], list[tuple[int,int]], bool]]) -> None:
        """
        Online update for addition moves using supervised success/failure signals.

        Args:
            examples: (solution_snapshot, move_edges, success_bool) tuples.
        """
        self._train_online_moves("add", examples)
        
    def update_removal_moves(self, examples: list[tuple[list[tuple[int,int]], list[tuple[int,int]], bool]]) -> None:
        """
        Online update for removal moves using supervised success/failure signals.

        Args:
            examples: (solution_snapshot, move_edges, success_bool) tuples.
        """
        self._train_online_moves("remove", examples)

    def update_removal_ranked(
        self,
        solution_uv: list[tuple[int,int]],
        pos_removed_uv: list[tuple[int,int]],
        neg_removed_uvs: list[list[tuple[int,int]]]
    ) -> None:
        """
        Ranking update for removals: pos_removed should score above neg_removed options.

        Args:
            solution_uv: current context snapshot.
            pos_removed_uv: removed edges for the accepted move.
            neg_removed_uvs: list of alternative removed-edge sets.
        """
        self._train_ranked_moves("remove", solution_uv, pos_removed_uv, neg_removed_uvs,
                                neg_cap_per_edge=self.neg_edge_cap)

    def update_addition_ranked(
        self,
        solution_uv: list[tuple[int,int]],
        pos_added_uv: list[tuple[int,int]],
        neg_added_uvs: list[list[tuple[int,int]]]
    ) -> None:
        """
        Ranking update for additions: pos_added should score above neg_added options.

        Args:
            solution_uv: current context snapshot.
            pos_added_uv: added edges for the accepted move.
            neg_added_uvs: list of alternative added-edge sets.
        """
        self._train_ranked_moves("add", solution_uv, pos_added_uv, neg_added_uvs,
                                neg_cap_per_edge=self.neg_edge_cap)

    
    # ---------- Save / load ----------

    def save(self, path: str) -> None:
        """
        Save model and optimizer states to a file.
        (Node vectors are NOT saved; set them again with set_node_vectors.)
        Replay buffers are NOT saved (they refill during new runs).

        Args:
            path: filesystem path to write checkpoint.
        """
        ckpt = {
            "k": self.k,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "exploration_prob": self.exploration_prob,
            "model_add": self.model_add.state_dict(),
            "model_remove": self.model_remove.state_dict(),
            "opt_add": self.opt_add.state_dict(),
            "opt_remove": self.opt_remove.state_dict(),
        }
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path: str, lr: float = 1e-3, device: torch.device | None = None) -> "OnlineNNEdgeSelector":
        """
        Load model from file. You still need to call set_node_vectors() afterwards.

        Args:
            path: checkpoint path.
            lr: learning rate to reinitialize optimizers (state is restored after init).
            device: torch device to map the checkpoint to.

        Returns:
            A reconstructed OnlineNNEdgeSelector with loaded weights/optimizer states.
        """
        ckpt = torch.load(path, map_location=device if device is not None else "cpu")


        obj = cls(
            k=ckpt["k"],
            hidden_dim=ckpt["hidden_dim"],
            num_hidden_layers=ckpt["num_hidden_layers"],
            lr=lr,
            exploration_prob=ckpt["exploration_prob"],
            device=device,
        )
        obj.model_add.load_state_dict(ckpt["model_add"])
        obj.model_remove.load_state_dict(ckpt["model_remove"])
        obj.opt_add.load_state_dict(ckpt["opt_add"])
        obj.opt_remove.load_state_dict(ckpt["opt_remove"])
        return obj
    
    # ---------- context helpers ----------
    def _solution_context(self, solution_set: set[tuple[int,int]], num_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build quick per-node context features from the current solution set.

        Args:
            solution_set: set of canonical solution edges.
            num_nodes: number of nodes.

        Returns:
            degS: tensor (num_nodes,) solution-induced degrees
            inS: tensor (num_nodes,) binary indicator node participates in solution
            sol_size_t: tensor (1,) number of solution edges
        """
        # degree within solution-induced edge set
        degS = torch.zeros(num_nodes, device=self.device, dtype=torch.float32)
        inS  = torch.zeros(num_nodes, device=self.device, dtype=torch.float32)

        for (a, b) in solution_set:
            degS[a] += 1.0
            degS[b] += 1.0
            inS[a] = 1.0
            inS[b] = 1.0

        sol_size = float(len(solution_set))
        sol_size_t = torch.tensor([sol_size], device=self.device, dtype=torch.float32)

        return degS, inS, sol_size_t

    # ---------- base graph + startup helpers ----------
    def set_base_graph(self, adj_np: np.ndarray, directed: bool = False) -> None:
        """
        Register the base/original graph for the current problem instance.

        Theoretical role:
          - The "base graph" is the unmodified graph.
          - The "current graph" is derived as base XOR solution_disturbs.
          - Structural features compare base vs current properties.

        Args:
            adj_np: (N,N) numpy array with 0/1 adjacency for original instance.
            directed: if True, treat adjacency as directed (neighbors from rows).
        """
        assert adj_np.ndim == 2 and adj_np.shape[0] == adj_np.shape[1]
        self.num_nodes = int(adj_np.shape[0])
        self._directed = bool(directed)

        # Store neighbors as python sets for fast membership.
        self.base_neighbors = []
        self.base_deg = np.zeros(self.num_nodes, dtype=np.int32)

        if directed:
            # treat adjacency row as outgoing neighbors
            for u in range(self.num_nodes):
                nbrs = set(np.nonzero(adj_np[u])[0].tolist())
                if u in nbrs:
                    nbrs.remove(u)
                self.base_neighbors.append(nbrs)
                self.base_deg[u] = len(nbrs)
        else:
            # undirected: neighbors where adj[u,v]=1
            # assume adj is symmetric (or at least you want symmetric behavior)
            for u in range(self.num_nodes):
                nbrs = set(np.nonzero(adj_np[u])[0].tolist())
                if u in nbrs:
                    nbrs.remove(u)
                self.base_neighbors.append(nbrs)
                self.base_deg[u] = len(nbrs)

        # Keep base adjacency available for base_edge flag
        # (store as bool for cheap lookup)
        self.base_adj_bool = (adj_np != 0)

    def _count_intersection(self, set_a: set[int], set_b: set[int]) -> int:
        """
        Count |set_a ∩ set_b| efficiently by iterating the smaller set.

        Args:
            set_a: first set
            set_b: second set

        Returns:
            Intersection size.
        """
        if len(set_a) > len(set_b):
            set_a, set_b = set_b, set_a
        return sum(1 for x in set_a if x in set_b)

    def _deg_current(self, u: int, tog_u: set[int]) -> int:
        """
        Degree of u in current graph (base XOR disturbs incident to u).

        Theoretical note:
          - If an incident edge (u,w) is disturbed:
              * if (u,w) exists in base, degree decreases by 1
              * else degree increases by 1
          - So:
              deg_curr = deg_base + (#disturbed_not_in_base) - (#disturbed_in_base)
                       = deg_base + len(tog_u) - 2*(#disturbed_in_base)

        Computed in O(|tog_u|).

        Args:
            u: node id
            tog_u: set of nodes w for which edge (u,w) is disturbed.

        Returns:
            Current degree of u.
        """
        if not tog_u:
            return int(self.base_deg[u])
        baseN = self.base_neighbors[u]
        in_base = sum(1 for w in tog_u if w in baseN)
        # disturbs flip: present->absent (-1), absent->present (+1)
        return int(self.base_deg[u] + (len(tog_u) - 2 * in_base))

    def _cn_current(self, u: int, v: int, tog_u: set[int], tog_v: set[int]) -> tuple[int, int]:
        """
        Compute common neighbors in base and current graphs: (cn_base, cn_curr).

        Theoretical trick:
          - cn_base = |N_base(u) ∩ N_base(v)|
          - Only nodes whose adjacency changed for u or v can affect cn in the XOR model,
            i.e. nodes in (tog_u ∪ tog_v).
          - For each such node w, we correct whether w is a neighbor in base vs current
            for u and v, then adjust the intersection count.

        Complexity:
          O(|tog_u| + |tog_v|) after computing cn_base.

        Args:
            u: node id
            v: node id
            tog_u: disturbed neighbors for u
            tog_v: disturbed neighbors for v

        Returns:
            (cn_base, cn_curr)
        """
        Nu = self.base_neighbors[u]
        Nv = self.base_neighbors[v]

        cn_base = self._count_intersection(Nu, Nv)

        if not tog_u and not tog_v:
            return cn_base, cn_base

        # Only nodes in tog_u ∪ tog_v can change membership compared to base.
        union_tog = tog_u.union(tog_v)
        delta = 0
        for w in union_tog:
            bu = (w in Nu)
            bv = (w in Nv)
            base_in = (bu and bv)

            cu = (bu ^ (w in tog_u))
            cv = (bv ^ (w in tog_v))
            curr_in = (cu and cv)

            delta += (1 if curr_in else 0) - (1 if base_in else 0)

        cn_curr = cn_base + delta
        if cn_curr < 0:
            cn_curr = 0
        return cn_base, cn_curr

    # ---------- logging ----------
    def _cuda_sync(self) -> None:
        """Synchronize CUDA for accurate timing (no-op on CPU)."""
        if self._use_cuda_timing:
            torch.cuda.synchronize()

    def _ema_update(self, key: str, value: float) -> float:
        """
        Update an exponential moving average for a metric.

        Args:
            key: metric name.
            value: new value.

        Returns:
            Updated EMA value (or existing value if input is not finite).
        """
        if value is None or not math.isfinite(value):
            return self._ema.get(key, float("nan"))
        old = self._ema.get(key, value)
        new = self.ema_beta * old + (1.0 - self.ema_beta) * value
        self._ema[key] = new
        return new

    def _log_train_status(self) -> None:
        """Logs whatever is in self._last_train + EMAs."""
        if not self._last_train:
            self.logger.info("[train] no training events yet")
            return

        lt = self._last_train
        self.logger.info(
            "[train][call=%d mode=%s model=%s] "
            "loss=%.6g (ema=%.6g) grad_norm=%.4g (ema=%.4g) "
            "param_abs_sum=%.6g Δ=%.3g "
            "lr=%.3g step_time=%.4fs (ema=%.4fs) ex/s=%.1f",
            lt.get("train_calls", -1),
            lt.get("mode", "?"),
            lt.get("name", "?"),
            lt.get("loss", float("nan")),
            self._ema.get(f"loss/{lt.get('name','?')}", float("nan")),
            lt.get("grad_norm", float("nan")),
            self._ema.get(f"grad/{lt.get('name','?')}", float("nan")),
            lt.get("param_abs_sum_post", float("nan")),
            lt.get("param_abs_sum_delta", float("nan")),
            lt.get("lr", float("nan")),
            lt.get("step_time_s", float("nan")),
            self._ema.get(f"time/{lt.get('name','?')}", float("nan")),
            lt.get("examples_per_s", float("nan")),
        )

        # extra debug line: output distribution
        self.logger.debug(
            "[train][%s] logits(mean=%.4g std=%.4g min=%.4g max=%.4g) "
            "probs(mean=%.4g std=%.4g min=%.4g max=%.4g)",
            lt.get("name", "?"),
            lt.get("logits_mean", float("nan")),
            lt.get("logits_std", float("nan")),
            lt.get("logits_min", float("nan")),
            lt.get("logits_max", float("nan")),
            lt.get("probs_mean", float("nan")),
            lt.get("probs_std", float("nan")),
            lt.get("probs_min", float("nan")),
            lt.get("probs_max", float("nan")),
        )

    def _log_propose_status(self) -> None:
        """NO external params. Logs whatever is in self._last_propose."""
        if not self._last_propose:
            self.logger.info("[propose] no propose events yet")
            return

        lp = self._last_propose
        self.logger.info(
            "[propose][call=%d mode=%s] candidates=%d X=%d "
            "explore=%s eps=%.3f "
            "p(mean=%.4g std=%.4g min=%.4g max=%.4g)",
            lp.get("propose_calls", -1),
            lp.get("mode", "?"),
            lp.get("num_candidates", -1),
            lp.get("X", -1),
            lp.get("did_explore", False),
            lp.get("exploration_prob", float("nan")),
            lp.get("p_mean", float("nan")),
            lp.get("p_std", float("nan")),
            lp.get("p_min", float("nan")),
            lp.get("p_max", float("nan")),
        )
