from __future__ import annotations
import random
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Any, Dict, Optional
import math
import time
import torch.nn.functional as F
from collections import defaultdict


from typing import Literal
import torch
import torch.nn as nn


ActivationName = Literal["ReLU", "LeakyReLU", "ELU", "Swish"]


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")

class ScoreNet(nn.Module):
    """
    Small MLP that outputs a single logit score for each input feature vector.

    Uses BatchNorm + activation + dropout after each hidden Linear layer.
    BatchNorm is skipped when batch_size == 1 to avoid unstable statistics.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer width.
        num_hidden_layers: Number of hidden layers.
        activation_function: Activation to use ("ReLU", "LeakyReLU", "ELU", "Swish").
        dropout_prob: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        activation_function: ActivationName = "ReLU",
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

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

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act)
            layers.append(nn.Dropout(dropout_prob))
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, input_dim).

        Returns:
            Logits tensor of shape (batch,).
        """
        if x.size(0) == 1:
            # Skip BatchNorm when batch=1.
            for layer in self.net:
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                x = layer(x)
            return x.squeeze(-1)

        return self.net(x).squeeze(-1)
         
class OnlineNNEdgeSelector:
    """
    Online neural model to bias which edges to add/remove.

    Usage:
        selector = OnlineNNEdgeSelector(k=node_vecs.shape[1])
        selector.set_node_vectors(node_vecs)  # np.ndarray (num_nodes, k)

        # propose:
        chosen_add_pairs = selector.propose_additions(solution_pairs, X)

        # after evaluating (reward in [0,1]):
        selector.update_additions(chosen_add_pairs, reward)
    """

    def __init__(
        self,
        k: int,  # dimension of node vectors
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        lr: float = 1e-3,
        exploration_prob: float = 0.2,
        device: torch.device | None = None,
    ):
        self.example_count = 0
        self.k = k
        self.struct_dim = 26
        self.input_dim = 4 * k + 4 + self.struct_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.exploration_prob = exploration_prob
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model_add = ScoreNet(self.input_dim, hidden_dim, num_hidden_layers, activation_function='Swish').to(self.device)
        self.model_remove = ScoreNet(self.input_dim, hidden_dim, num_hidden_layers, activation_function='Swish').to(self.device)

        self.opt_add = optim.Adam(self.model_add.parameters(), lr=lr)
        self.opt_remove = optim.Adam(self.model_remove.parameters(), lr=lr)
        
        # Learning rate scheduler (using ReduceLROnPlateau)
        self.scheduler_add = optim.lr_scheduler.ReduceLROnPlateau(self.opt_add, mode='min', factor=0.5, patience=5)
        self.scheduler_remove = optim.lr_scheduler.ReduceLROnPlateau(self.opt_remove, mode='min', factor=0.5, patience=5)


        self.loss_fn = nn.BCEWithLogitsLoss()

        self.node_vecs: torch.Tensor | None = None  # will be set by set_node_vectors()
        self.node_vecs_np: np.ndarray | None = None

        # Track performance for dynamic exploration adjustment
        self.performance_counter = 0
        self.last_best_size = None
        
        self.temp_add = 1.5      
        self.temp_remove = 1.2
        
        self.neg_edge_counts_add = defaultdict(int)
        self.neg_edge_counts_remove = defaultdict(int)
        self.neg_edge_cap = 5  # tuneable
        
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
        self._ema = {}  # name -> float

        # store last events
        self._last_train = {}
        self._last_propose = {}

        
        self._use_cuda_timing = (self.device.type == "cuda")

        # Use per-sample loss so we can log extra stats
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        
    # ---------- Node vectors & features ----------

    def build_tensors(self, node_vecs_np: np.ndarray) -> torch.Tensor:
        """
        Call this once (or whenever node vectors change).

        node_vecs_np: shape (num_nodes, k)
        """
        return torch.as_tensor(
            node_vecs_np, dtype=torch.float32, device=self.device
        )
        
    def set_node_vectors(self, node_vecs_np: np.ndarray):
        """
        Call this once (or whenever node vectors change).

        node_vecs_np: shape (num_nodes, k)
        """
        assert node_vecs_np.shape[1] == self.k
        self.node_vecs = torch.as_tensor(
            node_vecs_np, dtype=torch.float32, device=self.device
        )
        
    def set_node_vectors_from_tensor(self, node_vecs_list: list[torch.Tensor]):
        """
        node_vecs_list: list of length num_nodes,
                        each a 1D tensor of shape (k,)
                        possibly with requires_grad=True.
        """
        # Detach from computation graph and stack
        with torch.no_grad():
            node_vecs = torch.stack(
                [v.detach() for v in node_vecs_list], dim=0
            )  # (num_nodes, k)

            assert node_vecs.shape[1] == self.k

            self.node_vecs = node_vecs.to(self.device, dtype=torch.float32)

    def _pair_features(self, pairs_tensor: torch.Tensor, toggled_neighbors: dict[int, set[int]], solution_size: int) -> torch.Tensor:
        """
        Build features for many pairs at once.
        pairs_tensor: LongTensor (M,2)
        toggled_neighbors: node -> set of toggled neighbors in current solution
        solution_size: |solution_set|
        """
        assert self.node_vecs is not None, "Call set_node_vectors() first."
        assert hasattr(self, "base_neighbors"), "Call set_base_graph() first."
        assert hasattr(self, "num_nodes"), "Call set_base_graph() first."

        u_idx = pairs_tensor[:, 0]
        v_idx = pairs_tensor[:, 1]

        v_u = self.node_vecs[u_idx]  # (M,k)
        v_v = self.node_vecs[v_idx]  # (M,k)

        diff = v_u - v_v
        prod = v_u * v_v
        dot = (v_u * v_v).sum(dim=1, keepdim=True)
        norm_u = v_u.norm(dim=1, keepdim=True)
        norm_v = v_v.norm(dim=1, keepdim=True)
        dist = diff.norm(dim=1, keepdim=True)

        # --- structural features (python loop; M is small/moderate) ---
        M = pairs_tensor.size(0)
        eps = 1e-9
        N = float(self.num_nodes)
        denom_deg = max(1.0, N - 1.0)
        denom_cn  = max(1.0, N - 2.0)
        sol_size = max(1, int(solution_size))

        struct_rows = []
        pairs_cpu = pairs_tensor.detach().cpu().numpy()

        for (u, v) in pairs_cpu:
            u = int(u); v = int(v)
            tog_u = toggled_neighbors.get(u, set())
            tog_v = toggled_neighbors.get(v, set())

            base_edge = 1.0 if self.base_adj_bool[u, v] else 0.0
            # current edge = base XOR toggled(pair)
            # (pair is toggled if v in tog_u, symmetric if undirected)
            pair_toggled = (v in tog_u)  # for undirected this is enough
            curr_edge = 1.0 if (base_edge > 0.5) ^ pair_toggled else 0.0

            deg_u_base = float(self.base_deg[u]) / denom_deg
            deg_v_base = float(self.base_deg[v]) / denom_deg

            deg_u_curr_raw = float(self._deg_current(u, tog_u))
            deg_v_curr_raw = float(self._deg_current(v, tog_v))

            deg_u_curr = deg_u_curr_raw / denom_deg
            deg_v_curr = deg_v_curr_raw / denom_deg

            sol_deg_u = float(len(tog_u)) / float(sol_size)
            sol_deg_v = float(len(tog_v)) / float(sol_size)

            cn_base, cn_curr = self._cn_current(u, v, tog_u, tog_v)

            u_in_sol = 1.0 if len(tog_u) > 0 else 0.0
            v_in_sol = 1.0 if len(tog_v) > 0 else 0.0

            deg_u_base_raw = float(self.base_deg[u])
            deg_v_base_raw = float(self.base_deg[v])

            # normalized cn
            cn_base_n = float(cn_base) / denom_cn
            cn_curr_n = float(cn_curr) / denom_cn

            # jaccard
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
            ])

        struct = torch.as_tensor(struct_rows, dtype=torch.float32, device=self.device)  # (M,struct_dim)

        feats = torch.cat(
            [v_u, v_v, diff, prod, dot, norm_u, norm_v, dist, struct],
            dim=1
        )
        assert feats.shape[1] == self.input_dim, (feats.shape, self.input_dim)
        return feats

    # ---------- Sampling logic ----------

    def _sample_indices(self, weights_np: np.ndarray, X: int):
        """
        weights_np: array length M, non-negative, doesn't need to sum to 1
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

        # ---- Greedy path (Efficiency-first) ----
        if self.greedy_top1 and X == 1:
            return [int(np.argmax(w))]

        if self.greedy_topk and X <= self.greedy_k_threshold:
            # deterministic top-k
            idx = np.argsort(-w)[:X]
            return idx.tolist()

        # ---- Otherwise weighted sampling ----
        eps = 1e-12
        w[w < eps] = eps
        s = w.sum()
        if s <= 0.0:
            return random.sample(range(n), X)

        p = w / s
        idx = np.random.choice(np.arange(n), size=X, replace=False, p=p)
        return idx.tolist()
    
    def _sample_indices_no_explore(self, probs_np: np.ndarray, X: int):
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

    def _propose(self, solution_pairs, X: int, mode: str):
        """
        Internal: propose X pairs for given mode ('add' or 'remove').

        Parameters
        ----------
        solution_pairs : list[tuple[int, int]]
            Pairs (u, v) that are currently in the solution.
            Assumed to be node indices in [0, num_nodes).

        X : int
            Number of pairs to propose.

        mode : str
            'add'  -> propose pairs from the universe that are NOT in solution_pairs.
            'remove' -> propose pairs that ARE in solution_pairs.
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
        
        toggled_neighbors = defaultdict(set)
        for (a, b) in solution_set:
            toggled_neighbors[a].add(b)
            toggled_neighbors[b].add(a)  # undirected assumption
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

            # focus around nodes already involved in the current solution
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

        # If there are no candidates, or fewer than X, return them all
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
            feats = self._pair_features(pairs_tensor, toggled_neighbors=toggled_neighbors, solution_size=solution_size) # (M, input_dim)
            logits = model(feats)                              # (M,)
            probs = torch.sigmoid(logits).cpu().numpy()        # P(success)
        
        # ---- save propose stats ----
        if M >= 2:
            top2 = np.partition(probs, -2)[-2:]
            p_gap = float(np.max(top2) - np.min(top2))
        else:
            p_gap = 0.0
        self._last_propose["p_gap_top2"] = p_gap

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
                # Fix 3: greedy argmax when not exploring
                idx = [int(np.argmax(logits_np))]
            else:
                idx = self._sample_indices_no_explore(weights, X)

        return [candidate_pairs[i] for i in idx]
    
    def _canonical_pair(self, u: int, v: int) -> tuple[int, int] | None:
        if u == v:
            return None
        a, b = (u, v) if u < v else (v, u)
        return (a, b)
    
    def _weights_from_logits(self, logits: np.ndarray, mode: str) -> np.ndarray:
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
        Sample up to pool_size candidate edges (u,v) that are NOT in solution_set.
        If focus_nodes is provided, bias sampling so at least one endpoint is in focus_nodes.
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

    def propose_additions(self, solution, X: int):
        """
        Choose X candidate pairs to ADD.
        `solution` is the current solution as a list of (u, v) pairs.
        """
        
        return self._propose(solution, X, mode="add")

    def propose_removals(self, solution, X: int):
        """
        Choose X candidate pairs to REMOVE.
        `solution` is the current solution as a list of (u, v) pairs.
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
    ):
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
            if not solution_set:
                continue

            toggled_neighbors = defaultdict(set)
            for (a, b) in solution_set:
                toggled_neighbors[a].add(b)
                toggled_neighbors[b].add(a)

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
                    toggled_neighbors=toggled_neighbors,
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
                    toggled_neighbors=toggled_neighbors,
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

    def _move_score(self, mode: str, solution_uv: list[tuple[int,int]], move_uv: list[tuple[int,int]],
                max_edges_per_move: int = 15) -> torch.Tensor:
        """
        Returns scalar logit score for a move = mean(edge_logits).
        solution_uv: context snapshot
        move_uv: list of edges involved in the move
        """
        model = self.model_add if mode == "add" else self.model_remove

        # context
        solution_set = {(min(u,v), max(u,v)) for (u,v) in solution_uv if u != v}
        toggled_neighbors = defaultdict(set)
        for (a, b) in solution_set:
            toggled_neighbors[a].add(b)
            toggled_neighbors[b].add(a)
        solution_size = len(solution_set)

        move_pairs = [(min(u,v), max(u,v)) for (u,v) in move_uv if u != v]
        if not move_pairs:
            return None

        # subsample edges inside big move (cheap)
        if len(move_pairs) > max_edges_per_move:
            move_pairs = random.sample(move_pairs, max_edges_per_move)

        pairs_tensor = torch.as_tensor(move_pairs, dtype=torch.long, device=self.device)
        feats = self._pair_features(pairs_tensor, toggled_neighbors=toggled_neighbors, solution_size=solution_size)

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
    ):
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

    def update_addition_moves(self, examples: list[tuple[list[tuple[int,int]], list[tuple[int,int]], bool]]):
        self._train_online_moves("add", examples)
        
    def update_removal_moves(self, examples: list[tuple[list[tuple[int,int]], list[tuple[int,int]], bool]]):
        self._train_online_moves("remove", examples)

    def update_removal_ranked(self, solution_uv, pos_removed_uv, neg_removed_uvs):
        self._train_ranked_moves("remove", solution_uv, pos_removed_uv, neg_removed_uvs,
                                neg_cap_per_edge=self.neg_edge_cap)

    def update_addition_ranked(self, solution_uv, pos_added_uv, neg_added_uvs):
        self._train_ranked_moves("add", solution_uv, pos_added_uv, neg_added_uvs,
                                neg_cap_per_edge=self.neg_edge_cap)

    
    # ---------- Save / load ----------

    def save(self, path: str):
        """
        Save model and optimizer states to a file.
        (Node vectors are NOT saved; set them again with set_node_vectors.)
        Replay buffers are NOT saved (they refill during new runs).
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
    def _solution_context(self, solution_set: set[tuple[int,int]], num_nodes: int):
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
    def set_base_graph(self, adj_np: np.ndarray, directed: bool = False):
        """
        adj_np: (N,N) 0/1 numpy array for the ORIGINAL graph of the current instance.
        Call once per instance before proposing/training.
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
        # no temp set allocation
        if len(set_a) > len(set_b):
            set_a, set_b = set_b, set_a
        return sum(1 for x in set_a if x in set_b)

    def _deg_current(self, u: int, tog_u: set[int]) -> int:
        """
        Degree of u in current graph (base XOR toggles incident to u).
        Computed in O(|tog_u|).
        """
        if not tog_u:
            return int(self.base_deg[u])
        baseN = self.base_neighbors[u]
        in_base = sum(1 for w in tog_u if w in baseN)
        # toggles flip: present->absent (-1), absent->present (+1)
        return int(self.base_deg[u] + (len(tog_u) - 2 * in_base))

    def _cn_current(self, u: int, v: int, tog_u: set[int], tog_v: set[int]) -> tuple[int, int]:
        """
        Returns (cn_base, cn_curr) using cheap correction over toggles.
        cn_base computed by membership counting, cn_curr adjusted in O(|tog_u|+|tog_v|).
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
    def _cuda_sync(self):
        if self._use_cuda_timing:
            torch.cuda.synchronize()

    def _ema_update(self, key: str, value: float) -> float:
        if value is None or not math.isfinite(value):
            return self._ema.get(key, float("nan"))
        old = self._ema.get(key, value)
        new = self.ema_beta * old + (1.0 - self.ema_beta) * value
        self._ema[key] = new
        return new

    def _log_train_status(self):
        """Logs whatever is in self._last_train + EMAs."""
        if not self._last_train:
            self.logger.info("[train] no training events yet")
            return

        lt = self._last_train
        # keep this compact but informative
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

    def _log_propose_status(self):
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