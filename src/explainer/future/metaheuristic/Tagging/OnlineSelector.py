from __future__ import annotations
import random
from dataclasses import dataclass

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
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")

class ScoreNet(nn.Module):
    """
    MLP with Batch Normalization, alternate activation functions, residual connections, and Dropout.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_hidden_layers: int = 2,
                 activation_function: str = "ReLU", dropout_prob: float = 0.2):
        super().__init__()

        # Choose activation function
        if activation_function == "ReLU":
            self.activation = nn.ReLU()
        elif activation_function == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation_function == "ELU":
            self.activation = nn.ELU()
        elif activation_function == "Swish":
            self.activation = nn.SiLU()  # Swish is implemented as SiLU in PyTorch
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")

        layers = []
        dim = input_dim
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add Batch Normalization
            layers.append(self.activation)  # Apply chosen activation function
            layers.append(nn.Dropout(dropout_prob))  # Apply Dropout
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))  # Output logit
        self.net = nn.Sequential(*layers)
        
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        if x.size(0) == 1:  # If batch size is 1, disable BatchNorm
            for layer in self.net:
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                x = layer(x)
            return x.squeeze(-1)
        else:
            out = self.net(x)
            return out.squeeze(-1)            
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
        self.input_dim = 4 * k + 4 + 5  # see _pair_features()
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

    def _pair_features(self, pairs_tensor, degS=None, inS=None, sol_size=None, Eplus=None) -> torch.Tensor:
        """
        Build features for many pairs at once.

        pairs_tensor: LongTensor of shape (M, 2) with (u, v) indices.

        Returns:
            feats: (M, input_dim)
        """
        assert self.node_vecs is not None, "Call set_node_vectors() first."
        u_idx = pairs_tensor[:, 0]
        v_idx = pairs_tensor[:, 1]

        v_u = self.node_vecs[u_idx]  # (M, k)
        v_v = self.node_vecs[v_idx]  # (M, k)

        diff = v_u - v_v                             # (M, k)
        prod = v_u * v_v                             # (M, k)
        dot = (v_u * v_v).sum(dim=1, keepdim=True)   # (M, 1)
        norm_u = v_u.norm(dim=1, keepdim=True)       # (M, 1)
        norm_v = v_v.norm(dim=1, keepdim=True)       # (M, 1)
        dist = diff.norm(dim=1, keepdim=True)        # (M, 1)

        feats = torch.cat(
            [v_u, v_v, diff, prod, dot, norm_u, norm_v, dist],
            dim=1
        )
        
        # ----- context features (if provided) -----
        if degS is None or inS is None or sol_size is None or Eplus is None:
            # fallback: zeros if context not provided
            M = pairs_tensor.size(0)
            extra = torch.zeros((M, self.ctx_dim), device=self.device, dtype=torch.float32)
        else:
            u_idx = pairs_tensor[:, 0]
            v_idx = pairs_tensor[:, 1]

            deg_u = degS[u_idx].unsqueeze(1)
            deg_v = degS[v_idx].unsqueeze(1)

            in_u  = inS[u_idx].unsqueeze(1)
            in_v  = inS[v_idx].unsqueeze(1)

            sol_size_norm = (sol_size / max(1.0, float(Eplus))).expand(pairs_tensor.size(0), 1)

            extra = torch.cat([deg_u, deg_v, in_u, in_v, sol_size_norm], dim=1)  # (M,5)

        feats = torch.cat([feats, extra], dim=1)
        assert feats.shape[1] == self.input_dim
        return feats

        assert feats.shape[1] == self.input_dim
        return feats

    # ---------- Sampling logic ----------

    def _sample_indices(self, probs_np: np.ndarray, X: int):
        """
        Sample X indices using epsilon-greedy over probs_np.
        probs_np: array of length M, ideally in [0,1]
        """
        n = len(probs_np)
        n = len(probs_np)
        if n <= X:
            self._last_propose["did_explore"] = False
            return list(range(n))
        
        # Exploration: pure random choice
        if np.random.rand() < self.exploration_prob:
            self._last_propose["did_explore"] = True
            return random.sample(range(n), X)

        self._last_propose["did_explore"] = False

        probs = np.asarray(probs_np, dtype=float)
        probs = np.maximum(probs, 0.0)

        # If too few non-zero entries, fall back to uniform sample
        nonzero_count = np.count_nonzero(probs)
        if nonzero_count < X:
            return random.sample(range(n), X)

        # Ensure no entry is exactly zero (avoid numpy "Fewer non-zero entries" error)
        eps = 1e-12
        probs[probs < eps] = eps

        total = probs.sum()
        if total <= 0.0:
            # degenerate case, uniform again
            return random.sample(range(n), X)
        probs = probs / total

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
        
        degS, inS, sol_size = self._solution_context(solution_set, num_nodes)
        Eplus = num_nodes * (num_nodes - 1) / 2  # undirected

        

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
            feats = self._pair_features(pairs_tensor, degS=degS, inS=inS, sol_size=sol_size, Eplus=Eplus) # (M, input_dim)
            logits = model(feats)                              # (M,)
            probs = torch.sigmoid(logits).cpu().numpy()        # P(success)
        
        # ---- save propose stats ----
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
        idx = self._sample_indices(weights, X)
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

    def _train_online(
        self,
        mode: str,
        pairs: list[tuple[int, int]] | None = None,
        success: bool = True,   # NEW
    ):
        if not pairs:
            return
        if mode not in ("add", "remove"):
            raise ValueError(f"mode must be 'add' or 'remove', got {mode!r}")
        if self.node_vecs is None:
            raise RuntimeError("set_node_vectors() must be called before training.")

        pairs_tensor = torch.as_tensor(pairs, dtype=torch.long, device=self.device)
        feats = self._pair_features(pairs_tensor).to(self.device)

        y = 1.0 if success else 0.0

        # Train ONLY the model corresponding to the action
        if mode == "add":
            steps = [("add", self.model_add, self.opt_add, y)]
            # optional: only if success, teach remove model "this shouldn't be removed"
            if success:
                steps.append(("remove", self.model_remove, self.opt_remove, 0.0))
        else:
            steps = [("remove", self.model_remove, self.opt_remove, y)]
            # optional: only if success, teach add model "this shouldn't be added back"
            if success:
                steps.append(("add", self.model_add, self.opt_add, 0.0))

        self.train_calls += 1
        if mode == "add":
            self.train_calls_add += 1
        else:
            self.train_calls_remove += 1

        for name, model, opt, target_value in steps:
            targets = torch.full((feats.size(0),), float(target_value),
                                dtype=torch.float32, device=self.device)

            self._cuda_sync()
            t0 = time.perf_counter()

            with torch.no_grad():
                param_abs_sum_pre = float(sum(p.detach().abs().sum().item() for p in model.parameters()))

            model.train()
            opt.zero_grad(set_to_none=True)

            preds = model(feats)
            per_sample = self.loss_fn(preds, targets)
            loss = per_sample.mean()

            if not torch.isfinite(loss):
                self.logger.warning("[%s] loss is NaN/Inf, skipping step", name)
                continue

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
            opt.step()

            self._cuda_sync()
            dt = time.perf_counter() - t0

            with torch.no_grad():
                param_abs_sum_post = float(sum(p.detach().abs().sum().item() for p in model.parameters()))
                param_abs_sum_delta = param_abs_sum_post - param_abs_sum_pre

                # output distribution
                probs = torch.sigmoid(preds.detach())
                logits_mean = float(preds.detach().mean().item())
                logits_std  = float(preds.detach().std(unbiased=False).item()) if preds.numel() > 1 else 0.0
                logits_min  = float(preds.detach().min().item())
                logits_max  = float(preds.detach().max().item())

                probs_mean = float(probs.mean().item())
                probs_std  = float(probs.std(unbiased=False).item()) if probs.numel() > 1 else 0.0
                probs_min  = float(probs.min().item())
                probs_max  = float(probs.max().item())

            # optimizer LR
            lr = float(opt.param_groups[0]["lr"])
            examples = int(feats.size(0))
            ex_per_s = float(examples / dt) if dt > 0 else float("inf")

            # save last train event so _log_train_status()
            self._last_train = {
                "train_calls": self.train_calls,
                "mode": mode + " " + str(success),
                "name": name,
                "loss": float(loss.item()),
                "grad_norm": grad_norm,
                "lr": lr,
                "step_time_s": float(dt),
                "examples_per_s": ex_per_s,
                "param_abs_sum_pre": param_abs_sum_pre,
                "param_abs_sum_post": param_abs_sum_post,
                "param_abs_sum_delta": param_abs_sum_delta,
                "logits_mean": logits_mean,
                "logits_std": logits_std,
                "logits_min": logits_min,
                "logits_max": logits_max,
                "probs_mean": probs_mean,
                "probs_std": probs_std,
                "probs_min": probs_min,
                "probs_max": probs_max,
            }

            # update EMAs
            self._ema_update(f"loss/{name}", float(loss.item()))
            self._ema_update(f"grad/{name}", grad_norm)
            self._ema_update(f"time/{name}", float(dt))
            
            self._last_train["success"] = bool(success)
            self._last_train["target"] = float(target_value)

            self._log_train_status()

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

        num_nodes = int(self.node_vecs.shape[0])
        Eplus = (num_nodes * (num_nodes - 1)) / 2

        model.train()
        opt.zero_grad(set_to_none=True)

        losses = []
        for (solution_uv, move_uv, success) in examples:
            if not move_uv:
                continue

            # canonicalize solution snapshot
            solution_set = {(min(u,v), max(u,v)) for (u,v) in solution_uv if u != v}

            degS, inS, sol_size = self._solution_context(solution_set, num_nodes)
            
            move_pairs = [(min(u,v), max(u,v)) for (u,v) in move_uv if u != v]
            
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

            # if move is big, sub-sample edges
            if len(move_pairs) > max_edges_per_move:
                pairs_tensor_all = torch.as_tensor(move_pairs, dtype=torch.long, device=self.device)
                feats_all = self._pair_features(pairs_tensor_all, degS=degS, inS=inS, sol_size=sol_size, Eplus=Eplus)

                with torch.no_grad():
                    logits_all = model(feats_all)  # (m,)
                    probs_all = torch.sigmoid(logits_all)

                if (not success) and hard_within_move_for_neg:
                    # for negatives: keep edges model currently thinks are good (hard negatives)
                    idx = torch.topk(probs_all, k=max_edges_per_move, largest=True).indices
                else:
                    # for positives: random subset keeps diversity
                    idx = torch.randperm(len(move_pairs), device=self.device)[:max_edges_per_move]

                pairs_tensor = pairs_tensor_all[idx]
                feats = feats_all[idx]
            else:
                pairs_tensor = torch.as_tensor(move_pairs, dtype=torch.long, device=self.device)
                feats = self._pair_features(pairs_tensor, degS=degS, inS=inS, sol_size=sol_size, Eplus=Eplus)

            logits = model(feats)  # (m,)
            move_logit = logits.mean()  # mean pooling

            target = torch.tensor([1.0 if success else 0.0], device=self.device)
            loss = F.binary_cross_entropy_with_logits(move_logit.view(1), target)

            # downweight negatives so they don't dominate
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


    # ---------- Online update ----------

    def update_additions(self, chosen_pairs, success: bool):
        if not chosen_pairs:
            return
        self._train_online("add", chosen_pairs, success=success)

    def update_removals(self, chosen_pairs, success: bool):
        if not chosen_pairs:
            return
        self._train_online("remove", chosen_pairs, success=success)
        
    def update_addition_moves(self, examples: list[tuple[list[tuple[int,int]], list[tuple[int,int]], bool]]):
        self._train_online_moves("add", examples)
        
    def update_removal_moves(self, examples: list[tuple[list[tuple[int,int]], list[tuple[int,int]], bool]]):
        self._train_online_moves("remove", examples)

    

    
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