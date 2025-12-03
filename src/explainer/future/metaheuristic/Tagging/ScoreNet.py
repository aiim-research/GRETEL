from __future__ import annotations
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ScoreNet(nn.Module):
    """
    Small MLP that takes pair features and outputs a logit (score).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_hidden_layers: int = 2):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))  # output logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        out = self.net(x)
        return out.squeeze(-1)  # (batch,)


class OnlineNNEdgeSelector:
    """
    Online neural model to bias which edges to add/remove.

    Usage:
        selector = OnlineNNEdgeSelector(k=node_vecs.shape[1])
        selector.set_node_vectors(node_vecs)  # np.ndarray (num_nodes, k)

        # propose:
        chosen_add_pairs = selector.propose_additions(candidate_pairs, X)
        # after evaluating:
        selector.update_additions(chosen_add_pairs, success=True/False)
    """

    def __init__(
        self,
        k: int, # dimension of node vectors
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        lr: float = 1e-3,
        exploration_prob: float = 0.2,
        device: torch.device | None = None,
    ):
        self.k = k
        self.input_dim = 4 * k + 4           # based on our feature design
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.exploration_prob = exploration_prob

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Separate models for add/remove actions
        self.model_add = ScoreNet(self.input_dim, hidden_dim, num_hidden_layers).to(self.device)
        self.model_remove = ScoreNet(self.input_dim, hidden_dim, num_hidden_layers).to(self.device)

        self.opt_add = optim.Adam(self.model_add.parameters(), lr=lr)
        self.opt_remove = optim.Adam(self.model_remove.parameters(), lr=lr)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.node_vecs: torch.Tensor | None = None  # will be set by set_node_vectors()

    # ---------- Node vectors & features ----------

    def set_node_vectors(self, node_vecs_np: np.ndarray):
        """
        Call this once (or whenever node vectors change).

        node_vecs_np: shape (num_nodes, k)
        """
        self.node_vecs_np = node_vecs_np
        assert node_vecs_np.shape[1] == self.k
        self.node_vecs = torch.as_tensor(
            node_vecs_np, dtype=torch.float32, device=self.device
        )

    def _pair_features(self, pairs_tensor: torch.Tensor) -> torch.Tensor:
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

        diff = v_u - v_v            # (M, k)
        prod = v_u * v_v            # (M, k)
        dot = (v_u * v_v).sum(dim=1, keepdim=True)        # (M, 1)
        norm_u = v_u.norm(dim=1, keepdim=True)            # (M, 1)
        norm_v = v_v.norm(dim=1, keepdim=True)            # (M, 1)
        dist = diff.norm(dim=1, keepdim=True)             # (M, 1)

        feats = torch.cat(
            [v_u, v_v, diff, prod, dot, norm_u, norm_v, dist],
            dim=1
        )
        assert feats.shape[1] == self.input_dim
        return feats

    # ---------- Sampling logic ----------

    def _sample_indices(self, probs_np: np.ndarray, X: int):
        """
        Sample X indices using epsilon-greedy over probs_np.
        probs_np: array of length M, in [0,1]
        """
        n = len(probs_np)
        if n <= X:
            return list(range(n))

        # Exploration: random choice
        if np.random.rand() < self.exploration_prob:
            return random.sample(range(n), X)

        # Exploitation: weighted by probs
        probs = np.clip(probs_np, 0.0, 1.0)
        total = probs.sum()
        if total <= 0.0:
            # Degenerate case -> uniform
            return random.sample(range(n), X)

        probs = probs / total
        idx = np.random.choice(np.arange(n), size=X, replace=False, p=probs)
        return idx.tolist()

    # ---------- Propose pairs ----------

    def _propose(self, candidate_pairs, X: int, mode: str):
        """
        Internal: propose X pairs for given mode ('add' or 'remove').
        candidate_pairs: list of (u, v)
        """
        if not candidate_pairs or X <= 0:
            return []

        M = len(candidate_pairs)
        if M <= X:
            return list(candidate_pairs)

        pairs_tensor = torch.as_tensor(candidate_pairs, dtype=torch.long, device=self.device)

        model = self.model_add if mode == "add" else self.model_remove
        model.eval()
        with torch.no_grad():
            feats = self._pair_features(pairs_tensor)  # (M, input_dim)
            logits = model(feats)                     # (M,)
            probs = torch.sigmoid(logits).cpu().numpy()  # P(success)

        idx = self._sample_indices(probs, X)
        return [candidate_pairs[i] for i in idx]

    def propose_additions(self, candidate_pairs, X: int):
        """
        Choose X candidate pairs to ADD.
        """
        return self._propose(candidate_pairs, X, mode="add")

    def propose_removals(self, candidate_pairs, X: int):
        """
        Choose X candidate pairs to REMOVE.
        """
        return self._propose(candidate_pairs, X, mode="remove")

    # ---------- Online update ----------

    def _update(self, chosen_pairs, success: bool, mode: str):
        """
        Internal: update model from chosen pairs & global success flag.
        """
        if not chosen_pairs:
            return

        pairs_tensor = torch.as_tensor(chosen_pairs, dtype=torch.long, device=self.device)
        feats = self._pair_features(pairs_tensor)

        labels = torch.full(
            (feats.shape[0],),
            float(bool(success)),
            dtype=torch.float32,
            device=self.device,
        )

        model = self.model_add if mode == "add" else self.model_remove
        opt = self.opt_add if mode == "add" else self.opt_remove

        model.train()
        opt.zero_grad()
        logits = model(feats)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        opt.step()

    def update_additions(self, chosen_pairs, success: bool):
        """
        Online training step for ADD actions.
        """
        self._update(chosen_pairs, success, mode="add")

    def update_removals(self, chosen_pairs, success: bool):
        """
        Online training step for REMOVE actions.
        """
        self._update(chosen_pairs, success, mode="remove")

    # ---------- Save / load ----------

    def save(self, path: str):
        """
        Save model and optimizer states to a file.
        (Node vectors are NOT saved; set them again with set_node_vectors.)
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
