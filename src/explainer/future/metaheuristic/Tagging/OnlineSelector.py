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


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, epsilon: float = 1e-5):
        """
        A class for Prioritized Experience Replay using total rewards as priority.
        
        Parameters:
        - capacity: Maximum number of experiences in the buffer.
        - epsilon: Small value to ensure priorities are never zero, preventing NaNs.
        """
        self.capacity = capacity
        self.epsilon = epsilon  # Small value to ensure priorities are not zero
        self.buffer = []
        self.rewards = []  # Store rewards as priorities
        self.pos = 0

    def add(self, experience, reward):
        """
        Add an experience to the buffer with priority based on its reward.
        If the pair already exists, increment/decrement its priority based on the new reward.
        """
        # Check if the experience already exists in the buffer
        for i, (exp, current_reward) in enumerate(zip(self.buffer, self.rewards)):
            if exp == experience:
                # If the experience exists, update its priority
                new_priority = current_reward + reward  # Cumulative reward-based priority
                self.rewards[i] = max(new_priority, self.epsilon)  # Ensure priority is never zero
                return  # Exit the function since we updated the existing pair
        
        # If experience is not in the buffer, add it
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.rewards.append(reward)
        else:
            # If buffer is full, replace the oldest experience
            self.buffer[self.pos] = experience
            self.rewards[self.pos] = reward

        # Ensure reward (priority) is never zero
        self.rewards[self.pos] = max(self.rewards[self.pos], self.epsilon)

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample experiences from the buffer according to their priority."""
        # Normalize priorities to sum to 1
        priorities = np.array(self.rewards)
        probs = priorities / priorities.sum()

        # Check for NaNs in probabilities
        if np.any(np.isnan(probs)) or np.any(probs < 0):
            print("Warning: NaN probabilities encountered!")
            probs = np.ones_like(probs) / len(probs)  # Fallback to uniform sampling

        # Sample indices based on probability distribution
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)

        batch = [self.buffer[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** -1  # Importance-sampling weights
        weights /= weights.max()  # Normalize weights

        return batch, indices, rewards, weights

    def update_priorities(self, indices, rewards):
        """Update the priorities of sampled experiences based on new rewards."""
        for i, reward in zip(indices, rewards):
            self.rewards[i] = max(reward, self.epsilon)  # Ensure priority is never zero

            
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
        batch_size: int = 64,
        replay_capacity: int = 50000,
    ):
        self.example_count = 0
        self.k = k
        self.input_dim = 4 * k + 4
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.exploration_prob = exploration_prob
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity

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

        # Replay buffers: list of (u, v, reward)
        self.replay_add: list[tuple[int, int, float]] = []
        self.replay_remove: list[tuple[int, int, float]] = []
        
        # Track performance for dynamic exploration adjustment
        self.performance_counter = 0
        self.last_best_size = None
        
        # Initialize a prioritized replay buffer for add and remove actions
        self.replay_add = PrioritizedReplayBuffer(self.replay_capacity)
        self.replay_remove = PrioritizedReplayBuffer(self.replay_capacity)
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
        assert feats.shape[1] == self.input_dim
        return feats

    # ---------- Sampling logic ----------

    def _sample_indices(self, probs_np: np.ndarray, X: int):
        """
        Sample X indices using epsilon-greedy over probs_np.
        probs_np: array of length M, ideally in [0,1]
        """
        n = len(probs_np)
        if n <= X:
            # not enough candidates to worry about sampling
            return list(range(n))

        # Exploration: pure random choice
        if np.random.rand() < self.exploration_prob:
            return random.sample(range(n), X)

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

        if mode == "remove":
            # Candidates are exactly the edges in the current solution
            candidate_pairs = list(solution_set)

        elif mode == "add":
            # Candidates are all pairs in the universe that are NOT in the current solution
            candidate_pairs = []
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    if (u, v) not in solution_set:
                        candidate_pairs.append((u, v))

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
            feats = self._pair_features(pairs_tensor)          # (M, input_dim)
            logits = model(feats)                              # (M,)
            probs = torch.sigmoid(logits).cpu().numpy()        # P(success)

        # Sample indices according to probs (epsilon-greedy)
        idx = self._sample_indices(probs, X)
        return [candidate_pairs[i] for i in idx]

    def propose_additions(self, solution, X: int):
        """
        Choose X candidate pairs to ADD.
        `solution` is the current solution as a list of (u, v) pairs.
        """
        self._update_exploration_prob()
        return self._propose(solution, X, mode="add")

    def propose_removals(self, solution, X: int):
        """
        Choose X candidate pairs to REMOVE.
        `solution` is the current solution as a list of (u, v) pairs.
        """
        self._update_exploration_prob()
        return self._propose(solution, X, mode="remove")
    
    
    def _update_exploration_prob(self):
        """
        Update exploration probability dynamically based on performance.
        """
        if self.last_best_size is None:
            return

        # Check if the current best solution size has improved
        improvement = self.last_best_size - self.best_size if self.last_best_size else 0
        if improvement < self.performance_threshold:
            self.performance_counter += 1
        else:
            self.performance_counter = 0  # Reset if there is improvement

        # If stuck for a while, increase exploration
        if self.performance_counter >= self.exploration_update_interval:
            self.exploration_prob = min(1.0, self.exploration_prob * 1.05)

        # Decay exploration otherwise
        else:
            self.exploration_prob = max(0.05, self.exploration_prob * 0.995)

        # Ensure that exploration_prob stays within bounds
        self.exploration_prob = max(0.05, min(1.0, self.exploration_prob))

    # ---------- Replay buffer & mini-batch training ----------

    def _append_replay(self, mode: str, pairs, reward: float):
        """Store (pair, reward) experiences in the replay buffer."""
        if not pairs:
            return

        buf = self.replay_add if mode == "add" else self.replay_remove
        r = float(reward)

        for (u, v) in pairs:
            buf.add((u, v), r)  # Add experience with its corresponding TD error


    def _train_from_replay(self, mode: str):
        """Sample a mini-batch from replay buffer and do one optimizer step."""
        buf = self.replay_add if mode == "add" else self.replay_remove
        if not buf:
            return

        batch_size = min(self.batch_size, len(buf.buffer))
        if batch_size <= 1:
            return
        
        batch, indices, rewards, weights = buf.sample(batch_size)
         # Unpack the batch into pairs and rewards
        pairs = [(b[0], b[1]) for b in batch]  # List of (u, v) pairs

        # Convert pairs and rewards to tensors
        pairs_tensor = torch.as_tensor(pairs, dtype=torch.long, device=self.device)
        feats = self._pair_features(pairs_tensor)
        labels = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        
        model = self.model_add if mode == "add" else self.model_remove
        opt = self.opt_add if mode == "add" else self.opt_remove
        
        model.train()
        opt.zero_grad()
        
        logits = model(feats)
        loss = self.loss_fn(logits, labels)

        # Apply importance sampling weights to the loss
        weighted_loss = (loss * torch.tensor(weights, dtype=torch.float32, device=self.device)).mean()

        weighted_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()

        # Update priorities based on TD errors
        td_errors = logits - labels
        buf.update_priorities(indices, td_errors.cpu().detach().numpy())

    
    def _adjust_buffer_size(self):
        """Dynamically adjust the buffer size based on performance."""
        if len(self.replay_add.buffer) > self.replay_capacity * 0.8:
            self.replay_capacity = max(1000, self.replay_capacity - 1000)

    # ---------- Online update (reward-based) ----------

    def update_additions(self, chosen_pairs, reward: float):
        """
        Online training for ADD actions, using a scaled reward in [0,1].
        """
        if not chosen_pairs:
            return
        
        self._append_replay("add", chosen_pairs, reward)

        self._train_from_replay("add")


    def update_removals(self, chosen_pairs, reward: float):
        """
        Online training for REMOVE actions, using a scaled reward in [0,1].
        """
        if not chosen_pairs:
            return
        
        self._append_replay("remove", chosen_pairs, reward)

        # Train the model from replay
        self._train_from_replay("remove")

    # ---------- Reward computation ----------
    def compute_reward(found_: bool) -> float:
        """
        Map (valid move, size improvement) -> reward in [0, 1].
        """

        if not found_:
            return -1
        return 1.0

    
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
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
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

        batch_size = ckpt.get("batch_size", 64)
        replay_capacity = ckpt.get("replay_capacity", 50000)

        obj = cls(
            k=ckpt["k"],
            hidden_dim=ckpt["hidden_dim"],
            num_hidden_layers=ckpt["num_hidden_layers"],
            lr=lr,
            exploration_prob=ckpt["exploration_prob"],
            device=device,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
        )
        obj.model_add.load_state_dict(ckpt["model_add"])
        obj.model_remove.load_state_dict(ckpt["model_remove"])
        obj.opt_add.load_state_dict(ckpt["opt_add"])
        obj.opt_remove.load_state_dict(ckpt["opt_remove"])
        return obj

    # ---------- logging ----------
    def _param_stats(self, model: nn.Module) -> Dict[str, float]:
        """Fast-ish summary stats to detect changes without dumping tensors."""
        abs_sum = 0.0
        sq_sum = 0.0
        max_abs = 0.0
        n = 0

        with torch.no_grad():
            for p in model.parameters():
                if p is None:
                    continue
                t = p.detach()
                if t.numel() == 0:
                    continue
                tf = t.float()
                a = tf.abs()
                abs_sum += a.sum().item()
                sq_sum += (tf * tf).sum().item()
                max_abs = max(max_abs, a.max().item())
                n += tf.numel()

        l2 = math.sqrt(sq_sum) if sq_sum > 0 else 0.0
        mean_abs = abs_sum / max(n, 1)
        return {
            "param_numel": float(n),
            "param_l2": float(l2),
            "param_mean_abs": float(mean_abs),
            "param_max_abs": float(max_abs),
            # a simple "checksum" that should move if weights move:
            "param_abs_sum": float(abs_sum),
        }

    def _bn_running_stats(self, model: nn.Module) -> Dict[str, float]:
        """BatchNorm running stats are a good 'is training doing anything?' signal."""
        means = []
        vars_ = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                if hasattr(m, "running_mean") and m.running_mean is not None:
                    means.append(m.running_mean.detach().float().mean().item())
                if hasattr(m, "running_var") and m.running_var is not None:
                    vars_.append(m.running_var.detach().float().mean().item())

        return {
            "bn_running_mean_mean": float(sum(means) / max(len(means), 1)) if means else float("nan"),
            "bn_running_var_mean": float(sum(vars_) / max(len(vars_), 1)) if vars_ else float("nan"),
            "bn_layers": float(len(means)),
        }

    def log_model_status(
        self,
        mode: str = "add",
        step: Optional[int] = None,
        loss: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """
        Logs + returns a dict describing whether the model is changing.
        Call it occasionally (e.g. every N training steps).

        mode: "add" or "remove"
        step: optional global step counter
        loss: last loss value if you have it
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        model = self.model_add if mode == "add" else self.model_remove
        opt = self.opt_add if mode == "add" else self.opt_remove
        buf = self.replay_add if mode == "add" else self.replay_remove

        stats = {}
        stats.update(self._param_stats(model))
        stats.update(self._bn_running_stats(model))

        lr = opt.param_groups[0]["lr"] if opt.param_groups else float("nan")
        stats.update({
            "mode": mode,
            "step": step if step is not None else -1,
            "loss": float(loss) if loss is not None else float("nan"),
            "lr": float(lr),
            "exploration_prob": float(self.exploration_prob),
            "replay_size": float(len(getattr(buf, "buffer", []))),
            "model_training_flag": float(model.training),
        })

        # Compare against last snapshot to see if weights actually moved
        if not hasattr(self, "_last_status"):
            self._last_status = {"add": None, "remove": None}

        last = self._last_status.get(mode)
        if last is None:
            stats["delta_param_abs_sum"] = float("nan")
            stats["delta_param_l2"] = float("nan")
            stats["changed"] = False
        else:
            stats["delta_param_abs_sum"] = stats["param_abs_sum"] - last["param_abs_sum"]
            stats["delta_param_l2"] = stats["param_l2"] - last["param_l2"]
            stats["changed"] = abs(stats["delta_param_abs_sum"]) > 1e-6

        self._last_status[mode] = stats.copy()

        if extra:
            stats.update(extra)

        logger.info(
            "[%s] step=%s loss=%s lr=%.3g replay=%d eps=%.3f "
            "abs_sum=%.6g (Δ%.3g) l2=%.6g bn_mean=%.3g bn_var=%.3g changed=%s",
            mode,
            stats["step"],
            "nan" if math.isnan(stats["loss"]) else f"{stats['loss']:.6g}",
            stats["lr"],
            int(stats["replay_size"]),
            stats["exploration_prob"],
            stats["param_abs_sum"],
            stats["delta_param_abs_sum"] if not math.isnan(stats["delta_param_abs_sum"]) else float("nan"),
            stats["param_l2"],
            stats["bn_running_mean_mean"],
            stats["bn_running_var_mean"],
            stats["changed"],
        )

        return stats