import warnings
import hnswlib
import numpy as np
from typing import Iterable, Optional

class ANNIndexWeighted:
    """
    HNSW ANN index over weighted-cosine space.
    """

    def __init__(
        self,
        X: np.ndarray,                 # shape (N, K), float32 preferred
        ef_construction: int = 100,
        ef: int = 100,
        initial_weight: float = 0.5,   # start all dims at 0.5
        seed: Optional[int] = None,
        weights: Optional[np.ndarray] = None
    ):
        assert X.ndim == 2, "X must be (N,K)"
        self.X_raw = X.astype(np.float32, copy=True)
        self.N, self.K = self.X_raw.shape   # pairs of nodes, each with K-dim vector
        
        if(weights is not None):
            assert weights.shape == (self.K,), "weights must have shape (K,)"
            print("inherited weights: ", np.round(weights, 2))
            self.w = weights.astype(np.float32)
        else:
            # weights in [0,1], start at 0.5
            self.w = np.full(self.K, float(initial_weight), dtype=np.float32)
        self.ef_construction = ef_construction
        self.ef = ef
        self.seed = seed

        # build first index
        self._rebuild_index()

    # -------- public API --------

    def set_weights(self, w: np.ndarray):
        """Set full weight vector (shape K), values in [0,1], then rebuild."""
        w = np.asarray(w, dtype=np.float32)
        assert w.shape == (self.K,), "w must have shape (K,)"
        # clip to [0,1]
        np.clip(w, 0.0, 1.0, out=w)
        self.w = w
        self._rebuild_index()
    
    def recalculate_weights_from_indices(
        self,
        added: set[int],
        removed: set[int],
        old_weight_share: float = 0.5,  # share for current weights in the blend
    ) -> None:
        """
        Recompute self.w using both "added" (upweight) and "removed" (downweight).
        - Per-dimension 'added' signal is mean(|X_raw[added, j]|), normalized to [0,1].
        - Per-dimension 'removed' penalty is mean(|X_raw[removed, j]|) * current_weight_j,
        normalized to [0,1] before applying the current-weight modulation.
        - Combined signal = scaled_added - (scaled_removed * current_weight).
        - Map combined signal from [-1,1] -> [0,1] via (signal+1)/2.
        - Blend: new_w = old_weight_share * old_w + (1-old_weight_share) * mapped_signal.
        Always clips to [0,1] and rebuilds the index.
        """
        def _normalize01(arr: np.ndarray) -> np.ndarray:
            # Normalize a 1D nonnegative array to [0,1] robustly
            maxv = float(np.max(arr))
            if not np.isfinite(maxv) or maxv <= 1e-12:
                return np.zeros_like(arr, dtype=np.float32)
            return (arr / maxv).astype(np.float32)

        # Coerce to sorted np arrays (and allow empty)
        added_idx = np.asarray(sorted(added), dtype=np.int32) if added else np.empty(0, dtype=np.int32)
        removed_idx = np.asarray(sorted(removed), dtype=np.int32) if removed else np.empty(0, dtype=np.int32)

        # Fast exit if both empty
        if added_idx.size == 0 and removed_idx.size == 0:
            return

        # Bounds checks
        if added_idx.size and np.any((added_idx < 0) | (added_idx >= self.N)):
            raise IndexError("Some indices in 'added' are out of bounds.")
        if removed_idx.size and np.any((removed_idx < 0) | (removed_idx >= self.N)):
            raise IndexError("Some indices in 'removed' are out of bounds.")

        # ----- Per-dimension signals -----
        # Added signal: larger magnitudes -> higher weight
        if added_idx.size:
            X_add = self.X_raw[added_idx]                      # (m_add, K)
            score_add = np.mean(np.abs(X_add), axis=0)         # (K,)
            scaled_add = _normalize01(score_add)               # [0,1]
        else:
            scaled_add = np.zeros(self.K, dtype=np.float32)

        # Removed signal: larger magnitudes -> stronger penalty
        if removed_idx.size:
            X_rem = self.X_raw[removed_idx]                    # (m_rem, K)
            score_rem = np.mean(np.abs(X_rem), axis=0)         # (K,)
            scaled_rem = _normalize01(score_rem)               # [0,1]
            # Modulate penalty by current weights so already-important dims lose more
            penalty = scaled_rem * self.w                      # [0,1]
        else:
            penalty = np.zeros(self.K, dtype=np.float32)

        # ----- Combine into a single proposal in [0,1] -----
        # signal in [-1,1]
        signal = np.clip(scaled_add - penalty, -1.0, 1.0).astype(np.float32)
        # map to [0,1]
        proposal = (signal + 1.0) * 0.5

        # ----- Blend with current weights -----
        alpha = float(old_weight_share)
        alpha = min(1.0, max(0.0, alpha))  # clamp
        new_w = alpha * self.w + (1.0 - alpha) * proposal
        new_w = np.clip(new_w.astype(np.float32), 0.0, 1.0)

        # Apply & rebuild to reflect new metric
        self.w = new_w
        self._rebuild_index()

        
        
    def update_weights(self, idxs: Iterable[int], vals: Iterable[float]):
        """Update some dimensions and rebuild. idxs are dim indices [0..K-1]."""
        for i, v in zip(idxs, vals):
            if 0 <= i < self.K:
                self.w[i] = np.float32(min(1.0, max(0.0, v)))
            else:
                raise IndexError(f"dimension {i} out of range [0,{self.K-1}]")
        self._rebuild_index()

    def get_weights(self) -> np.ndarray:
        return self.w.copy()

    
    # -------- Adding --------
    def neighbors_of_S_union(
        self,
        S: set[int],
        top_k: int,
    ) -> tuple[set[int], set[int]]:
        """
        For each s in S, retrieve its neighbors in the weighted-cosine space,
        union all neighbor labels (deduped), then return a RANDOM batch of size
        `top_k` from that union (excluding S itself).

        If fewer than `top_k` unique candidates exist, returns all of them and emits a warning.

        Returns:
            (full_solution_indices, added_indices) as sets of 0-based dataset indices.
        """
        # if rng is None:
        rng = np.random.default_rng()

        if top_k <= 0 or not S:
            return set(S), set()

        # Ask a bit more per query to offset filtering of S itself
        per_query_k = min(self.N, top_k + len(S))

        labels_union: set[int] = set()

        for s in S:  # iterate the set directly; no need to build an array
            # weighted + normalized query vector
            q = self._normalize(self._weighted(self.X_raw[s]))
            local_labels, _ = self.index.knn_query(q, k=per_query_k)

            # Deduplicate across queries and exclude S; also guard invalid labels
            for lab in local_labels[0]:
                li = int(lab)
                if li >= 0 and li < self.N and li not in S:
                    labels_union.add(li)

        if not labels_union:
            return set(S), set()

        candidates = np.fromiter(labels_union, dtype=np.int32)

        k = min(top_k, candidates.size)
        if k < top_k:
            warnings.warn(
                f"Only {k} unique candidates available (requested top_k={top_k}). Returning all.",
                RuntimeWarning,
            )

        # Sample without replacement; returns an ndarray
        picked = rng.choice(candidates, size=k, replace=False)

        added = set(map(int, picked))
        full = set(S) | added
        return full, added


    def neighbors_of_centroid(
        self,
        S: set[int],
        top_k: int,
    ) -> tuple[set[int], set[int]]:
        """
        Find neighbors near the weighted centroid of rows in S.
        Returns (full_solution_indices, added_indices) as sets of 0-based dataset indices.
        """
        # Work in arrays, but avoid Python lists
        S_arr = np.fromiter(S, dtype=np.int32, count=len(S))

        top_k_per_query = min(self.N, top_k + S_arr.size)

        # Weighted centroid, then normalize
        centroid = self._weighted(self.X_raw[S_arr]).mean(axis=0)
        centroid = self._normalize(centroid)

        labels, _ = self.index.knn_query(centroid, k=top_k_per_query)
        result = labels[0]  # np.ndarray of candidate indices

        # Keep ANN order: use np.isin mask (not setdiff1d, which sorts)
        mask = ~np.isin(result, S_arr, assume_unique=False)
        filtered = result[mask]

        # Take top_k after exclusion
        added = filtered[:top_k]

        # Build sets without creating intermediate Python lists
        added_set = set(map(int, added))
        kept_set = set(map(int, S_arr))
        full_set = kept_set | added_set

        return full_set, added_set

    
    
    
    # -------- Removing --------
    
    def prune_farthest_in_S(
        self,
        S: set[int],                        # set of 0-based dataset indices
        remove_k: int,                      # how many to drop
        temperature: float = 0.3,           # lower = greedier, 0 -> deterministic farthest
    ) -> tuple[set[int], set[int]]:
        """
        Stochastically remove `remove_k` elements from S, biased toward those farthest
        from S's weighted, L2-normalized centroid (cosine distance in normalized space).

        - temperature <= 0  -> deterministic: drop the farthest `remove_k`.
        - higher temperature -> more randomness.

        Returns:
            (kept_indices, removed_indices) as sets of dataset indices.
        """
        # Stable ordering for reproducibility / deterministic tie-breaking
        S_arr = np.fromiter(S, dtype=np.int32, count=len(S))

        n = S_arr.size
        if remove_k <= 0 or n == 0:
            return set(S_arr.tolist()), set()
        if remove_k >= n:
            return set(), set(S_arr.tolist())

        rng = np.random.default_rng()

        # Weighted + normalized reps of S
        Xs = self._normalize(self._weighted(self.X_raw[S_arr]))

        # Weighted centroid (normalized)
        centroid = self._normalize(Xs.mean(axis=0))

        # Cosine distance to centroid: d = 1 - cos_sim  (both normalized)
        sims = Xs @ centroid
        dists = 1.0 - sims  # in [0, 2] for normalized vectors

        if temperature <= 0.0:
            # pick farthest remove_k deterministically
            idx = np.argpartition(dists, -remove_k)[-remove_k:]
            idx = idx[np.argsort(dists[idx])[::-1]]  # sort descending distance
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            return set(S_arr[mask].tolist()), set(S_arr[idx].tolist())

        # Probabilistic removal via softmax over distances
        eps = 1e-12
        logits = dists / max(temperature, eps)
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.full_like(probs, 1.0 / probs.size)
        else:
            probs /= probs_sum

        # Sample without replacement according to probs
        idx_remove = rng.choice(n, size=remove_k, replace=False, p=probs)
        mask = np.ones(n, dtype=bool)
        mask[idx_remove] = False

        kept = set(map(int, S_arr[mask]))
        removed = set(map(int, S_arr[idx_remove]))

        return kept, removed

    
    
    
    # -------- internals --------
    def _weighted(self, X: np.ndarray) -> np.ndarray:
        """Apply per-dim weights: w ⊙ X. Accepts (K,) or (N,K)."""
        return X * self.w

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        eps = 1e-12
        if X.ndim == 1:
            return X / (np.linalg.norm(X) + eps)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    def _rebuild_index(self) -> None:
        """Rebuild HNSW on weighted+normalized data (distance definition changes)."""
        Xw = self._normalize(self._weighted(self.X_raw)).astype(np.float32, copy=False)

        self.index = hnswlib.Index(space="cosine", dim=self.K)
        self.index.init_index(
            max_elements=self.N,
            ef_construction=self.ef_construction,
            M=16,
            random_seed=(self.seed if self.seed is not None else 100),
        )
        labels = np.arange(self.N, dtype=np.int32)
        self.index.add_items(Xw, labels)
        self.index.set_ef(self.ef)