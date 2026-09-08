"""Bounded Local Search guided by the online edge selector, version 2.

Same search skeleton and strategy ordering as ``local_search_selection_net.py`` (remove first, then
swap, then add on a reduced solution, first-improvement acceptance, oracle budget), so that results
are comparable with the thesis runs and with the ordering ablation of the paper. What changes is how
the selector is fed and queried (see ``Tagging/OnlineSelectorV2.py``):

* dense soft labels from ``oracle.predict_proba`` (probability mass on a class other than the
  original one), at no extra oracle-call cost,
* listwise (softmax) updates over the moves tried in the same context, with logQ correction,
* per-edge labels from single-edge trials, replay buffer, Thompson-sampling exploration,
* correct context for the end-of-run positives (the added edges are scored in the context
  ``best \\ added``, not ``best``),
* the whole ``minimize`` call runs under the per-dataset checkpoint lock, so parallel workers do
  not overwrite each other's updates,
* ``selector_mode: "random"`` runs the identical skeleton with uniform proposals and no learning
  (the controlled ablation), and ``seed`` makes the run reproducible.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
from filelock import FileLock

from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from src.explainer.future.metaheuristic.Tagging.OnlineSelectorV2 import OnlineNNEdgeSelectorV2
from src.explainer.future.metaheuristic.Tagging.selector_monitor import SelectorMonitor
from src.explainer.future.metaheuristic.Tagging.vectors_builder import VectorsBuilder
from src.explainer.future.metaheuristic.local_search.cache import FixedSizeCache
from src.explainer.future.metaheuristic.manipulation.methods import (
    average_smoothing, feature_aggregation, heat_kernel_diffusion, identity,
    laplacian_regularization, random_walk_diffusion, weighted_smoothing,
)
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.comparison import get_edge_differences
from src.utils.seeding import set_seed


class BinaryModelV2:
    """Wraps the oracle and exposes (flipped, p_cf) per query, where p_cf is the probability mass
    the oracle assigns to any class other than the original one. One oracle call per query, exactly
    like ``BinaryModel.classify`` (``predict`` and ``predict_proba`` both count once)."""

    def __init__(self, oracle, instance: GraphInstance, dense: bool = True):
        self.oracle = oracle
        self.dense = dense
        self.initial = int(oracle.predict(instance))

    @staticmethod
    def _to_probs(out) -> np.ndarray:
        try:
            import torch
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
        except ImportError:  # pragma: no cover
            pass
        arr = np.asarray(out, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return arr
        if (arr < 0).any() or abs(arr.sum() - 1.0) > 1e-3:
            z = arr - arr.max()
            e = np.exp(z)
            arr = e / e.sum()
        return arr

    def classify(self, graph: GraphInstance) -> tuple[bool, float]:
        if not self.dense:
            flipped = int(self.oracle.predict(graph)) != self.initial
            return flipped, (1.0 if flipped else 0.0)
        probs = self._to_probs(self.oracle.predict_proba(graph))
        if probs.size == 0:
            flipped = int(self.oracle.predict(graph)) != self.initial
            return flipped, (1.0 if flipped else 0.0)
        label = int(np.argmax(probs))
        flipped = label != self.initial
        p_cf = 1.0 - float(probs[self.initial]) if self.initial < probs.size else 0.0
        if flipped:
            return True, 1.0
        return False, float(min(p_cf, 0.49))


class LocalSearch(ExplanationMinimizer):

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        p.setdefault('neigh_factor', 4)
        p.setdefault('runtime_factor', 4)
        p.setdefault('max_runtime', 50)
        p.setdefault('max_neigh', 30)
        p.setdefault('attributed', False)
        p.setdefault('max_oracle_calls', 10000)

    def init(self):
        super().init()
        self.logger = self.context.logger
        p = self.local_config['parameters']
        self.neigh_factor = p['neigh_factor']
        self.runtime_factor = p['runtime_factor']
        self.max_runtime = p['max_runtime']
        self.max_neigh = p['max_neigh']
        self.attributed = p['attributed']
        self.max_oracle_calls = p['max_oracle_calls']
        self.recompute_features = p.get('recompute_features', True)
        self.seed = p.get('seed', None)
        set_seed(self.seed)

        # v2 knobs
        self.selector_mode = p.get('selector_mode', 'learned')          # learned | random
        self.dense_reward = bool(p.get('dense_reward', True))
        self.flush_every = int(p.get('flush_every', 10))
        self.p_neg_keep = float(p.get('p_neg_keep', 0.3))               # multi-edge failures kept for BCE
        self.p_neg_keep_swap = float(p.get('p_neg_keep_swap', 0.15))    # swap failures (ambiguous credit)
        self.max_listwise_negs = int(p.get('max_listwise_negs', 24))
        self.adaptive_trials = bool(p.get('adaptive_trials', True))
        # v2b: ranked block removal with adaptive block size (x2 on success), descending ladder to 1
        self.block_removal = bool(p.get('block_removal', False))
        self.block_init = int(p.get('block_init', 8))
        # v2b: alternate ranked windows with uniformly random blocks in the removal rungs, so an
        # uninformative or adversarial ranking can never do worse than plain LBS on that sweep
        self.block_random_mix = bool(p.get('block_random_mix', False))
        self.block_size = self.block_init
        # v2b: which swap outcomes feed the selector (add credit in a swap is ambiguous)
        self.swap_record_add = bool(p.get('swap_record_add', True))
        self.swap_record_remove = bool(p.get('swap_record_remove', True))
        self.flush_every_add = int(p.get('flush_every_add', p.get('flush_every', 10)))
        # v2b: like plain LBS, a failed outer iteration does not end the instance while n > 0 and the
        # global oracle budget is not exhausted (v1/v2 quit after the first failed outer iteration)
        self.retry_outer = bool(p.get('retry_outer', False))
        # v2c: explicit per-outer-iteration budget shares per phase (fractions of the local budget
        # max_oracle_calls * local_budget_frac). A phase that exhausts its share yields to the next one
        # instead of starving it (in v2/v2b the (-) sweep could eat the whole local budget).
        self.phase_budget = p.get('phase_budget', None)   # e.g. {"remove": 0.5, "swap": 0.25, "add": 0.25}
        self.local_budget_frac = float(p.get('local_budget_frac', 0.2))
        # v2c: shrink the block size after a removal sweep with no success (block grows only from successes)
        self.block_shrink_on_fail = bool(p.get('block_shrink_on_fail', False))
        # v2c: weight of block (multi-edge) removal examples relative to single-edge ones
        self.block_example_weight = float(p.get('block_example_weight', 1.0))
        # v2d: choose removal rung sizes with a Thompson-sampling bandit over {1,2,4,...}: value of a
        # size = edges removed per trial (size * P(success)); rung 1 is always tried. Stats persist
        # across the instances of the run. Replaces the doubling ladder when set.
        self.block_policy = p.get('block_policy', 'ladder')          # ladder | bandit | unlock
        self.unlock_min_trials = int(p.get('unlock_min_trials', 6))
        self.unlock_rate = float(p.get('unlock_rate', 0.5))
        # v2e: the single-edge rung uses whatever remains of the remove share instead of a fixed count
        self.fill_remove_share = bool(p.get('fill_remove_share', False))
        self.bandit_max_rungs = int(p.get('bandit_max_rungs', 3))
        self.bandit_min_value = float(p.get('bandit_min_value', 0.0))   # a block rung needs size*p_hat above this
        self.bandit_block_trials = int(p.get('bandit_block_trials', 0))  # 0 = schedule default
        self.rung_stats = {}                                         # size -> [successes, trials]
        # v2d: phase shares follow observed productivity (successes per call), with a floor
        self.adaptive_phase_budget = bool(p.get('adaptive_phase_budget', False))
        self.phase_share_floor = float(p.get('phase_share_floor', 0.1))
        self.phase_prod = {"remove": 0.02, "swap": 0.01, "add": 0.01}
        # v2d: try node-feature manipulation methods in order of past success (attributed mode)
        self.adaptive_method_order = bool(p.get('adaptive_method_order', False))
        self.method_success = None
        self.lock_whole_minimize = bool(p.get('lock_whole_minimize', True))
        self.model_dir = p.get('model_dir', 'models')
        self.model_tag = p.get('model_tag', 'v2')
        self.selector_kwargs = dict(p.get('selector', {}))
        self.monitor_log_every = int(p.get('monitor_log_every', 200))
        self.monitor_dir = p.get('monitor_dir', 'lab/output/selector_logs')
        self.monitor = None
        self.node_metrics = p.get('node_metrics', [
            "degree", "local_clustering", "triangle_count", "coreness",
            "avg_neighbor_degree", "ego_density", "pagerank",
        ])

        self.methods = [
            lambda data, features: identity(data, features),
            lambda data, features: average_smoothing(data, features, iterations=1),
            lambda data, features: weighted_smoothing(data, features, iterations=1),
            lambda data, features: laplacian_regularization(data, features, lambda_reg=0.01, iterations=1),
            lambda data, features: feature_aggregation(data, features, alpha=0.5, iterations=1),
            lambda data, features: heat_kernel_diffusion(data, features, t=0.5),
            lambda data, features: random_walk_diffusion(data, features, steps=1),
        ]

        self.size_ema = {"remove": 1.0, "add": 1.0}
        self.learning = (self.selector_mode == 'learned')

    # ------------------------------------------------------------------ entry

    def minimize(self, explaination: LocalGraphCounterfactualExplanation) -> DataInstance:
        instance = explaination.input_instance
        self.G = instance
        self.N = instance.num_nodes
        self.E = instance.num_edges
        self.EPlus = int((self.N * (self.N - 1)) / 2)

        self.M = BinaryModelV2(self.oracle, instance, dense=self.dense_reward)

        self.labels = [(i, j) for i in range(self.N - 1) for j in range(i + 1, self.N)]
        self.label_to_id = {uv: idx for idx, uv in enumerate(self.labels)}

        metrics_features = VectorsBuilder(self.node_metrics, self.G.data).X
        node_features = np.asarray(instance.node_features, dtype=np.float32)
        if node_features.ndim == 1:
            node_features = node_features.reshape(self.N, -1)
        total_features = np.concatenate((node_features, metrics_features), axis=1).astype(np.float32)
        k = total_features.shape[1]

        min_ctf = explaination.counterfactual_instances[0]
        _, diff_matrix = get_edge_differences(self.G, min_ctf)
        coords = np.where(diff_matrix == 1)
        filtered = [c for c in zip(coords[0], coords[1]) if c[0] < c[1]]
        actual = self.uv_to_id(filtered)
        if len(actual) == 0:
            self.logger.info("Initial solution size is 0")
            return min_ctf

        path, lock_path = self.model_paths(self.dataset.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lock = FileLock(lock_path)

        def run():
            self.selector = self.load_or_initialize_selector(path, k)
            self.selector.set_node_vectors(total_features)
            self.selector.set_base_graph(instance.data, directed=instance.directed)
            self.cache = FixedSizeCache(capacity=500000)
            if self.monitor is None:
                self.monitor = SelectorMonitor(self.logger, self.monitor_dir, self.dataset.name, self.model_tag,
                                               self.local_config['parameters'].get('fold_id', -1),
                                               self.selector_mode, log_every=self.monitor_log_every)
            self.monitor.reset_instance(getattr(instance, 'id', None), len(actual))
            result = self.get_approximation(actual, set(actual), min_ctf)
            self.monitor.end_instance(self._final_size, self.k, self.selector,
                                      extra={"calls_remove": self.phase_calls["remove"], "calls_swap": self.phase_calls["swap"],
                                             "calls_add": self.phase_calls["add"], "succ_remove": self.phase_success["remove"],
                                             "succ_swap": self.phase_success["swap"], "succ_add": self.phase_success["add"],
                                             "accepted_block_mean": (sum(self.accepted_sizes) / len(self.accepted_sizes)) if self.accepted_sizes else float("nan"),
                                             "accepted_block_max": max(self.accepted_sizes) if self.accepted_sizes else 0,
                                             "retries": self.retries,
                                             "rung_stats": json.dumps({int(k): v for k, v in sorted(self.rung_stats.items())}),
                                             "phase_shares": json.dumps({k: round(v, 3) for k, v in (self.phase_budget or {}).items()}),
                                             "method_success": json.dumps(self.method_success)})
            if self.learning:
                self.selector.flush()
                self.selector.save(path)
                self.logger.info("[selector] " + self.selector.stats_line())
            return result

        if self.learning and self.lock_whole_minimize:
            with lock:
                return run()
        return run()

    # --------------------------------------------------------------- selector

    def model_paths(self, dataset_id: str):
        path = os.path.join(self.model_dir, f"edge_selector_{self.model_tag}_{dataset_id}.pt")
        return path, path + ".lock"

    def load_or_initialize_selector(self, path: str, k: int) -> OnlineNNEdgeSelectorV2:
        kw = dict(self.selector_kwargs)
        kw.setdefault("seed", self.seed)
        if not self.learning:
            kw["acquisition"] = "random"
            kw["exploration_prob"] = 0.0
            return OnlineNNEdgeSelectorV2(k=k, **kw)
        if os.path.exists(path):
            try:
                sel = OnlineNNEdgeSelectorV2.load(path, overrides=kw)
                if sel.k == k:
                    return sel
                self.logger.warning("[selector] checkpoint k=%d != %d, starting a new one", sel.k, k)
            except Exception as e:
                self.logger.warning("[selector] could not load %s (%s), starting a new one", path, e)
        sel = OnlineNNEdgeSelectorV2(k=k, **kw)
        sel.save(path)
        return sel

    # ------------------------------------------------------------- main loop

    def _local_cap(self) -> float:
        return self.max_oracle_calls * self.local_budget_frac

    def _budget_exceeded(self) -> bool:
        return self.k > self.max_oracle_calls or self.k_local > self._local_cap()

    def _phase_exceeded(self, phase: str) -> bool:
        """True when the global budget, the local budget, or this phase's share of it is exhausted."""
        if self._budget_exceeded():
            return True
        if not self.phase_budget:
            return False
        share = float(self.phase_budget.get(phase, 1.0))
        return self.phase_calls_iter.get(phase, 0) > share * self._local_cap()

    def _update_productivity(self) -> None:
        for ph in ("remove", "swap", "add"):
            c = self.phase_calls_iter.get(ph, 0)
            if c > 0:
                r = self.phase_succ_iter.get(ph, 0) / float(c)
                self.phase_prod[ph] = 0.7 * self.phase_prod[ph] + 0.3 * r

    def _note_success(self, mode: str, size: int) -> None:
        self.size_ema[mode] = 0.7 * self.size_ema[mode] + 0.3 * float(size)
        self.phase_succ_iter[self.phase] = self.phase_succ_iter.get(self.phase, 0) + 1
        self.phase_success[self.phase] += 1
        if mode == "remove" and self.phase == "remove":
            self.accepted_sizes.append(int(size))
        if mode == "remove" and self.block_removal:
            self.block_size = max(1, 2 * int(size))
        if self.learning:
            self.selector.new_head()

    def _maybe_flush(self) -> None:
        if not self.learning:
            return
        for m, every in (("add", self.flush_every_add), ("remove", self.flush_every)):
            if len(self.selector.pending[m]) >= every:
                self.selector.train_step(m)

    def get_approximation(self, actual, best, min_ctf):
        self.logger.info("Initial solution size: " + str(len(actual)))
        result = min_ctf
        initial_solution = set(actual)
        n = min(self.max_runtime, self.runtime_factor * len(actual))
        self.k = 0
        self.k_local = 0
        self.block_size = self.block_init
        self.phase = "remove"
        self.phase_calls = {"remove": 0, "swap": 0, "add": 0}
        self.phase_success = {"remove": 0, "swap": 0, "add": 0}
        self.accepted_sizes = []
        self.retries = 0
        self.phase_calls_iter = {"remove": 0, "swap": 0, "add": 0}
        self.phase_succ_iter = {"remove": 0, "swap": 0, "add": 0}

        while n > 0:
            n -= 1
            self.k_local = 0
            self._update_productivity()
            self.phase_calls_iter = {"remove": 0, "swap": 0, "add": 0}
            self.phase_succ_iter = {"remove": 0, "swap": 0, "add": 0}
            if self.adaptive_phase_budget and self.phase_budget:
                tot = sum(self.phase_prod.values()) or 1.0
                f = self.phase_share_floor
                self.phase_budget = {ph: f + (1.0 - 3 * f) * self.phase_prod[ph] / tot for ph in ("remove", "swap", "add")}
            if len(best) == 1:
                break
            if self._budget_exceeded():
                self.logger.info(f"Oracle calls limit reached, global: {self.k}, local: {self.k_local}")
                break
            found = False
            actual = best

            # ---------------- (-) removal ----------------
            neg_moves, neg_logqs = [], []
            self.phase = "remove"
            self.monitor.sweep_reset("remove")
            for s, removed, _, sol_ctx, lq, pred, rank, M in self.edge_remove(actual):
                if self._phase_exceeded("remove"):
                    break
                if self.cache.contains(s):
                    continue
                self.cache.add(s)
                found_, inst, y = self.evaluate(s)
                st = self.rung_stats.setdefault(len(removed), [0, 0]); st[1] += 1; st[0] += int(found_)
                self.monitor.observe("remove", pred, y)
                if found_:
                    self.monitor.success("remove", rank, M)
                self.monitor.maybe_log(self.k, len(best), self.selector)
                sol_uv = self.id_to_uv(sol_ctx)
                removed_uv = self.id_to_uv(removed)
                if self.learning:
                    if found_ or len(removed) == 1 or random.random() < self.p_neg_keep:
                        self.selector.record("remove", sol_uv, removed_uv, y,
                                             weight=1.0 if len(removed) == 1 else self.block_example_weight)
                    if found_:
                        if neg_moves:
                            self.selector.update_listwise("remove", sol_uv, removed_uv, neg_moves,
                                                          neg_logq=neg_logqs, pos_logq=lq)
                        self.selector.train_step("remove")
                    else:
                        self._keep_neg(neg_moves, neg_logqs, removed_uv, lq)
                    self._maybe_flush()
                if found_:
                    found = True
                    best = s
                    actual = s
                    result = inst
                    self._note_success("remove", len(removed))
                    n = min(self.max_runtime, self.runtime_factor * len(actual))
                    break
            if found:
                self.logger.info("============> (-) Found solution with size: " + str(len(actual)))
                continue
            if self.block_removal and self.block_shrink_on_fail:
                self.block_size = max(1, self.block_size // 2)

            # ---------------- reduce, then (=) swap and (+) add ----------------
            half = int(len(actual) / 2)
            reduce = min(half, random.randint(1, max(1, half * 4)))
            actual = self.reduce_random(best, reduce)
            found = False

            while len(best) - len(actual) > 1:
                if self._budget_exceeded():
                    break
                if self.phase_budget and self._phase_exceeded("swap") and self._phase_exceeded("add"):
                    break
                n -= 1

                self.phase = "swap"
                self.monitor.sweep_reset("remove"); self.monitor.sweep_reset("add")
                for s, removed, added, sol_ctx, temp_ctx, pred_r, pred_a in self.edge_swap(actual):
                    if self._phase_exceeded("swap"):
                        break
                    if self.cache.contains(s):
                        continue
                    self.cache.add(s)
                    found_, inst, y = self.evaluate(s)
                    self.monitor.observe("remove", pred_r, y); self.monitor.observe("add", pred_a, y)
                    if found_:
                        self.monitor.success("remove"); self.monitor.success("add")
                    self.monitor.maybe_log(self.k, len(best), self.selector)
                    if self.learning:
                        keep = found_ or random.random() < self.p_neg_keep_swap
                        if keep and removed and self.swap_record_remove:
                            self.selector.record("remove", self.id_to_uv(sol_ctx), self.id_to_uv(removed), y,
                                                 weight=1.0 if found_ else 0.5)
                        if keep and added and self.swap_record_add:
                            self.selector.record("add", self.id_to_uv(temp_ctx), self.id_to_uv(added), y,
                                                 weight=1.0 if found_ else 0.5)
                        if found_:
                            self.selector.flush()
                        self._maybe_flush()
                    if found_:
                        found = True
                        best = s
                        actual = s
                        result = inst
                        self._note_success("remove", len(removed))
                        self.phase_success["swap"] -= 1   # counted once for the swap, not twice
                        self._note_success("add", len(added))
                        n = min(self.max_runtime, self.runtime_factor * len(actual))
                        break
                if found:
                    self.logger.info("============> (=) Found solution with size: " + str(len(actual)))
                    break

                actual = self.reduce_random(best, len(actual))
                neg_moves, neg_logqs = [], []
                self.phase = "add"
                self.monitor.sweep_reset("add")
                for s, _, added, sol_ctx, lq, pred, rank, M in self.edge_add(actual, best):
                    if self._phase_exceeded("add"):
                        break
                    if self.cache.contains(s):
                        continue
                    self.cache.add(s)
                    found_, inst, y = self.evaluate(s)
                    self.monitor.observe("add", pred, y)
                    if found_:
                        self.monitor.success("add", rank, M)
                    self.monitor.maybe_log(self.k, len(best), self.selector)
                    sol_uv = self.id_to_uv(sol_ctx)
                    added_uv = self.id_to_uv(added)
                    if self.learning:
                        if found_ or len(added) == 1 or random.random() < self.p_neg_keep:
                            self.selector.record("add", sol_uv, added_uv, y)
                        if found_:
                            if neg_moves:
                                self.selector.update_listwise("add", sol_uv, added_uv, neg_moves,
                                                              neg_logq=neg_logqs, pos_logq=lq)
                            self.selector.train_step("add")
                        else:
                            self._keep_neg(neg_moves, neg_logqs, added_uv, lq)
                        self._maybe_flush()
                    if found_:
                        found = True
                        best = s
                        actual = s
                        result = inst
                        self._note_success("add", len(added))
                        n = min(self.max_runtime, self.runtime_factor * len(actual))
                        break
                if found:
                    self.logger.info("============> (+) Found solution with size: " + str(len(actual)))
                    break

                to_expand = int((len(best) - len(actual)) / 2) + 1
                expand = len(actual) + min(to_expand, random.randint(1, to_expand * 4))
                if expand > len(best):
                    break
                actual = self.reduce_random(best, expand)

            if found:
                continue
            if self.retry_outer and self.k <= self.max_oracle_calls:
                self.retries += 1
                self.logger.info(f"outer iteration without improvement (k={self.k}, k_local={self.k_local}), retrying (n={n})")
                continue
            break

        self.logger.info("Oracle calls: " + str(self.k))
        if self.oracle.predict(result) == self.oracle.predict(self.G):
            self.logger.info("ERROR, returning non ctf")

        self._final_size = len(best)
        removed_edges = initial_solution - best
        added_edges = best - initial_solution
        self.logger.info(f"original: {len(initial_solution)}, final: {len(best)}, "
                         f"removed: {len(removed_edges)}, added: {len(added_edges)}")

        if self.learning:
            # end-of-run positives, each in its correct context
            if removed_edges:
                self.selector.record("remove", self.id_to_uv(initial_solution), self.id_to_uv(removed_edges), 1.0)
            if added_edges:
                self.selector.record("add", self.id_to_uv(best - added_edges), self.id_to_uv(added_edges), 1.0)
        return result

    def _keep_neg(self, neg_moves: list, neg_logqs: list, move_uv, lq) -> None:
        """Keep the first negatives, then reservoir-sample the rest (cap max_listwise_negs)."""
        if len(neg_moves) < self.max_listwise_negs:
            neg_moves.append(move_uv)
            neg_logqs.append(lq)
            return
        j = random.randrange(len(neg_moves) + 1)
        if j < self.max_listwise_negs:
            neg_moves[j] = move_uv
            neg_logqs[j] = lq

    # ---------------------------------------------------------------- evaluate

    def evaluate(self, solution: set[int]) -> tuple[bool, GraphInstance | None, float]:
        """Returns (flipped, instance, y_soft). y_soft is 1.0 on a flip and the probability mass on
        the counterfactual class otherwise (0/1 when the oracle exposes no probabilities)."""
        new_data = np.copy(self.G.data)
        self.disturb(new_data, self.G.directed, solution)
        y_best = 0.0
        if self.attributed:
            if self.method_success is None:
                self.method_success = [0] * len(self.methods)
            order = range(len(self.methods))
            if self.adaptive_method_order:
                order = sorted(order, key=lambda i: -self.method_success[i])
            for mi in order:
                method = self.methods[mi]
                self.k += 1
                self.k_local += 1
                self.phase_calls[self.phase] += 1
                self.phase_calls_iter[self.phase] += 1
                node_features = method(new_data, self.G.node_features)
                new_g = GraphInstance(id=self.G.id, label=0, data=new_data,
                                      directed=self.G.directed, node_features=node_features)
                flipped, p_cf = self.M.classify(new_g)
                y_best = max(y_best, p_cf)
                if flipped:
                    self.method_success[mi] += 1
                    return True, new_g, 1.0
        else:
            self.k += 1
            self.k_local += 1
            self.phase_calls[self.phase] += 1
            self.phase_calls_iter[self.phase] += 1
            new_g = GraphInstance(id=self.G.id, label=0, data=new_data,
                                  directed=self.G.directed, node_features=self.G.node_features)
            if self.recompute_features:
                self.dataset.manipulate(new_g)
            flipped, p_cf = self.M.classify(new_g)
            y_best = max(y_best, p_cf)
            if flipped:
                return True, new_g, 1.0
        return False, None, y_best

    def disturb(self, data, directed, solution: set[int]):
        for i in solution:
            (n1, n2) = self.labels[i]
            data[n1, n2] = (data[n1, n2] + 1) % 2
            if not directed:
                data[n2, n1] = (data[n2, n1] + 1) % 2

    # ----------------------------------------------------------- neighbourhoods

    def _trial_schedule(self, mode: str, levels: list[int]) -> list[int]:
        """Trials per level. Budget-neutral reallocation towards the level closest to the EMA of
        successful move sizes when ``adaptive_trials`` is on (Sonnerat et al. 2021 style)."""
        base = self.neigh_factor * 3
        if not self.adaptive_trials or len(levels) < 3:
            if mode == "remove" and self.block_removal:
                out = [2 * base if lv <= 2 else base for lv in levels]
                if self.block_policy == "bandit" and self.bandit_block_trials > 0:
                    out = [self.bandit_block_trials if lv > 1 else t for lv, t in zip(levels, out)]
                return out
            return [base] * len(levels)
        target = self.size_ema[mode]
        j_star = min(range(len(levels)), key=lambda j: abs(levels[j] - target))
        far = max(1, len(levels) // 2)
        sched = []
        for j in range(len(levels)):
            if j == j_star:
                sched.append(2 * base)
            elif abs(j - j_star) >= far:
                sched.append(max(1, base // 2))
            else:
                sched.append(base)
            if mode == "remove" and self.block_removal:
                # block rungs keep the full trial count; single/pair rungs keep the v2 count
                sched[-1] = max(sched[-1], 2 * base if levels[j] <= 2 else base)
                if self.block_policy == "bandit" and self.bandit_block_trials > 0 and levels[j] > 1:
                    sched[-1] = self.bandit_block_trials
        return sched

    def _bandit_ladder(self, ceiling: int) -> list[int]:
        """Rung sizes ordered by Thompson-sampled value size * P(success); rung 1 always present."""
        sizes = [1]
        k = 2
        while k <= min(64, ceiling - 1):
            sizes.append(k)
            k *= 2
        scored = []
        for sz in sizes:
            if sz == 1:
                continue
            su, tr = self.rung_stats.get(sz, [0, 0])
            # prior Beta(1, sz): larger blocks must earn their trials; never permanently excluded
            p_hat = random.betavariate(su + 1, tr - su + sz)
            scored.append((sz * p_hat, sz))
        scored.sort(reverse=True)
        blocks = [sz for v, sz in scored[: max(0, self.bandit_max_rungs - 1)] if v > self.bandit_min_value]
        return sorted(blocks, reverse=True) + [1]

    def _unlock_ladder(self, ceiling: int) -> list[int]:
        """Descending rungs [largest eligible, ..., 1]. Size 2k is eligible only when size k has been
        tried at least ``unlock_min_trials`` times with success rate >= ``unlock_rate``; an eligible size
        whose own rate dropped below unlock_rate/2 is skipped. Rung 1 is always tried last."""
        rungs = [1]
        k = 1
        while 2 * k <= min(64, ceiling - 1):
            su, tr = self.rung_stats.get(k, [0, 0])
            if tr < self.unlock_min_trials or su / float(tr) < self.unlock_rate:
                break
            k *= 2
            su2, tr2 = self.rung_stats.get(k, [0, 0])
            if tr2 >= self.unlock_min_trials and su2 / float(tr2) < self.unlock_rate / 2.0:
                continue
            rungs.append(k)
        return sorted(rungs, reverse=True)

    def _block_ladder(self, ceiling: int, step: int) -> list[int]:
        """Level order for ranked block removal: start at the adaptive block size, halve down to 1
        (the v2 behaviour is the tail of this ladder), then the usual ascending levels above it."""
        k = max(1, min(int(self.block_size), ceiling - 1))
        ladder = []
        while k >= 1:
            ladder.append(k)
            if k == 1:
                break
            k //= 2
        # Descending rungs only. The block size doubles after each success, so larger blocks are
        # reached through successes, not by sweeping every level: a sweep where even single removals
        # fail must stay cheap (in attributed mode each trial costs several oracle calls) so that the
        # swap and add phases still get their share of the per-iteration budget.
        return ladder

    @staticmethod
    def _window(order, k: int, t: int):
        m = len(order)
        if k >= m:
            return list(order)
        start = (t * k) % m
        end = start + k
        if end <= m:
            return order[start:end]
        return order[start:] + order[: end - m]

    def reduce_random(self, solution: set[int], i: int) -> set[int]:
        if len(solution) < i:
            raise ValueError("The set does not have enough elements.")
        removed = self.selector.propose_removals(self.id_to_uv(solution), i)
        return solution.difference(self.uv_to_id(removed))

    def edge_remove(self, solution: set[int]):
        uv_solution = self.id_to_uv(solution)
        ceiling = len(uv_solution)
        if ceiling == 0:
            return
        order, acq, logq = self.selector.get_trial_order("remove", uv_solution, seed=random.randrange(1 << 31))
        m = len(order)
        if m == 0:
            return
        step = int((ceiling / self.max_neigh) + 1)
        levels = list(range(1, ceiling, step))
        if self.block_removal:
            if self.block_policy == "bandit":
                levels = self._bandit_ladder(ceiling)
            elif self.block_policy == "unlock":
                levels = self._unlock_ladder(ceiling)
            else:
                levels = self._block_ladder(ceiling, step)
        for i, trials in zip(levels, self._trial_schedule("remove", levels)):
            k = min(i, m)
            if self.fill_remove_share and i == 1 and self.phase_budget:
                trials = max(trials, m)      # bounded by the phase share check in the caller
            for t in range(trials):
                if self.block_random_mix and (t % 2 == 1) and k < m:
                    idx = sorted(random.sample(range(m), k))
                else:
                    idx = self._window(list(range(m)), k, t // 2 if self.block_random_mix else t)
                removed = [order[j] for j in idx]
                lq = float(np.mean(logq[idx])) if len(idx) else 0.0
                pred = float(np.mean(acq[idx])) if len(idx) else 0.0
                rank = int((acq > pred).sum())
                removed_set = self.uv_to_id(removed)
                yield [solution.difference(removed_set), removed_set, [], solution, lq, pred, rank, m]

    def edge_add(self, solution: set[int], best):
        uv_solution = self.id_to_uv(solution)
        ceiling = (len(best) - len(solution)) + 1
        step = int(ceiling / self.max_neigh) + 1
        order, acq, logq = self.selector.get_trial_order("add", uv_solution, seed=random.randrange(1 << 31))
        m = len(order)
        if m == 0:
            return
        levels = list(range(1, ceiling, step))
        for i, trials in zip(levels, self._trial_schedule("add", levels)):
            k = min(i, m)
            for t in range(trials):
                idx = self._window(list(range(m)), k, t)
                added = [order[j] for j in idx]
                lq = float(np.mean(logq[idx])) if len(idx) else 0.0
                pred = float(np.mean(acq[idx])) if len(idx) else 0.0
                rank = int((acq > pred).sum())
                added_set = self.uv_to_id(added)
                yield [solution.union(added_set), [], added_set, solution, lq, pred, rank, m]

    def edge_swap(self, solution: set[int]):
        ceiling = min(len(solution), (self.EPlus - len(solution))) + 1
        step = int(ceiling / self.max_neigh) + 1
        for i in range(1, ceiling, step):
            for _ in range(self.neigh_factor * 3):
                removed, pred_r = self.selector.propose_scored("remove", self.id_to_uv(solution), i)
                removed_set = self.uv_to_id(removed)
                temp_solution = solution.difference(removed_set)
                added, pred_a = self.selector.propose_scored("add", self.id_to_uv(temp_solution), i)
                added_set = self.uv_to_id(added)
                yield [temp_solution.union(added_set), removed_set, added_set, solution, temp_solution, pred_r, pred_a]

    # ------------------------------------------------------------------ helpers

    def id_to_uv(self, ids) -> list[tuple[int, int]]:
        return [self.labels[i] for i in ids]

    def uv_to_id(self, uv) -> set[int]:
        return {self.label_to_id[(i, j) if i < j else (j, i)] for (i, j) in uv if i != j}

    def write(self):
        pass

    def read(self):
        pass
