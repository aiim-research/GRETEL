"""Data-driven Backward Search (DBS) — Abrate & Bonchi, KDD'21, Algorithm 2 with
the uniform pick() over the symmetric difference replaced by a class-distribution-
weighted pick().

This is the minimizer half of DDBS. Pair it with DFS for the full data-driven
bidirectional search. Functionally analogous to the existing `Random` minimizer
(which implements the *oblivious* backward search of OBS), but the edge-flip
choices use the same w(e) = |D⁺(e)| − |D⁻(e)| family of weights as DFS.
"""

import copy
import numpy as np

from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.metrics.ged import GraphEditDistanceMetric
from src.utils.comparison import get_all_edge_differences
from src.explainer.future.utils.data_driven_priors import (
    compute_class_counts,
    edge_weight_matrix,
    weighted_pick,
)


class DBS(ExplanationMinimizer):
    """Data-driven Backward Search minimizer."""

    def check_configuration(self):
        super().check_configuration()
        params = self.local_config['parameters']
        params['max_oc'] = params.get('max_oc', 2000)
        params['changes_batch_size'] = params.get('changes_batch_size', 5)
        params['epsilon'] = params.get('epsilon', 1e-6)
        params['random_seed'] = params.get('random_seed', None)

    def init(self):
        super().init()
        params = self.local_config['parameters']
        self.max_oc = params['max_oc']
        self.changes_batch_size = params['changes_batch_size']
        # Opt-in (hash-stable): skip per-candidate dataset.manipulate() when
        # the oracle ignores recomputed node features (e.g. ASD, Tree-Cycles).
        self.recompute_features = params.get('recompute_features', True)
        self.epsilon = params['epsilon']
        self._rng = np.random.default_rng(params['random_seed'])
        self.distance_metric = GraphEditDistanceMetric()

    # ------------------------------------------------------------------

    def minimize(self, explanation: LocalGraphCounterfactualExplanation) -> DataInstance:
        instance = explanation.input_instance
        cf_instance = explanation.counterfactual_instances[0]
        f_E = int(self.oracle.predict(instance))

        # Guard: a minimizer must never alter the generator's correctness. If
        # the generator did not actually produce a counterfactual (its output
        # still has the original label), return it unchanged instead of running
        # the backward search — otherwise reverting subsets of its edits can
        # stumble onto a counterfactual the generator missed and spuriously
        # *raise* correctness, breaking the "correctness is a generator
        # property" invariant (matches the guard LBS/RHC already have).
        if int(self.oracle.predict(cf_instance)) == f_E:
            return cf_instance

        # Edges in the symmetric difference E Δ E_c — these are the only positions
        # we will touch during minimization (Algorithm 2, line 2).
        changed_edges, _, _ = get_all_edge_differences(instance, [cf_instance])
        if not changed_edges:
            return cf_instance

        D_plus, D_minus = compute_class_counts(
            self.dataset, f_E, exclude_instance_id=instance.id
        )
        # Weights are computed using the *original* instance E (so positions where
        # E has an edge get D⁺-D⁻ and positions where E has no edge get D⁻-D⁺).
        # See §4.2 of the paper: the weight formula for DBS is the same as DFS
        # restricted to e ∈ E Δ E_c.
        weights = edge_weight_matrix(instance.data, D_plus, D_minus)

        return self._data_driven_backward_search(
            instance, cf_instance, changed_edges, weights, f_E,
        )

    # ------------------------------------------------------------------

    def _data_driven_backward_search(
        self, instance, cf_instance, changed_edges, weights, f_E
    ):
        initial_changed = len(changed_edges)
        reduction_success = False
        gc = np.copy(cf_instance.data)
        k = max(1, self.changes_batch_size)
        oracle_calls = 0
        # Use a list and pop from the head (cheaper than re-shuffling each iter);
        # the order is randomized only via weighted sampling below.
        pool = list(changed_edges)

        while oracle_calls < self.max_oc and pool:
            ki = min(k, len(pool))
            picks = weighted_pick(
                self._rng, pool, weights, k=ki, epsilon=self.epsilon
            )
            if not picks:
                break
            # Remove the picks from the pool.
            picks_set = set(picks)
            pool = [p for p in pool if p not in picks_set]

            gci = np.copy(gc)
            for (i, j) in picks:
                gci[i][j] = 1 - gci[i][j]
                if not instance.is_directed:
                    gci[j][i] = 1 - gci[j][i]

            reduced_cf = GraphInstance(
                id=instance.id,
                label=0,
                data=gci,
                directed=instance.directed,
                node_features=instance.node_features,
                graph_features=instance.graph_features,
            )
            if self.recompute_features:
                self.dataset.manipulate(reduced_cf)
            reduced_cf.label = int(self.oracle.predict(reduced_cf))
            oracle_calls += 1

            if reduced_cf.label != f_E:
                # Still a counterfactual — commit and grow k.
                reduction_success = True
                gc = np.copy(gci)
                k += 1
                final_changed = len(pool)
            else:
                # Lost counterfactuality. Shrink k and put the picks back into
                # the pool to retry later (mirrors OBS's adaptive-k behaviour).
                if k > 1:
                    k -= 1
                    pool = pool + picks
                else:
                    pool = pool + picks  # k already at 1 — keep cycling

        result_cf = GraphInstance(
            id=instance.id,
            label=0,
            data=gc,
            directed=instance.directed,
            node_features=instance.node_features,
        )
        if self.recompute_features:
            self.dataset.manipulate(result_cf)
        result_cf.label = int(self.oracle.predict(result_cf))

        if reduction_success:
            self.logger.info(
                f"DBS: reduced counterfactual for {instance.id} ({initial_changed} -> {final_changed})"
            )
            return result_cf
        self.logger.info(
            f"DBS: counterfactual for {instance.id} not reduced ({initial_changed})"
        )
        return cf_instance

    def write(self):
        pass

    def read(self):
        pass
