"""Data-driven Forward Search (DFS) — Abrate & Bonchi, KDD'21, Algorithm 1 with
the uniform pick() replaced by a class-distribution-weighted pick().

This is the generator half of DDBS. Pair it with the DBS minimizer for the full
data-driven bidirectional search of the paper.
"""

import copy
import numpy as np

from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import init_dflts_to_of
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.utils.data_driven_priors import candidate_edge_indices, prior_weights, weighted_pick


class DFS(Explainer):
    """Data-driven Forward Search.

    Replaces the uniform random pick() of OFS with a weighted pick() that favours:
      - removing edges strongly characteristic of the original class
      - adding edges strongly characteristic of the opposite class
    Weights are computed from the dataset's class-conditional edge frequencies: by node index (``prior``
    "index", the default, one graph size only) or by pair of node types (``prior`` "type_pair", protocol v2;
    see the DBS minimizer and src/explainer/future/utils/data_driven_priors.py).
    """

    @property
    def requires_fixed_size(self):   # the index prior is per node index, from the graphs of one size
        return self.local_config['parameters'].get('prior', 'index') == 'index'

    def check_configuration(self):
        super().check_configuration()
        params = self.local_config['parameters']
        params['max_oc'] = params.get('max_oc', 2000)
        params['changes_batch_size'] = params.get('changes_batch_size', 5)
        params['epsilon'] = params.get('epsilon', 1e-6)
        params['random_seed'] = params.get('random_seed', None)

        init_dflts_to_of(
            self.local_config,
            'distance_metric',
            'src.explainer.heuristic.obs_dist.ObliviousBidirectionalDistance',
        )

    def init(self):
        super().init()
        self.logger = self.context.logger
        params = self.local_config['parameters']
        self.max_oc = params['max_oc']
        self.changes_batch_size = params['changes_batch_size']
        self.epsilon = params['epsilon']
        self._rng = np.random.default_rng(params['random_seed'])

        self.distance_metric = get_instance_kvargs(
            params['distance_metric']['class'],
            params['distance_metric']['parameters'],
        )

    # ------------------------------------------------------------------

    def explain(self, instance):
        cf_inst = self._data_driven_forward_search(instance)
        return LocalGraphCounterfactualExplanation(
            context=self.context,
            dataset=self.dataset,
            oracle=self.oracle,
            explainer=self,
            input_instance=instance,
            counterfactual_instances=[cf_inst],
        )

    def _data_driven_forward_search(self, instance):
        f_E = int(self.oracle.predict(instance))
        n = int(instance.data.shape[0])

        params = self.local_config['parameters']
        weights = prior_weights(self.dataset, instance, f_E, params, params.get('fold_id', -1))

        directed = bool(getattr(instance, 'is_directed', False))
        all_positions = list(candidate_edge_indices(n, directed))

        g_c = np.copy(instance.data)
        # Tracked "already touched" positions, mirroring the L set in Algorithm 1.
        touched = np.zeros((n, n), dtype=bool)

        oracle_calls = 0
        while oracle_calls < self.max_oc:
            # Build the eligible pool: any position not in L (touched set).
            pool = [(i, j) for (i, j) in all_positions if not touched[i, j]]
            if not pool:
                break
            picks = weighted_pick(
                self._rng,
                pool,
                weights,
                k=self.changes_batch_size,
                epsilon=self.epsilon,
            )
            for (i, j) in picks:
                g_c[i, j] = 1 - g_c[i, j]
                if not directed:
                    g_c[j, i] = g_c[i, j]
                touched[i, j] = True
                if not directed:
                    touched[j, i] = True

            cand = GraphInstance(
                id=instance.id,
                label=f_E,
                data=g_c,
                node_features=instance.node_features,
            )
            cand.label = int(self.oracle.predict(cand))
            oracle_calls += 1
            if cand.label != f_E:
                return cand

        # Fail: return a copy of the original (caller checks label).
        return copy.deepcopy(instance)
