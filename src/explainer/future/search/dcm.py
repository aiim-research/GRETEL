"""DCM (DCEM) — Distance-based Class Medoid generator.

Trains a "cross-medoid" per class over the dataset: the graph in class c whose
sum of GEDs to graphs of *other* classes is minimum (Abrate-style data-driven
prior, formalised in the thesis §5.1).

Storage is **dataset-wide**, not fold-wide: the trained file is keyed by the
dataset hash (the explainer's `fold_id` is pinned to -1 in `check_configuration`,
so the on-disk name is the same regardless of which fold any caller passed).
The model itself is `{class: dataset_instance_index}` — a tiny dict of small
ints — so the saved file is a few hundred bytes even for thousands of graphs.

Performance: each unordered cross-class pair (g, g') has its GED computed
exactly once and the result is added to *both* endpoints' running sums. That
fixes a pre-existing double-counting bug and cuts work in half versus the
previous nested-loop implementation.

Use the helper script ``scripts/compute_dcm.py`` to train a DCM standalone for
a given dataset + oracle + proportion, outside of any generate-minimize run.
"""

from __future__ import annotations

import numpy as np

from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.dataset.instances.graph import GraphInstance
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.metrics.ged import GraphEditDistanceMetric


class DCM(Explainer, Trainable):

    # ------------------------------------------------------------------
    # Configurable hooks
    # ------------------------------------------------------------------

    def check_configuration(self):
        super().check_configuration()
        params = self.local_config['parameters']
        # Pin fold_id so the saved file is dataset-wide, not fold-wide. We deliberately
        # ignore whatever fold the caller passed: the medoids are a property of the
        # full dataset, and we don't want N copies of the same file on disk.
        params['fold_id'] = -1
        # Fraction of each class to use as medoid candidates during training.
        # 1.0 → all instances (slowest, exact). Smaller values trade some quality
        # for a quadratically faster fit.
        params['proportion'] = float(params.get('proportion', 1.0))
        # How to assign each graph to a class while training the medoids.
        # 'label'  — use the dataset's ground-truth label (fast, no oracle calls)
        # 'oracle' — use the oracle's prediction (per the thesis; adds N oracle calls
        #            during training, which is offline anyway)
        params['classify_with'] = params.get('classify_with', 'label')
        # RNG seed for reproducible candidate subsampling when proportion < 1.0
        params['random_seed'] = params.get('random_seed', 0)

    def init(self):
        self.device = "cpu"
        self.distance_metric = GraphEditDistanceMetric()
        self.logger = self.context.logger
        params = self.local_config['parameters']
        self.fold_id = -1
        self.proportion = float(params['proportion'])
        self.classify_with = params['classify_with']
        self._rng = np.random.default_rng(int(params['random_seed']))
        super().init()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def real_fit(self):
        """Compute one cross-medoid per class over the full dataset.

        Algorithm:
          1. Classify every instance (by oracle prediction or by stored label).
          2. For each (class_a, class_b) pair with a < b, iterate all (g_a, g_b)
             combinations; compute GED once; accumulate into both endpoints'
             running cross-class sums.
          3. For each class, the medoid is the candidate with the minimum sum.

        We store dataset indices (small ints) rather than instance objects, so
        the pickled file is tiny.
        """
        if self.proportion <= 0:
            self.logger.warning("DCM proportion=0; no candidates — model is empty.")
            self.model = {}
            super().real_fit()
            return

        classify = self._make_classifier()

        # Classify every instance and group candidate indices by class.
        idx_by_class: dict[int, list[int]] = {}
        instances = self.dataset.instances
        for i, inst in enumerate(instances):
            cls = int(classify(inst))
            idx_by_class.setdefault(cls, []).append(i)

        # Sub-sample candidates per class (deterministic).
        if self.proportion < 1.0:
            for c, idxs in idx_by_class.items():
                k = max(1, int(round(len(idxs) * self.proportion)))
                chosen = self._rng.choice(idxs, size=k, replace=False)
                idx_by_class[c] = sorted(int(x) for x in chosen)

        classes = sorted(idx_by_class.keys())
        total_candidates = sum(len(v) for v in idx_by_class.values())
        n_pairs = sum(
            len(idx_by_class[classes[a]]) * len(idx_by_class[classes[b]])
            for a in range(len(classes)) for b in range(a + 1, len(classes))
        )
        self.logger.info(
            f"DCM: training over {total_candidates} candidates across "
            f"{len(classes)} classes ({n_pairs} cross-class GED evaluations)"
        )

        # Accumulate cross-class distance sums per candidate index.
        cross_sum: dict[int, float] = {i: 0.0 for v in idx_by_class.values() for i in v}

        # Each unordered class pair, then each graph pair within — compute GED once.
        pairs_done = 0
        for a in range(len(classes)):
            for b in range(a + 1, len(classes)):
                ca, cb = classes[a], classes[b]
                idxs_a = idx_by_class[ca]
                idxs_b = idx_by_class[cb]
                for i in idxs_a:
                    g1 = instances[i]
                    for j in idxs_b:
                        g2 = instances[j]
                        d = self.distance_metric.evaluate(g1, g2)
                        cross_sum[i] += d
                        cross_sum[j] += d
                        pairs_done += 1
                self.logger.info(
                    f"DCM: finished pair (class {ca}, class {cb}); "
                    f"{pairs_done}/{n_pairs} GEDs done"
                )

        # Pick the medoid for each class (argmin of cross-class sums).
        medoids: dict[int, int] = {}
        for c, idxs in idx_by_class.items():
            if not idxs:
                continue
            medoids[c] = min(idxs, key=lambda i: cross_sum[i])

        self.model = medoids
        self.logger.info(f"DCM: medoid indices = {self.model}")
        super().real_fit()

    def _make_classifier(self):
        """Return a function that maps an instance to its class id."""
        if self.classify_with == 'oracle':
            return lambda inst: self.oracle.predict(inst)
        return lambda inst: getattr(inst, 'label', 0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def explain(self, instance):
        category = int(self.oracle.predict(instance))
        instances = self.dataset.instances

        best_d = float('inf')
        best_medoid = None
        for other_cls, medoid_idx in self.model.items():
            if other_cls == category:
                continue
            medoid = instances[medoid_idx]
            d = self.distance_metric.evaluate(instance, medoid)
            if d < best_d:
                best_d = d
                best_medoid = medoid

        if best_medoid is None:
            # No medoid for any other class — return a copy of the input so the
            # pipeline does not crash. Callers can detect this via correctness=False.
            best_medoid = instance

        cf_instance = GraphInstance(
            id=best_medoid.id,
            label=best_medoid.label,
            data=best_medoid.data,
            node_features=best_medoid.node_features,
        )
        return LocalGraphCounterfactualExplanation(
            context=self.context,
            dataset=self.dataset,
            oracle=self.oracle,
            explainer=self,
            input_instance=instance,
            counterfactual_instances=[cf_instance],
        )
