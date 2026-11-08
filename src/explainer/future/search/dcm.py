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
        # Protocol v2 (opt-in, read without being written into the config): return the nearest medoid of
        # another class whose projection onto the input's nodes is a counterfactual
        # (src/explainer/future/utils/projection.py). Without it, DCM behaves as always.
        self.projection = params.get('projection')

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
                        d = self._distance(g1, g2)
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

        if self.projection:
            # Protocol v2: also the medoid of each class among its graphs of at most n nodes, for every n,
            # as the running minimum of the same cross-class sums over the graphs sorted by size. An input
            # of n nodes can reach exactly those (the isolated-node policy deletes what it does not need).
            by_size = {}
            for c, idxs in idx_by_class.items():
                table, best = [], None
                for i in sorted(idxs, key=lambda i: (len(instances[i].data), i)):
                    if best is None or cross_sum[i] < cross_sum[best]:
                        best = i
                    size = len(instances[i].data)
                    if table and table[-1][0] == size:
                        table[-1] = (size, best)
                    else:
                        table.append((size, best))
                by_size[c] = table
            self.model = {"medoids": medoids, "by_size": by_size}
        else:
            self.model = medoids
        self.logger.info(f"DCM: medoid indices = {medoids}")
        super().real_fit()

    def _distance(self, g1, g2):
        """v1: the cell-wise GED. v2 (projection): the structural edit cost under the correspondence."""
        if not self.projection:
            return self.distance_metric.evaluate(g1, g2)
        from src.explainer.future.utils.projection import correspondence, structural_distance
        return structural_distance(g1, g2, correspondence(g1, g2, self.projection))

    def medoid_within(self, cls, n):
        """Index of the medoid of class ``cls`` among its graphs of at most ``n`` nodes, or None."""
        best = None
        for size, idx in self.model["by_size"].get(cls, []):
            if size > n:
                break
            best = idx
        return best

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

        if self.projection:
            return self.explain_projected(instance, category)

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

    def explain_projected(self, instance, input_label):
        """Protocol v2 generator contract, as DCE's (src/explainer/search/dces.py) over the medoids and decided
        by the generator alone: for each other class, its medoid among the graphs no larger than the input;
        the nearest whose full projection the oracle classifies differently from the input, else the nearest
        of them itself (seed_kind "fallback"; with classify_with "oracle" it is a counterfactual by
        construction), else the global medoid of another class."""
        import copy
        from src.explainer.future.utils.projection import project, ranked_candidates
        from src.utils.isolated import materialize, policy_of
        instances = self.dataset.instances
        n = len(instance.data)
        others = [c for c in self.model["medoids"] if c != int(input_label)]
        within = [self.medoid_within(c, n) for c in others]
        candidates = [instances[i] for i in within if i is not None]
        if not candidates:
            candidates = [instances[self.model["medoids"][c]] for c in others]
        ranked = ranked_candidates(instance, candidates, self.projection)
        seed, info = None, {}
        for _, _, candidate, mapping in ranked:
            projected = project(instance, candidate, mapping, features=True)
            if self.oracle.predict(projected) != input_label:
                seed = projected
                info = {"seed_kind": "projection", "candidate_id": candidate.id,
                        "correspondences": [materialize(projected, policy_of(self.dataset))[1]]}
                break
        if seed is None and getattr(self, "fallback_scan", False):
            # Opt-in (GenerateMinimize's fallback_scan): no medoid projects to a counterfactual, so go on with
            # DCE's candidates, every dataset graph the oracle classifies differently from the input (the
            # medoids already tried left out), nearest first; the fallback is then the nearest of all.
            tried = {id(c) for c in candidates}
            pool = [c for c in instances if id(c) not in tried and self.oracle.predict(c) != input_label]
            scanned = ranked_candidates(instance, pool, self.projection)
            for _, _, candidate, mapping in scanned:
                projected = project(instance, candidate, mapping, features=True)
                if self.oracle.predict(projected) != input_label:
                    seed = projected
                    info = {"seed_kind": "projection", "candidate_id": candidate.id, "scan": True,
                            "correspondences": [materialize(projected, policy_of(self.dataset))[1]]}
                    break
            if seed is None and scanned and (not ranked or scanned[0][0] < ranked[0][0]):
                ranked = scanned[:1]
        if seed is None and ranked:
            _, _, candidate, mapping = ranked[0]
            seed = copy.deepcopy(candidate)
            info = {"seed_kind": "fallback", "candidate_id": candidate.id, "correspondences": [mapping]}
        if seed is None:
            seed = copy.deepcopy(instance)
            info = {"seed_kind": "none"}
        seed.id = instance.id
        explanation = LocalGraphCounterfactualExplanation(context=self.context, dataset=self.dataset,
                                                          oracle=self.oracle, explainer=self,
                                                          input_instance=instance, counterfactual_instances=[seed])
        explanation._info.update(info)
        return explanation
