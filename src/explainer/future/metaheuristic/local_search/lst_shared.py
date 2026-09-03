"""Shared dataset-wide training artifacts for trainable LocalSearch variants.

Background
----------
Three minimizer variants (``local_search_trainable``,
``local_search_trainable_ponderation``, ``local_search_selection_net_trainable``)
all train an identical scored list of node-attribute manipulation method names
during ``fit()``. They previously did this independently, on the training fold,
yielding 30 redundant pickles for the same dataset.

This module centralises the work:

* :class:`LSTMethodsArtifact` is a :class:`Trainable` keyed only by the
  dataset hash (its ``fold_id`` is pinned to -1). It produces
  ``self.model = {"methods": [(score, name), ...]}`` where ``score`` is a
  signed integer reflecting how often each manipulation method changed the
  oracle prediction across a (proportion-sampled) sweep over the dataset.
  Top-half of the methods is kept (matching the original variants' filter).
* The variants now look this artifact up at ``init()`` time and reuse the
  scores instead of recomputing.

Medoids are *not* stored here — they live in :mod:`src.explainer.future.search.dcm`
and are loaded the same way (DCM is also dataset-wide). Both artifacts have
matching standalone trainers under :mod:`scripts/`.
"""

from __future__ import annotations

import random
import numpy as np

from src.core.trainable_base import Trainable
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.manipulation.methods import (
    average_smoothing,
    average_smoothing_zero,
    feature_aggregation,
    heat_kernel_diffusion,
    identity,
    laplacian_regularization,
    random_walk_diffusion,
    weighted_smoothing,
)
from src.utils.cfg_utils import retake_dataset, retake_oracle


# The 8 candidate manipulation functions. Keep names in sync with what the LST
# variants' ``convert(method_name)`` dispatchers expect.
METHOD_NAMES: list[str] = [
    "average_smoothing",
    "average_smoothing_zero",
    "weighted_smoothing",
    "laplacian_regularization",
    "feature_aggregation",
    "heat_kernel_diffusion",
    "random_walk_diffusion",
    "identity",
]


def _resolve_method(name: str):
    """Map a method name back to its callable. Mirrors the LST variants' own
    convert() dispatchers so the scoring is faithful to runtime use."""
    if name == "average_smoothing":
        return lambda data, features: average_smoothing(data, features, iterations=1)
    if name == "average_smoothing_zero":
        return lambda data, features: average_smoothing_zero(data, features, iterations=1)
    if name == "weighted_smoothing":
        return lambda data, features: weighted_smoothing(data, features, iterations=1)
    if name == "laplacian_regularization":
        return lambda data, features: laplacian_regularization(data, features, lambda_reg=0.01, iterations=1)
    if name == "feature_aggregation":
        return lambda data, features: feature_aggregation(data, features, alpha=0.5, iterations=1)
    if name == "heat_kernel_diffusion":
        return lambda data, features: heat_kernel_diffusion(data, features, t=0.5)
    if name == "random_walk_diffusion":
        return lambda data, features: random_walk_diffusion(data, features, steps=1)
    if name == "identity":
        return lambda data, features: identity(data, features)
    raise ValueError(f"unknown manipulation method: {name!r}")


class LSTMethodsArtifact(Trainable):
    """Dataset-wide scored manipulation methods cache.

    The hash key is fold-independent (``fold_id`` is pinned to -1), so every
    LST variant that loads this artifact for the same dataset gets the same
    pickle — one file on disk, shared by all the variants.
    """

    # ------------------------------------------------------------------

    def __init__(self, context, local_config):
        # Trainable expects dataset and oracle to be retrievable from
        # local_config (see retake_dataset / retake_oracle).
        self.dataset = retake_dataset(local_config)
        self.oracle = retake_oracle(local_config)
        super().__init__(context, local_config)

    def check_configuration(self):
        super().check_configuration()
        params = self.local_config["parameters"]
        # Make the on-disk hash fold-independent.
        params["fold_id"] = -1
        # Fraction of dataset to score over (default = full dataset).
        params["proportion"] = float(params.get("proportion", 1.0))
        # Half-or-better filter to mirror the original variants.
        params["keep_top_half"] = bool(params.get("keep_top_half", True))
        params["random_seed"] = int(params.get("random_seed", 0))

    def init(self):
        super().init()
        self.logger = self.context.logger
        params = self.local_config["parameters"]
        self.proportion = float(params["proportion"])
        self.keep_top_half = bool(params["keep_top_half"])
        self._rng = random.Random(int(params["random_seed"]))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def real_fit(self):
        """Score each manipulation method by oracle-flip frequency.

        For each sampled instance, apply each method to the node features
        (leaving the adjacency intact). If the manipulated graph receives a
        different oracle prediction than the original, the method gets +1;
        otherwise -1. The scoring is variant-independent, so the resulting
        list is shareable across all LST trainable variants.
        """
        instances = list(self.dataset.instances)
        if self.proportion < 1.0:
            n = max(1, int(round(len(instances) * self.proportion)))
            instances = self._rng.sample(instances, k=n)
        self.logger.info(
            f"LSTMethodsArtifact: scoring {len(METHOD_NAMES)} methods "
            f"over {len(instances)} sampled instances"
        )

        scores: dict[str, int] = {name: 0 for name in METHOD_NAMES}
        callables = {name: _resolve_method(name) for name in METHOD_NAMES}

        for i, inst in enumerate(instances, start=1):
            base_label = int(self.oracle.predict(inst))
            for name in METHOD_NAMES:
                try:
                    new_features = callables[name](inst.data, inst.node_features)
                except Exception as e:
                    # A method that throws on this instance is treated as a
                    # negative observation rather than crashing the sweep.
                    scores[name] -= 1
                    continue
                cand = GraphInstance(
                    id=inst.id,
                    label=0,
                    data=inst.data,
                    directed=getattr(inst, "directed", False),
                    node_features=new_features,
                )
                if int(self.oracle.predict(cand)) != base_label:
                    scores[name] += 1
                else:
                    scores[name] -= 1
            if i % max(1, len(instances) // 10) == 0:
                self.logger.info(
                    f"LSTMethodsArtifact: {i}/{len(instances)} scored "
                    f"(current scores: {scores})"
                )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        # Original variants kept methods with score ≥ (top + bottom)/2; we mirror
        # that here so the shared output matches what they used to produce.
        if self.keep_top_half and len(ranked) > 1:
            top = ranked[0][1]
            bottom = ranked[-1][1]
            mid = (top + bottom) // 2
            ranked = [(s, n) for n, s in ranked if s >= mid]
        else:
            ranked = [(s, n) for n, s in ranked]

        self.model = {"methods": ranked}
        self.logger.info(f"LSTMethodsArtifact: final methods = {ranked}")
        super().real_fit()


# ---------------------------------------------------------------------------
# Convenience loader used by the variants
# ---------------------------------------------------------------------------


def load_methods(
    context,
    dataset,
    oracle,
    proportion: float = 1.0,
    retrain: bool = False,
) -> list[tuple[int, str]]:
    """Return ``[(score, method_name), ...]`` for the given dataset/oracle.

    Loads the cached artifact if present (any (dataset, oracle) hash match);
    otherwise triggers training. Caller-provided ``proportion`` only takes
    effect when the artifact is being trained for the first time.
    """
    cfg = {
        "class": "src.explainer.future.metaheuristic.local_search.lst_shared.LSTMethodsArtifact",
        "dataset": dataset,
        "oracle": oracle,
        "parameters": {
            "fold_id": -1,
            "proportion": float(proportion),
            "retrain": bool(retrain),
        },
    }
    artifact = LSTMethodsArtifact(context=context, local_config=cfg)
    return artifact.model["methods"]
