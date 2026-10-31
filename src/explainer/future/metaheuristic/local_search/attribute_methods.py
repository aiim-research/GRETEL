"""Node-attribute strategies of BLS, by name, so a config can state which ones a run evaluates.

Every structural candidate is evaluated under a sequence of attribute strategies, first-accept
(``EvalWithAttrVariants`` in the paper), so the list and its order set the oracle calls a failed
candidate costs and which counterfactuals are reachable. The thesis implementation hard-coded six, the
ordering variants and the selector skeleton seven; ``attribute_methods`` in the minimizer parameters
chooses explicitly, and the defaults keep every existing config's behaviour (and hash) unchanged.
"""
from __future__ import annotations

from collections import OrderedDict

from src.explainer.future.metaheuristic.manipulation.methods import (
    average_smoothing, average_smoothing_zero, feature_aggregation, heat_kernel_diffusion, identity,
    laplacian_regularization, random_walk_diffusion, weighted_smoothing,
)

METHODS = OrderedDict([
    ("identity", lambda data, features: identity(data, features)),
    ("average_smoothing", lambda data, features: average_smoothing(data, features, iterations=1)),
    ("average_smoothing_zero", lambda data, features: average_smoothing_zero(data, features, iterations=1)),
    ("weighted_smoothing", lambda data, features: weighted_smoothing(data, features, iterations=1)),
    ("laplacian_regularization", lambda data, features: laplacian_regularization(data, features, lambda_reg=0.01, iterations=1)),
    ("feature_aggregation", lambda data, features: feature_aggregation(data, features, alpha=0.5, iterations=1)),
    ("heat_kernel_diffusion", lambda data, features: heat_kernel_diffusion(data, features, t=0.5)),
    ("random_walk_diffusion", lambda data, features: random_walk_diffusion(data, features, steps=1)),
])

# the six of the thesis implementation (local_search.py), in its order
THESIS_METHODS = ["identity", "average_smoothing", "weighted_smoothing", "laplacian_regularization",
                  "feature_aggregation", "random_walk_diffusion"]
# the seven of the selector skeleton and of the ordering variants: the paper's list
SEVEN_METHODS = ["identity", "average_smoothing", "weighted_smoothing", "laplacian_regularization",
                 "feature_aggregation", "heat_kernel_diffusion", "random_walk_diffusion"]


def build_methods(names) -> list:
    unknown = [n for n in names if n not in METHODS]
    if unknown:
        raise ValueError(f"unknown attribute_methods {unknown}; known: {list(METHODS)}")
    return [METHODS[n] for n in names]
