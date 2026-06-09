"""Deterministic seeding helper (Note C, REVISION_EXPERIMENTS.md).

Seeds Python's ``random``, NumPy's global RNG and (when available)
PyTorch with a single value so multi-seed runs are reproducible. Callers
typically expose ``seed`` as a config parameter and invoke :func:`set_seed`
from the explainer/minimizer ``init()``.
"""
from __future__ import annotations

import random
import numpy as np


def set_seed(seed):
    """Seed ``random``, ``numpy`` and (if importable) ``torch`` with ``seed``.

    Passing ``None`` is a no-op so legacy configs that omit the parameter
    keep their non-deterministic behavior and (more importantly) their
    hash, avoiding spurious cache misses on already-trained artifacts.
    """
    if seed is None:
        return
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(s)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
