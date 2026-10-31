import copy

import numpy as np
import torch

from src.core.oracle_base import Oracle


class MarginOracle(Oracle):
    """A trained oracle that only changes its answer with confidence: a graph gets a class other than the
    reference (the class the base oracle gives to the instance being explained) only when the base oracle
    gives that class a probability of at least ``margin`` (0.5 is the base oracle itself when there are
    two classes).

    Parameters: ``margin`` and ``base_oracle``, the base oracle's config exactly as it appears in the
    experiments (including the propagated ``fold_id`` and ``retrain``), so that it is loaded from its
    existing cache instead of being trained again. The evaluator calls ``set_reference`` with every
    instance before explaining it. ``predict`` and ``predict_proba`` agree: the probability of the other
    class is shifted down by ``margin - 0.5``, so the arg max flips exactly at the margin and the
    minimizers' shaped reward (the probability of the other class) keeps measuring the progress towards
    it. Only one call is counted per query, as for any oracle; the reference costs no call.
    """

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        if 'base_oracle' not in p:
            raise ValueError("MarginOracle needs the base_oracle config")
        p['margin'] = float(p.get('margin', 0.5))
        if not 0.5 <= p['margin'] < 1.0:
            raise ValueError("margin must be in [0.5, 1)")

    def init(self):
        pass

    def load_or_create(self, condition=False):
        # Nothing of its own to train or store: build the base oracle, which loads its own cache.
        self.base = self.context.factories['oracles'].get_oracle(
            copy.deepcopy(self.local_config['parameters']['base_oracle']), self.dataset)
        self.margin = self.local_config['parameters']['margin']
        self.reference = None

    def real_fit(self):
        pass

    def set_reference(self, instance):
        self.reference = int(np.argmax(self._base_probs(instance)))

    def _base_probs(self, instance) -> np.ndarray:
        out = self.base._real_predict_proba(instance)
        arr = out.detach().cpu().double().numpy() if isinstance(out, torch.Tensor) else np.asarray(out, dtype=np.float64)
        arr = arr.reshape(-1)
        if (arr < 0).any() or abs(arr.sum() - 1.0) > 1e-6:   # logits (the GCN oracles have no softmax)
            e = np.exp(arr - arr.max())
            arr = e / e.sum()
        return arr

    def _margin_probs(self, instance) -> np.ndarray:
        """All the mass on the reference class and the most probable other class: the latter gets its base
        probability minus (margin - 0.5), so it wins the arg max exactly when its base probability passes
        the margin (with two classes this is just the shifted distribution)."""
        p = self._base_probs(instance)
        if self.reference is None:
            return p
        others = [c for c in range(p.size) if c != self.reference]
        best = max(others, key=lambda c: p[c])
        q = np.zeros_like(p)
        q[best] = min(1.0, max(0.0, p[best] - (self.margin - 0.5)))
        q[self.reference] = 1.0 - q[best]
        return q

    def _real_predict(self, data_instance):
        # the same arg max the minimizers take from predict_proba, so both agree on every graph
        return int(np.argmax(self._margin_probs(data_instance)))

    def _real_predict_proba(self, data_instance):
        return torch.from_numpy(self._margin_probs(data_instance))
