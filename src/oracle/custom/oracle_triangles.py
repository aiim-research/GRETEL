import numpy as np

from src.core.oracle_base import Oracle


class TriangleCountOracle(Oracle):
    """Exact rule: class 1 when the graph has at least ``threshold`` triangles. ``predict_proba`` is a
    logistic function of the distance to the threshold (``scale`` triangles, default 4), whose arg max is
    the rule, so the minimizers' shaped reward measures how close a candidate is to flipping."""

    def init(self):
        super().init()
        self.model = ""

    def check_configuration(self):
        super().check_configuration()
        if 'threshold' not in self.local_config['parameters']:
            raise ValueError("TriangleCountOracle needs a threshold")

    def real_fit(self):
        pass

    @staticmethod
    def count_triangles(adj) -> int:
        a = (np.asarray(adj) != 0).astype(np.float64)
        np.fill_diagonal(a, 0)
        a = np.maximum(a, a.T)
        return int(round(np.trace(a @ a @ a) / 6))

    def _real_predict(self, data_instance):
        return int(self.count_triangles(data_instance.data) >= self.local_config['parameters']['threshold'])

    def _real_predict_proba(self, data_instance):
        p = self.local_config['parameters']
        z = (self.count_triangles(data_instance.data) - p['threshold'] + 0.5) / p.get('scale', 4)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.array([1.0 - p1, p1])
