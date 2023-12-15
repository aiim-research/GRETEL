import numpy as np
from src.n_dataset.manipulators.base import BaseManipulator

class Causality(BaseManipulator):

    def init(self):
        self.causality_dim_choice = self.local_config['parameters']['causality_dim_choice']
        self.causalities = self._calc_causalities()
        super().init()

    def check_configuration(self):
        self.local_config['parameters']['causality_dim_choice'] = self.local_config['parameters'].get('causality_dim_choice', 10)
        return super().check_configuration()

    def node_info(self, instance):
        u = int(self.causalities[instance.id])
        noise_1 = (self.max_1[u] - self.min_1[u]) * np.random.random_sample() + self.min_1[u]
        feat_x1 = noise_1 + 0.5 * np.mean(instance.degrees())
        feat_add = feat_x1.repeat(instance.num_nodes).reshape(-1,1)
        return { "node_causality": list(feat_add) }
    
    def graph_info(self, instance):
        return { "graph_causality": [self.causalities[instance.id]] }
    
    def _calc_causalities(self):
        splt = np.linspace(0.15, 1.0, num=self.causality_dim_choice + 1)
        self.min_1, self.max_1 = splt[:self.causality_dim_choice], splt[1:]

        return { instance.id:np.random.choice(self.causality_dim_choice) for instance in self.dataset.instances }

    