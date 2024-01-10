import numpy as np
from src.dataset.manipulators.base import BaseManipulator

class Causality(BaseManipulator):

    def check_configuration(self):
        super().check_configuration()
        self.causality_dim_choice = self.local_config['parameters'].get('causality_dim_choice', 10)
        self.causalities = self._calc_causalities() 
        #TODO Configuration of the Maniupalators affect the dataset signature: it must be invoked before performing the signature on the dataset
        # actually it doe not work properly with the defaults. This courrent above (init here the self.causalities) is a workaround.
    

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

    