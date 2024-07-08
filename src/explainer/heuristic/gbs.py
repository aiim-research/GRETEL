import random
import itertools
import numpy as np
import copy

from src.core.explainer_base import Explainer
from src.core.factory_base import get_class, get_instance_kvargs
from src.utils.cfg_utils import init_dflts_to_of
from src.dataset.instances.graph import GraphInstance
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import  inject_dataset, inject_oracle


class GenericBidirectionalSearchExplainer(Explainer):
    def check_configuration(self):
        super().check_configuration()

        inject_dataset(self.local_config['parameters']['aggregator'], self.dataset)
        inject_oracle(self.local_config['parameters']['aggregator'], self.oracle)

        for exp in self.local_config['parameters']['explainers']:
            exp['parameters']['fold_id'] = self.local_config['parameters']['fold_id']
            # In any case we need to inject oracle and the dataset to the model
            inject_dataset(exp, self.dataset)
            inject_oracle(exp, self.oracle)
        
