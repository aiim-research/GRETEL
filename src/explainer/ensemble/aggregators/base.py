from typing import List

import numpy as np

from src.core.factory_base import get_instance_kvargs

from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.core.configurable import Configurable
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.nodes.base import NodeFeatureAggregator

from src.utils.cfg_utils import init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset

class ExplanationAggregator(Configurable):

    def init(self):
        self.dataset: Dataset = retake_dataset(self.local_config)
        self.oracle: Oracle = retake_oracle(self.local_config)
        
        inject_dataset(self.local_config['parameters']['node_feature_aggregator'], self.dataset)
        inject_oracle(self.local_config['parameters']['node_feature_aggregator'], self.oracle)
        
        
        self.node_feature_aggregator: NodeFeatureAggregator = get_instance_kvargs(self.local_config['parameters']['node_feature_aggregator']['class'],
                                                                                  {'context':self.context,'local_config': self.local_config['parameters']['node_feature_aggregator']})
        super().init()

    def aggregate(self, instance: DataInstance, explanations: List[DataInstance]):
        aggregated_instance = self.real_aggregate(instance, explanations)
        # we need to combine:
        # 1) node features
        # 2) edge features
        # 3) graph features
        adj = aggregated_instance.data
        edges = np.nonzero(adj)
        # if there's at least one edge that the aggreagtor produced
        # then get the features of the incident nodes
        if edges[0].size:
            node_features = self.node_feature_aggregator.aggregate(
                np.array(list(range(adj.shape[0]))), 
                explanations
            )

            cf_candidate = GraphInstance(id=instance.id,
                                        label=1-instance.label,
                                        data=adj,
                                        node_features=node_features,
                                        dataset=instance._dataset)

            for manipulator in cf_candidate._dataset.manipulators:
                manipulator._process_instance(cf_candidate)
        else:
            cf_candidate = instance
        
        return cf_candidate
            
            
    def real_aggregate(self, instance: DataInstance, explanations: List[DataInstance]):
        pass
    
    
    def check_configuration(self):
        super().check_configuration()
        
        if 'node_feature_aggregator' not in self.local_config['parameters']:
            init_dflts_to_of(self.local_config,
                             'node_feature_aggregator',
                             'src.explainer.ensemble.aggregators.nodes.average.AverageAggregator')
