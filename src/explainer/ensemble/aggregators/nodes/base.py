from typing import List

from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.core.configurable import Configurable
from src.dataset.instances.base import DataInstance

from src.utils.cfg_utils import retake_oracle, retake_dataset

class NodeFeatureAggregator(Configurable):
        
    def init(self):
        self.dataset: Dataset = retake_dataset(self.local_config)
        self.oracle: Oracle = retake_oracle(self.local_config)
        super().init()

    def aggregate(self, nodes, instances: List[DataInstance]):
        pass
