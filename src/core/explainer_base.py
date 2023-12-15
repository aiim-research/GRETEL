from abc import ABCMeta, abstractmethod
from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle

class Explainer(Configurable, metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config):
        self.dataset = retake_dataset(local_config)
        self.oracle = retake_oracle(local_config)
        super().__init__(context, local_config)
        

    @abstractmethod
    def explain(self, instance):
        pass
