from abc import ABCMeta, abstractmethod
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.utils.context import Context
from src.utils.cfg_utils import retake_dataset, retake_oracle

#TODO: It Can Be Removed
class TrainableExplainer(Trainable, Explainer, metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config):
        self.dataset = retake_dataset(local_config)
        self.oracle = retake_oracle(local_config)
        super().__init__(context, local_config)


    @abstractmethod
    def explain(self, instance):
        pass
