from abc import ABCMeta, abstractmethod

from os.path import join

from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle, retake_explainer
from src.utils.context import Context

class Plotter(Configurable, metaclass=ABCMeta):       
    
    def init(self):
        super().init()
        self.dataset = retake_dataset(self.local_config)
        self.oracle = retake_oracle(self.local_config)
        self.explainer = retake_explainer(self.local_config)
        self.dump_path = self.local_config['parameters']['dump_path']
    
    @abstractmethod
    def plot(self, read_path):
        raise NotImplemented()
    
    def check_configuration(self):
        super().check_configuration()
        # the path where to read        
        self.local_config['parameters']['dump_path'] = self.local_config['parameters']\
            .get('dump_path', join('output', 'counterfactual', 'svgs'))
