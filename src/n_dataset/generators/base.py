from abc import ABCMeta, abstractmethod

from src.core.configurable import Configurable
from src.n_dataset.dataset_base import Dataset
from src.utils.context import Context


class Generator(Configurable, metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config, dataset=None) -> None:
        self.dataset = dataset
        super().__init__(context, local_config)
        self.current = 0
        
    @abstractmethod
    def get_num_instances(self):
        pass
    
    @abstractmethod
    def init(self):  
        pass
    
    @abstractmethod
    def generate_dataset(self):
        pass
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.current + 1) < self.get_num_instances():
            self.current += 1
            return self.dataset.instances[self.current]
        raise StopIteration
    
    def reset_iterator(self):
        self.current = 0