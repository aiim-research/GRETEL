from abc import ABC, abstractmethod
from typing import List

from src.dataset.instances.base import DataInstance
from src.core.oracle_base import Oracle


class Sampler(ABC):
    
    def __init__(self):
        self._name = 'abstract_sampler'
    
        
    @abstractmethod
    def sample(self, instance: DataInstance, oracle: Oracle, **kwargs) -> List[DataInstance]:
        pass