
from abc import ABCMeta, abstractmethod

from copy import deepcopy
from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset


class ConverterAB(metaclass=ABCMeta):
    
    def __init__(self):
        self.name = 'abstract_converter'
        
    @abstractmethod
    def convert_instance(self, instance: DataInstance) -> DataInstance:
        pass
    
    def convert(self, dataset: Dataset) -> Dataset:
        new_dataset = deepcopy(dataset)
        new_dataset.instances = [self.convert_instance(instance) for instance in dataset.instances]
        return new_dataset