from abc import ABCMeta, abstractmethod
from typing import final

from src.dataset.dataset_base import Dataset
from src.core.trainable_base import Trainable
from src.utils.context import Context


class Oracle(Trainable,metaclass=ABCMeta):
    def __init__(self, context:Context, local_config) -> None:
        super().__init__(context, local_config)
        self._call_counter = 0
        
    @final
    def predict(self, data_instance):
        """predicts the label of a given data instance
        -------------
        INPUT:
            data_instance : The instance whose class is going to be predicted 
        -------------
        OUTPUT:
            The predicted label for the data instance
        """
        self._call_counter += 1

        return self._real_predict(data_instance)

    @final
    def predict_proba(self, data_instance):
        """predicts the probability estimates for a given data instance
        -------------
        INPUT:
            data_instance : The instance whose class is going to be predicted 
        -------------
        OUTPUT:
            The predicted probability estimates for the data instance
        """
        self._call_counter += 1

        return self._real_predict_proba(data_instance)

    @final
    def get_calls_count(self):
        return self._call_counter
    
    @final
    def reset_call_count(self):
        self._call_counter = 0    

    @final
    def predict_list(self, dataset: Dataset, fold_id=0):
        sptest = dataset.get_split_indices()[fold_id]['test']
        result = [self.predict(dataset.get_instance(i)) for i in sptest]
        return result
   
    '''@abstractmethod'''#TODO: need to be reactivated and implemented. May can be removed accordingly to Mario and GRETEL philosphy
    def evaluate(self, dataset: Dataset, fold_id=0):
        pass
    
    @abstractmethod
    def _real_predict(self, data_instance):
        pass
    
    @abstractmethod
    def _real_predict_proba(self, data_instance):
        pass
