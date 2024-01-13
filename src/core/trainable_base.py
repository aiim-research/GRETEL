import pickle
import time
from abc import ABCMeta, abstractmethod
from src.core.savable import Savable
from src.dataset.dataset_base import Dataset
from src.utils.cfg_utils import retake_dataset
from src.utils.context import Context,clean_cfg

class Trainable(Savable,metaclass=ABCMeta):

    def __init__(self, context: Context, local_config):
        self.dataset: Dataset = retake_dataset(local_config)
        super().__init__(context, local_config)

    def load_or_create(self, condition=False):
        super().load_or_create(self._to_retrain() or condition)        

    def _to_retrain(self):
        retrain = self.local_config['parameters'].get('retrain', False)
        self.local_config['parameters']['retrain']= False
        return retrain

    def retrain(self):
        self.fit()
        self.write()
        self.context.logger.info(str(self)+" re-saved.")

    def fit(self):
        stime = time.time()
        self.real_fit()
        if hasattr(self, 'device') and self.device is not None:
            self.context.logger.info(self.__class__.__name__+" trained on "+self.device+" in: "+str((time.time()-stime))+" secs")   
        else:
            self.context.logger.info(self.__class__.__name__+" trained in: "+str((time.time()-stime))+" secs")  
        
    def create(self):
        self.fit()

    def write(self):
        filepath = self.context.get_path(self)
        dump = {
            "model" : self.model,
            "config": clean_cfg(self.local_config)
        }
        with open(filepath, 'wb') as f:
          pickle.dump(dump, f)
      
    def read(self):
        dump_file = self.context.get_path(self)        
        if self.saved:
            with open(dump_file, 'rb') as f:
                dump = pickle.load(f)
                self.model = dump['model']
                #self.local_config = dump['config']

    @abstractmethod
    def real_fit(self):
        pass

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['fold_id'] =  self.local_config['parameters'].get('fold_id', -1)
        self.fold_id = self.local_config['parameters']['fold_id'] 
    

    