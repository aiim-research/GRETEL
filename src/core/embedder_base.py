from abc import ABCMeta, abstractmethod
import pickle
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import clean_cfg

class Embedder(Trainable,metaclass=ABCMeta):
        
    @abstractmethod
    def get_embeddings(self):
        pass
    
    @abstractmethod
    def get_embedding(self, instance):
        pass

    def write(self):#TODO: Support multiple models
        filepath = self.context.get_path(self)
        dump = {
            "model" : self.model,
            "embeddings": self.embedding,
            "config": clean_cfg(self.local_config)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(dump, f)
      
    def read(self):#TODO: Support multiple models
        dump_file = self.context.get_path(self)        
        #TODO: manage the  if not file exist  case even if it is already managed by the general mecanism
        if self.saved:
            with open(dump_file, 'rb') as f:
                dump = pickle.load(f)
                self.model = dump['model']
                self.embedding =  dump['embeddings']
                #self.local_config = dump['config']
    
    


   