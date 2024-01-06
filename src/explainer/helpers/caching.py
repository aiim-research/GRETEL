

import pickle
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import clean_cfg

class ExplainerCache(Explainer, Trainable):
    def explain(self, instance):
        if instance in self.cache.keys():
            return self.cache[instance.id]
        else:
            expl = self.instance.explain(instance)
            self.cache[instance.id]=expl

    def init(self):
        self.cache = {}
        self.instance.init()
        self.trainable = True if isinstance(self.instance, Trainable) else False

    def real_fit(self):
        if self.trainable:
            self.instance.real_fit()

    def load_or_create(self, condition=False):
        super().load_or_create(self._to_retrain() or condition)        
        if self.trainable:
            self.instance.load_or_create(condition)        

    def _to_retrain(self):
        if self.trainable:
            return self.instance._to_retrain()
        return False

    def retrain(self):
        if self.trainable:
            self.instance.retrain()
   
    def fit(self):
        if self.trainable:
            self.instance.fit()
        
    def create(self):
        if self.trainable:
            self.instance.create()

    def write(self):
        filepath = self.context.get_path(self)
        dump = {
            "cache" : self.cache,
            "config": clean_cfg(self.local_config)
        }
        with open(filepath, 'wb') as f:
          pickle.dump(dump, f)

        if self.trainable:
            self.instance.write()
      
    def read(self):
        dump_file = self.context.get_path(self)        
        if self.saved:
            with open(dump_file, 'rb') as f:
                dump = pickle.load(f)
                self.cache = dump['cache']

        if self.trainable:
            self.instance.read()

    def real_fit(self):
        if self.trainable:
            self.instance.real_fit()

    def check_configuration(self):
        self.instance = self.local_config['instance']
        self.instance.check_configuration()
       
    

    