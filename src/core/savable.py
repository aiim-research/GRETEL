import os
from src.core.configurable import Configurable
from flufl.lock import Lock
from datetime import timedelta
from abc import ABCMeta,abstractmethod

from src.utils.context import Context

class Savable(Configurable,metaclass=ABCMeta):

    def __init__(self, context: Context, local_config):
        super().__init__(context, local_config)
        self.load_or_create()

    @abstractmethod 
    def write(self):
        pass

    @abstractmethod 
    def read(self):
        pass
    
    @abstractmethod
    def create(self):
        pass

    def saved(self):
        return os.path.exists(self.context.get_path(self))
    
    def load_or_create(self, condition=False):
        condition = condition or not self.saved() if condition else not self.saved()
        lock = Lock(self.context.get_path(self)+'.lck',lifetime=timedelta(hours=self.context.lock_release_tout))
        with lock: #TODO: Check if it is possible to move it inside the if TRUE branch below to avoid small locks: NO IT IS NOT POSSIBLE
            if condition:
                self.context.logger.info(f"Creating: {self}")
                self.create()
                self.write()
                self.context.logger.info(f"Saved: {self}")
            else:
                self.context.logger.info(f"Loading: {self}")
                self.read()
    
   