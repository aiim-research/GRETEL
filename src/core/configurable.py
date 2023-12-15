from abc import ABCMeta,abstractmethod
from src.core.grtl_base import Base
from src.utils.context import Context
import copy

class Configurable(Base,metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config):
        super().__init__(context)
        self.local_config = local_config#copy.deepcopy(local_config)
        self.check_configuration()
        self.init()
    
    def check_configuration(self):
        self.local_config['parameters'] = self.local_config.get('parameters',{})

    @abstractmethod
    def init(self):
        pass

