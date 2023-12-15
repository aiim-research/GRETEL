from abc import ABCMeta
from src.utils.context import Context

class Base(metaclass=ABCMeta):    
    def __init__(self, context: Context):
        super().__init__()
        self.context:Context = context
        
    @property
    def name(self):
        return self.context.get_name(self)
    
    def __str__(self):
        return self.name