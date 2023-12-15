from abc import ABCMeta
import re
from src.core.grtl_base import Base
from src.utils.logger import GLogger

class Factory(Base,metaclass=ABCMeta):    
      
    def _get_object(self, object_snippet):
        base_obj = get_class(object_snippet['class'])(self.context, object_snippet)
        self.context.logger.info("Created: "+ str(base_obj))
        return base_obj


################ Utilities functions for Object creation ################


def get_instance_kvargs(kls, param):
    GLogger.getLogger().info("Instantiating: "+kls)
    return  get_class(kls)(**param)

def get_instance(kls, param):
    GLogger.getLogger().info("Instantiating: "+kls)
    return  get_class(kls)(param)

def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m
    
__cls_param_ptrn = re.compile('(^.*)'+ '\(' +'(.*)'+'\)')

def build_w_params_string( class_parameters ):
    if  isinstance(class_parameters, str):
        res = __cls_param_ptrn.findall(class_parameters)
        if len(res)==0:
            return get_class(class_parameters)()
        else:
            return  get_class(res[0][0])(**eval(res[0][1]))
    else:   
        return class_parameters 
    

