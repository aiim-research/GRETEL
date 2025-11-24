from src.core.factory_base import Factory
from src.utils.cfg_utils import inject_dataset
class LLMFactory(Factory):      
    
    def get_llm(self, llm_snippet):
        return self._get_object(llm_snippet)
            
    def get_llms(self, config_list):
        return [self.get_oracle(obj) for obj in config_list]