from src.core.factory_base import Factory
from src.utils.cfg_utils import inject_dataset
class OracleFactory(Factory):      
    
    def get_oracle(self, oracle_snippet, dataset):
        inject_dataset(oracle_snippet, dataset)
        return self._get_object(oracle_snippet)
            
    def get_oracles(self, config_list, dataset):
        return [self.get_oracle(obj, dataset) for obj in config_list]