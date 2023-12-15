from src.core.factory_base import Factory
from src.utils.cfg_utils import inject_dataset, inject_oracle
class ExplainerFactory(Factory):

    def get_explainer(self, explainer_snippet, dataset, oracle):
        inject_dataset(explainer_snippet, dataset)
        inject_oracle(explainer_snippet, oracle)        
        return self._get_object(explainer_snippet)
            
    def get_explainers(self, config_list, dataset, oracle):
        return [self.get_explainer(obj, dataset, oracle) for obj in config_list]