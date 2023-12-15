from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.explainer.ensemble.ensemble_base import EnsembleExplainer
from src.explainer.ensemble.ensemble_pe import PEEnsembleExplainer
from src.core.explainer_base import Explainer

class EnsembleFactory:

    def __init__(self, explainer_store_path, explainer_factory) -> None:
        self._explainer_id_counter = 0
        self._explainer_store_path = explainer_store_path
        self._explainer_factory = explainer_factory

    def build_explainer(self, explainer_dict, metric_factory : EvaluationMetricFactory) -> EnsembleExplainer:
        ensemble_dict = explainer_dict['inner']

        explainer_name = ensemble_dict['name']
        explainer_parameters = ensemble_dict['parameters']
        weak_explainers_dicts = ensemble_dict['weak_explainers']

        if (weak_explainers_dicts is None or len(weak_explainers_dicts) == 0):
            raise ValueError('''Ensemble require a set of weak explainers''')

        weak_explainers = [self._explainer_factory.get_explainer_by_name(dict, metric_factory) for dict in weak_explainers_dicts]

        if explainer_name == 'pe_ensemble':
            # Returning the explainer
            return self.get_pe_ensemble_explainer(explainer_dict, weak_explainers)
        
        else:
            raise ValueError('''The provided explainer name does not match any explainer provided by the factory''')

    def get_pe_ensemble_explainer(self, config_dict=None, weak_explainers=None) -> Explainer:
        result = PEEnsembleExplainer(self._explainer_id_counter, config_dict, weak_explainers)
        self._explainer_id_counter += 1  
        return result