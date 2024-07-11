from src.core.factory_base import Factory
from src.utils.cfg_utils import inject_dataset, inject_oracle, inject_explainer, inject_scope, inject_run_number, inject_results_store_path

class EvaluatorFactory(Factory):

    def get_evaluator(self, evaluator_snippet, dataset, oracle, explainer, scope, results_store_path, run_number):
        inject_dataset(evaluator_snippet, dataset)
        inject_oracle(evaluator_snippet, oracle)
        inject_explainer(evaluator_snippet, explainer)
        inject_scope(evaluator_snippet, scope)
        inject_results_store_path(evaluator_snippet, results_store_path)
        inject_run_number(evaluator_snippet, run_number)
        
        return self._get_object(evaluator_snippet)
