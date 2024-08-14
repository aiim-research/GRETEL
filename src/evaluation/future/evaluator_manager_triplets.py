import random
from src.dataset.dataset_factory import DatasetFactory
from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.evaluation.evaluator_base import Evaluator
from src.explainer.explainer_factory import ExplainerFactory
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_factory import OracleFactory
from src.utils.context import Context
from src.evaluation.future.evaluator_factory import EvaluatorFactory


class EvaluatorManager:

    def __init__(self, context: Context) -> None:
        self.context = context
        self._output_store_path = self.context.output_store_path

        self._evaluators = []
        
        #NOTE: Move the Factories creation outside?
        self.context.factories['datasets'] = DatasetFactory(context)
        self.context.factories['embedders'] = EmbedderFactory(context)
        self.context.factories['oracles'] = OracleFactory(context)
        self.context.factories['explainers'] = ExplainerFactory(context)
        self.context.factories['evaluators'] = EvaluatorFactory(context)
        # self.context.factories['metrics'] = EvaluationMetricFactory(context.conf)

        self._create_evaluators()
    
    @property
    def evaluators(self):
        return self._evaluators

    def _create_evaluators(self):
        # Get the lists of main componets from the main configuration file.

        triplets_list = self.context.conf['doe-triplets']
        evaluator_conf = self.context.conf['evaluator']
        experiment_scope = self.context.conf['experiment']['scope']
        # results_store_path = self.context.conf['']

        # Shuffling dataset_oracles pairs and explainers will enabling by chance
        # parallel distributed cration and training.

        #shuffle triplets
        random.shuffle(triplets_list)

        for triplet_snippet in triplets_list:

            dataset = self.context.factories['datasets'].get_dataset(triplet_snippet['dataset'])
            oracle = self.context.factories['oracles'].get_oracle(triplet_snippet['oracle'], dataset)
            explainer = self.context.factories['explainers'].get_explainer(triplet_snippet['explainer'], dataset, oracle)
            evaluator = self.context.factories['evaluators'].get_evaluator(evaluator_snippet=evaluator_conf, 
                                                                           dataset=dataset, 
                                                                           oracle=oracle,
                                                                           explainer=explainer,
                                                                           scope=experiment_scope,
                                                                           results_store_path=self._output_store_path,
                                                                           run_number=self.context.run_number)

            self._evaluators.append(evaluator)


                
    def evaluate(self):
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        for evaluator in self._evaluators:
            evaluator.evaluate()

        
    def evaluate_multiple_runs(self, n_runs):
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics.
        Each evaluator is run "n_runs" times
         
        -------------
        INPUT: n_runs the number of times the evaluate method of each evaluator is going to be called

        -------------
        OUTPUT: None
        """
        for evaluator in self._evaluators:
            for i in range(0, n_runs):
                evaluator.evaluate()


    def pickle_explanations(self, store_path):
        for evaluator in self._evaluators:
            evaluator.pickle_explanations(store_path)