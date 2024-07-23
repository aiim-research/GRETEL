import random
from src.dataset.dataset_factory import DatasetFactory
from src.evaluation.future.metrics.metric_factory import MetricFactory
from src.evaluation.future.evaluator_base import Evaluator
from src.explainer.explainer_factory import ExplainerFactory
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_factory import OracleFactory
from src.utils.context import Context


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
        self.context.factories['evaluation_metrics'] = MetricFactory(context)

        self._create_evaluators()
    
    @property
    def evaluators(self):
        return self._evaluators

    def _create_evaluators(self):
        # Get the lists of main componets from the main configuration file.
        '''datasets_list = self.context.conf['datasets']
        oracles_list = self.context.conf['oracles']'''

        do_pairs_list = self.context.conf['do-pairs']
        metrics_list = self.context.conf['evaluation_metrics']
        explainers_list = self.context.conf['explainers']
        evaluation_metrics = []

        # Shuffling dataset_oracles pairs and explainers will enabling by chance
        # parallel distributed cration and training.
        random.shuffle(do_pairs_list)
        random.shuffle(explainers_list) 

        # Instantiate the evaluation metrics that will be used for the evaluation;
        evaluation_metrics = self.context.factories['evaluation_metrics'].get_metrics(metrics_list)   

        for explainer_snippet in explainers_list:
            for do_pair_snippet in do_pairs_list:
                # The get_dataset method return an already builded/loaded/generated dataset with all its features already in place;
                dataset = self.context.factories['datasets'].get_dataset(do_pair_snippet['dataset'])
                
                # The get_oracle method returns a fitted oracle on the dataset;
                oracle = self.context.factories['oracles'].get_oracle(do_pair_snippet['oracle'], dataset)                    

                # The get_explainer method returns an (fitted in case is trainable) explainer for the dataset and the oracle;                
                explainer = self.context.factories['explainers'].get_explainer(explainer_snippet, dataset, oracle)       
            
                # Creating the evaluator
                evaluator = Evaluator(self.context._scope, dataset, oracle, explainer, evaluation_metrics,
                                            self._output_store_path, self.context.run_number)

                # Adding the evaluator to the evaluator's list
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