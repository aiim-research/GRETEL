import random
from src.dataset.dataset_factory import DatasetFactory
from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.evaluation.evaluator_base import Evaluator
from src.explainer.explainer_factory import ExplainerFactory
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_factory import OracleFactory
from src.utils.context import Context


class EvaluatorManager:

    def __init__(self, context: Context) -> None:
        self.context = context
        
        self.context.factories['datasets'] = DatasetFactory(context.dataset_store_path)
        self.context.factories['embedders'] = EmbedderFactory(context)
        self.context.factories['oracles'] = OracleFactory(context, context.conf['oracles'])
        self.context.factories['explainers'] = ExplainerFactory(context.explainer_store_path)
        self.context.factories['metrics'] = EvaluationMetricFactory(context.conf)
        self._output_store_path = context.output_store_path

        self.datasets = []
        self.oracles = []
        self.explainers = []
        self._evaluators = []
        self.evaluation_metrics = []

    
    @property
    def evaluators(self):
        return self._evaluators


    @evaluators.setter
    def evaluators(self, new_evaluators_list):
        self._evaluators = new_evaluators_list
        

    def generate_synthetic_datasets(self):
        """Generates the synthetic datasets and stores them on disk, allowing them to be loaded later
         by the other methods without the need for them to be generated on the fly

        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        # Given the datasets are generated on the fly, this method only calls the get_dataset_by_name method
        for dataset_dict in self.context.conf['datasets']:
            self.context.factories['datasets'].get_dataset_by_name(dataset_dict)


    def train_oracles(self):
        """Trains the oracles and store them on disk, allowing to use them later without needing
         to train again. Trains one oracle of each kind for each dataset.
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        dataset_dicts = self.context.conf['datasets']
        oracle_dicts = self.context.conf['oracles']

        # Create the datasets needed for fitting the oracles
        for dataset_dict in dataset_dicts:
            self.datasets.append(self.context.factories['datasets'].get_dataset_by_name(dataset_dict))

        # For each dataset create and fit all oracles
        for dataset in self.datasets:
            for oracle_dict in oracle_dicts:
                oracle_dict['dataset'] = dataset
                # The get_oracle_by_name method returns a fitted oracle
                oracle = self.context.factories['oracles'].get_oracle(oracle_dict)
                self.oracles.append(oracle)

        # The goal of this method is to train the oracles and store them on disk to use them later
        # For this reason we clean the datasets and oracles lists after using them
        self.datasets = []
        self.oracles = []

        
    def create_evaluators(self):
        """Creates one evaluator for each combination of dataset-oracle-explainer using the chosen metrics
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        dataset_dicts = self.context.conf['datasets']
        oracle_dicts = self.context.conf['oracles']
        metric_dicts = self.context.conf['evaluation_metrics']
        explainer_dicts = self.context.conf['explainers']

        # Create the datasets
        for dataset_dict in dataset_dicts:
            self.datasets.append(self.context.factories['datasets'].get_dataset_by_name(dataset_dict))

        # Create the evaluation metrics
        for metric_dict in metric_dicts:
            eval_metric = self.context.factories['metrics'].get_evaluation_metric_by_name(metric_dict)
            self.evaluation_metrics.append(eval_metric)

        #TODO: DANGEROUS wrong logic: explainer creation must be inseterd in the nested loop to avoid side effects. Copy the snippet before passing it.
        for explainer_dict in explainer_dicts:
            explainer = self.context.factories['explainers'].get_explainer_by_name(explainer_dict, self.context.factories['metrics'])
            self.explainers.append(explainer)

        #TODO: Shuffling dataset and explainers creation. Must be better implemented after refactoring.
        random.shuffle(self.datasets)
        random.shuffle(self.explainers)
        random.shuffle(oracle_dicts)

        evaluator_id = 0
        for dataset in self.datasets:
            for explainer in self.explainers:
                for oracle_dict in oracle_dicts:
                    oracle_dict['dataset'] = dataset
                    # The get_oracle_by_name method returns a fitted oracle
                    oracle = self.context.factories['oracles'].get_oracle(oracle_dict)

                    # Creating the evaluator
                    evaluator = Evaluator(evaluator_id, dataset, oracle, explainer, self.evaluation_metrics,
                                             self._output_store_path, self.context.run_number)

                    # Adding the evaluator to the evaluator's list
                    self.evaluators.append(evaluator)

                    # increasing the evaluator id counter
                    evaluator_id +=1

                
    def evaluate(self):
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        for evaluator in self.evaluators:
            evaluator.evaluate()

        
    def evaluate_multiple_runs(self, n_runs):
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics.
        Each evaluator is run "n_runs" times
         
        -------------
        INPUT: n_runs the number of times the evaluate method of each evaluator is going to be called

        -------------
        OUTPUT: None
        """
        for evaluator in self.evaluators:
            for i in range(0, n_runs):
                evaluator.evaluate()