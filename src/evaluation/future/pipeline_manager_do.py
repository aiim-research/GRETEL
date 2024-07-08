import random
from src.dataset.dataset_factory import DatasetFactory
from src.evaluation.future.stages.stage_factory import StageFactory
from src.evaluation.future.evaluator_base import Evaluator
from src.explainer.explainer_factory import ExplainerFactory
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_factory import OracleFactory
from src.utils.context import Context
from src.evaluation.future.pipeline import Pipeline


class PipelineManager:

    def __init__(self, context: Context) -> None:
        self.context = context
        self._output_store_path = self.context.output_store_path

        self._pipelines = []
        self.context.factories['datasets'] = DatasetFactory(context)
        self.context.factories['embedders'] = EmbedderFactory(context)
        self.context.factories['oracles'] = OracleFactory(context)
        self.context.factories['explainers'] = ExplainerFactory(context)
        self.context.factories['stages'] = StageFactory(context)

        self._create_pipelines()
    
    @property
    def evaluators(self):
        return self._pipelines

    def _create_pipelines(self):
        # Get the lists of main componets from the main configuration file.
        '''datasets_list = self.context.conf['datasets']
        oracles_list = self.context.conf['oracles']'''

        do_pairs_list = self.context.conf['do-pairs']
        stages_list = self.context.conf['stages']
        explainers_list = self.context.conf['explainers']
        stages = []

        # Shuffling dataset_oracles pairs and explainers will enabling by chance
        # parallel distributed cration and training.
        random.shuffle(do_pairs_list)
        random.shuffle(explainers_list) 

        # Instantiate the evaluation metrics that will be used for the evaluation;
        stages = self.context.factories['stages'].get_stages(stages_list)   

        for explainer_snippet in explainers_list:
            for do_pair_snippet in do_pairs_list:
                # The get_dataset method return an already builded/loaded/generated dataset with all its features already in place;
                dataset = self.context.factories['datasets'].get_dataset(do_pair_snippet['dataset'])
                
                # The get_oracle method returns a fitted oracle on the dataset;
                oracle = self.context.factories['oracles'].get_oracle(do_pair_snippet['oracle'], dataset)                    

                # The get_explainer method returns an (fitted in case is trainable) explainer for the dataset and the oracle;                
                explainer = self.context.factories['explainers'].get_explainer(explainer_snippet, dataset, oracle)       
            
                # Creating the pipeline
                pipeline = Pipeline(context=self.context, 
                                    local_config=self.context.conf, 
                                    scope=self.context._scope,
                                    dataset=dataset,
                                    oracle=oracle,
                                    explainer=explainer,
                                    stages=stages,
                                    results_store_path=self._output_store_path,
                                    run_number=self.context.run_number)

                # Adding the pipeline to the pipelines list
                self._pipelines.append(pipeline)

               
                
    def execute(self) -> None:
        """
        Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics
        """
        for pipeline in self._pipelines:
            pipeline.run()

        
    def execute_multiple_runs(self, n_runs: int) -> None:
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics.
        Each evaluator is run "n_runs" times
        -------------
        INPUT: n_runs the number of times the evaluate method of each evaluator is going to be called
        """
        for pipeline in self._pipelines:
            for i in range(0, n_runs):
                pipeline.run()