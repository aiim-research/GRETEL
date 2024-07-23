import sys

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import  init_dflts_to_of 
from src.evaluation.future.metrics.base import EvaluationMetric
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation




class InstabilityMetric(EvaluationMetric):
    """
    Verifies how much is the variation in the produced counterfactuals in relation to the variation in the input
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

        dst_metric='src.utils.metrics.ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        self.name = 'instability'
        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])



    def evaluate(self, explanation: LocalGraphCounterfactualExplanation):
        input_inst = explanation.input_instance

        dataset = explanation.dataset
        oracle = explanation.oracle
        explainer = explanation.explainer

        # Find the closest instance to the input instance in the dataset
        closer_inst = None
        closer_inst_dst = sys.float_info.max # This variable will contain the distance between the two instances from the dataset
        for example in dataset.instances:
            # The instance sould belong to the same class than the input instance
            if example.label == input_inst.label and example.id != input_inst.id:
                example_dst = self.distance_metric.evaluate(input_inst, example, oracle)
                # The instance cannot be equal to the input one
                if example_dst > 0 and example_dst < closer_inst_dst:
                    closer_inst = example
                    closer_inst_dst = example_dst

        # If there was not other instance of the same class in the dataset then return an error value
        if closer_inst is None:
            return -1
        
        # Get the explanation for the closest instance
        closer_inst_explanation = explainer.explain(closer_inst)

        term1 = 1/(1+closer_inst_dst)
        term2 = 1/(len(explanation.counterfactual_instances)*len(closer_inst_explanation.counterfactual_instances))

        term3 = 0

        # Get the distance between all the counterfactual examples of both explanations
        # (the one for the input instance and the one for the closest instance to input instance)
        for input_inst_cf in explanation.counterfactual_instances:
            for closer_inst_cf in closer_inst_explanation.counterfactual_instances:
                term3 += self.distance_metric.evaluate(input_inst_cf, closer_inst_cf, oracle)

        
        instability = term1*term2*term3

        return instability

        
