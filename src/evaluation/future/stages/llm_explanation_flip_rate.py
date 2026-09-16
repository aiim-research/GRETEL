import numpy as np

from src.evaluation.future.stages.metric_stage import MetricStage
from src.future.explanation.base import Explanation
from src.LLMexplaneability.feature_extractor import *
from src.LLMexplaneability.prompt_generator import *
from src.LLMexplaneability.llm_explainer import *
from src.LLMexplaneability.flip_rate_evaluation import *


class LLMexplanationFlipRate(MetricStage):
    

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

    def init(self):
        super().init()
        # if not hasattr(self.context, 'llm'):
        #     explanation.context.llm = LocalLlamaExplainer()
        #     self.context.llm = explanation.context.llm
        # else:
        #     explanation.context.llm = getattr(self.context, 'llm', None)

        # if not hasattr(self.context, 'llm'):
        #     self.logger("ISSUE")


    def process(self, explanation: Explanation) -> Explanation:
        # Get the input instance from the explanation and get its label
        input_inst = explanation.input_instance
        input_inst_lbl = explanation.oracle.predict(input_inst)
        explanation.oracle._call_counter -= 1



        flipped = 0

        if(explanation.stages_info.get('src.evaluation.future.stages.llm_explanation.LLMexplanation', False)):

            for exp, graph_text, modif in zip(explanation.stages_info['src.evaluation.future.stages.llm_explanation.LLMexplanation']['direct_explanation'], explanation.stages_info['src.evaluation.future.stages.llm_explanation.LLMexplanation']['graph_text'], explanation.stages_info['src.evaluation.future.stages.llm_explanation.LLMexplanation']['modifications_text']):

                flipRate = FlipRateEvaluator(graph_text, modif, explanation.dataset.domain ,"EXPLANATION:\n" + exp + "\n\n")
                sys, prompt2 = flipRate.flip_rate_prompt()


                response2 = explanation.context.llm.explain_counterfactual(sytem =sys, prompt= prompt2)
                
                try:
                    flip_modifications = flipRate.parse_proposals(response2)

                    new_graph = flipRate.edit_graph(input_inst, flip_modifications)

                    new_class = explanation.oracle.predict(new_graph)
                    explanation.oracle._call_counter -= 1

                    if new_class != input_inst_lbl:
                        flipped +=1
                except:
                    self.logger.warning("Error parsing flip rate response or editing graph.")

        self.write_into_explanation(explanation, flipped / np.max((len(explanation.counterfactual_instances), 1)))
        return explanation

    @classmethod
    def aggregate(cls, measure_list, instances_correctness_list=None):
        # If no correctness list is provided aggregate all the measures
        if instances_correctness_list is None:
            return super().aggregate(measure_list, instances_correctness_list)
        else: # If correctness list is provided then aggregate only the measures of the correct instances
            filtered_measure_list = [item for item, flag in zip(measure_list, instances_correctness_list) if flag > 0.0]

            # Avoid aggregating an empty list
            if len(filtered_measure_list) > 0:
                return np.mean(filtered_measure_list), np.std(filtered_measure_list)
            else:
                return 0.0, 0.0
