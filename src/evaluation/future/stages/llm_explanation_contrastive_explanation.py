import numpy as np

from src.evaluation.future.stages.metric_stage import MetricStage
from src.future.explanation.base import Explanation
from src.LLMexplaneability.feature_extractor import *
from src.LLMexplaneability.prompt_generator import *
from src.LLMexplaneability.llm_explainer import *
from src.LLMexplaneability.flip_rate_evaluation import *


class LLMexplanationContrastiveExplanation(MetricStage):
    

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

    def init(self):
        super().init()


    def process(self, explanation: Explanation) -> Explanation:

       
        contrastive_explanations = 0

        if(explanation.stages_info.get('src.evaluation.future.stages.llm_explanation.LLMexplanation', False)):

            for direct, inverse in zip(explanation.stages_info['src.evaluation.future.stages.llm_explanation.LLMexplanation']['direct_explanation'], explanation.stages_info['src.evaluation.future.stages.llm_explanation.LLMexplanation']['inverse_explanation']):

                prompt = f"You are given two texts about a graph counterfactual process.\n\n\
Goal: Decide if they describe the SAME process but in OPPOSITE DIRECTIONS\n\
(A: input → counterfactual, B: counterfactual → input).\n\
\n\
Criteria for YES (all must hold):\n\
1) Same graph/entities/features throughout.\n\
2) Each modification in one text has an inverse in the other.\n\
3) Start/end classes are swapped consistently (input class ↔ counterfactual class).\n\
\n\
If any criterion is not met or information is insufficient, answer NO.\n\
\n\
Do not explain or justify. Output exactly one word: YES or NO.\n\
\n\
--- TEXT A ---\n\
{direct}\n\
\n\
--- TEXT B ---\n\
{inverse}"


                response = explanation.context.llm.explain_counterfactual(prompt= prompt)
            

                if response.strip().upper() == "YES":
                    contrastive_explanations +=1

        self.write_into_explanation(explanation, contrastive_explanations / np.max((len(explanation.counterfactual_instances), 1)))
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
