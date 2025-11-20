import numpy as np

from src.utils.metrics.ged import graph_edit_distance_metric
from src.future.explanation.base import Explanation
from src.evaluation.future.stages.stage import Stage

from src.LLMexplaneability.feature_extractor import *
from src.LLMexplaneability.prompt_generator import *
from src.LLMexplaneability.llm_explainer import *
from src.LLMexplaneability.flip_rate_evaluation import *


class LLMexplanation(Stage):
    """Provides a text explaination in natural language for the counterfactual examples generated.
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

    def init(self):
        super().init()
        

    def process(self, explanation: Explanation) -> Explanation:
        # Get the input instance from the explanation and get its label
        input_inst = explanation.input_instance
        input_inst_lbl = explanation.oracle.predict(input_inst)
        explanation.oracle._call_counter -= 1

        i = 0
        explanations = {'direct_explanation':[], 'inverse_explanation':[], 'graph_text': [], 'modifications_text': []}

        graph_feature = FeatureExtractor(input_inst, input_inst_lbl)

        for counterfactual in explanation.counterfactual_instances:

            counterfactual_lbl = explanation.oracle.predict(counterfactual)
            explanation.oracle._call_counter -= 1

            counterfactual_feature = FeatureExtractor(counterfactual, counterfactual_lbl)
            prompt_generator = PromptGenerator(graph_feature, counterfactual_feature, explanation.dataset.domain)
            
            prompt = prompt_generator.generate_prompt()
            prompt_generator.export_prompt(f"./lab/llm_explanations/prompt_{explanation.dataset.name}_{input_inst.id}_{i}.txt")

            response = explanation.context.llm.explain_counterfactual(prompt= prompt)
            explanation.context.llm.export_explanation(response, file = f"./lab/llm_explanations/explanation_{explanation.dataset.name}_{input_inst.id}_{i}.txt")

            prompt_generator2 = PromptGenerator(counterfactual_feature, graph_feature, explanation.dataset.domain)
            prompt2 = prompt_generator2.generate_prompt()

            response2 = explanation.context.llm.explain_counterfactual(prompt= prompt2)


            explanations['direct_explanation'].append(response)
            explanations['graph_text'].append( prompt_generator.graph_to_text())
            explanations['modifications_text'].append( prompt_generator.modifications_to_text())
            explanations['inverse_explanation'].append(response2)
        
            i +=1

        self.write_into_explanation(explanation, explanations)
        return explanation

