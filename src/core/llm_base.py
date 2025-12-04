from abc import ABCMeta, abstractmethod
from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle

class LLM(Configurable, metaclass=ABCMeta): 
    def __init__(self,  context:Context, local_config):
        super().__init__(context, local_config)
       

    def explain_counterfactual(self, system, prompt):
        pass

    def export_explanation(self, explanation = None, system= None, prompt = None, file = "explanation.txt"):
        if explanation == None:
            explanation = self.explain_counterfactual(system, prompt)
        
        with open(file, "w", encoding="utf-8") as f:
            f.write(explanation)          # sobrescribe si existe
    
