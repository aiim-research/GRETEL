from typing import List

from src.utils.context import Context,clean_cfg
from src.utils.logger import GLogger
from src.core.factory_base import get_class, get_instance_kvargs
from src.future.explanation.base import Explanation
from src.evaluation.future.stages.pipeline import Pipeline
from src.evaluation.future.stages.runtime import Runtime


class MainPipeline(Pipeline):
    """
    This class defines a pipeline of actions (stages) to be performed on a data instance
    For each input instance the pipeline creates an explanation object.
    The pipeline is defined with a list of stages that are execute in order. 
    Each stage performs some action and writes information in the explanation object that will be passed to the next stage.
    """

    def __init__(self, 
                 context: Context, 
                 local_config) -> None:
        
        super().__init__(context=context, local_config=local_config)
        self._logger = GLogger.getLogger()


    def check_configuration(self):
        if len(self.local_config['parameters']['stages']) < 1:
             raise ValueError('The pipeline cannot be empty')
        
        if self.local_config['parameters']['stages'][0]['class'] != Runtime.__class__:
             raise ValueError('The first stage of a Main Pipeline should be a Runtime stage')
        
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()

        self._stages = [get_instance_kvargs(stage['class'],
                    {'context':self.context,'local_config':stage}) for stage in self.local_config['parameters']['stages']]

    
    def process(self, explanation: Explanation) -> Explanation:
        for stage in self._stages:     
                explanation = stage.process(explanation)
        
        return explanation