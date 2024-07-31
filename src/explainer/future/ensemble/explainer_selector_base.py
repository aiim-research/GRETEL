from src.core.explainer_base import Explainer
from src.core.factory_base import get_class, get_instance_kvargs
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import  inject_dataset, inject_oracle


class ExplainerSelector(Explainer, Trainable):
    """The base class for the Explainer Selector. It should provide the common logic 
    for explainers that chose receive multiple base explainers and choose which one to use"""

    def check_configuration(self):
        super().check_configuration()

        for exp in self.local_config['parameters']['explainers']:
            exp['parameters']['fold_id'] = self.local_config['parameters']['fold_id']
            # In any case we need to inject oracle and the dataset to the model
            inject_dataset(exp, self.dataset)
            inject_oracle(exp, self.oracle)


    def init(self):
        super().init()
        
        self.base_explainers = [ get_instance_kvargs(exp['class'],
                    {'context':self.context,'local_config':exp}) for exp in self.local_config['parameters']['explainers']]
        


    def explain(self, instance):
        input_label = self.oracle.predict(instance)

        explanations = []
        for explainer in self.base_explainers:
            exp = explainer.explain(instance)
            exp.producer = explainer
            explanations.append(exp)

        result = self.explanation_aggregator.aggregate(explanations)
        result.explainer = self

        return result
    

    def real_fit(self):
        pass

    def write(self):
        pass
      
    def read(self):
        pass