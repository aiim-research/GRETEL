from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import  inject_dataset, inject_oracle

class ExplainerEnsemble(Explainer, Trainable):
    """The base class for the Explainer Ensemble. It should provide the common logic 
    for integrating multiple explainers and produce unified explanations"""

    def init(self):
        super().init()

        self.explanation_aggregator = get_instance_kvargs(self.local_config['parameters']['aggregator']['class'], 
                                                          {'context':self.context,'local_config': self.local_config['parameters']['aggregator']})
        
        self.base_explainers = [ get_instance_kvargs(exp['class'],
                    {'context':self.context,'local_config':exp}) for exp in self.local_config['parameters']['explainers']]


    def explain(self, instance):
        input_label = self.oracle.predict(instance)

        explanations = []
        for explainer in self.base_explainers:
            exp = explainer.explain(instance)
            exp.producer = explainer
            explanations.append(exp)

        result = self.explanation_aggregator.aggregate(instance, explanations)

        return result
    

    def real_fit(self):
        pass

    
    def check_configuration(self):
        super().check_configuration()

        inject_dataset(self.local_config['parameters']['aggregator'], self.dataset)
        inject_oracle(self.local_config['parameters']['aggregator'], self.oracle)

        for exp in self.local_config['parameters']['explainers']:
            exp['parameters']['fold_id'] = self.local_config['parameters']['fold_id']
            # In any case we need to inject oracle and the dataset to the model
            inject_dataset(exp, self.dataset)
            inject_oracle(exp, self.oracle)


    def write(self):
        pass
      
    def read(self):
        pass