from src.core.explainer_base import Explainer
from src.core.factory_base import get_class, get_instance_kvargs
from src.utils.cfg_utils import  inject_dataset, inject_oracle

class Cascade(Explainer):
    """The base class for the Explainer Ensemble. It should provide the common logic 
    for integrating multiple explainers and produce unified explanations"""

    def init(self):
        super().init()
        self.logger = self.context.logger
        self.base_explainers = [ get_instance_kvargs(exp['class'],
                    {'context':self.context,'local_config':exp}) for exp in self.local_config['parameters']['explainers']]


    def explain(self, instance):
        input_label = self.oracle.predict(instance)

        for explainer in self.base_explainers:
            exp = explainer.explain(instance)
            exp.producer = explainer
            
            cf = exp.counterfactual_instances[0]
            cf_label = self.oracle.predict(cf)
            
            if(cf_label != input_label):
                self.logger.info("explainer: " + str(explainer))
                result = exp
                result.explainer = self

                return result

        return None
    

    
    def check_configuration(self):
        super().check_configuration()

        for exp in self.local_config['parameters']['explainers']:
            exp['parameters']['fold_id'] = self.local_config['parameters']['fold_id']
            # In any case we need to inject oracle and the dataset to the model
            inject_dataset(exp, self.dataset)
            inject_oracle(exp, self.oracle)


    def write(self):
        pass
      
    def read(self):
        pass

    @property
    def name(self):
        # alias = get_class( self.local_config['parameters']['aggregator']['class'] ).__name__
        return self.context.get_name(self,alias="testing")