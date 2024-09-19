from src.core.explainer_base import Explainer
from src.core.factory_base import get_class, get_instance_kvargs
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import  inject_dataset, inject_oracle


class GenerateMinimizeExplainer(Explainer, Trainable):
    """This meta-explainer uses an explanation method to generate a first counterfactual and
    then a minimizer function to reduce the distance between the counterfactual and the original
    instance"""

    def check_configuration(self):
        super().check_configuration()

        if 'generator' not in self.local_config['parameters']:
            raise Exception('A generate-minimize method requires a generator')
        
        if 'minimizer' not in self.local_config['parameters']:
            raise Exception('A generate-minimize method requires a minimizer')

        # Inject the oracle and the dataset into the generator explainer
        inject_dataset(self.local_config['parameters']['generator'], self.dataset)
        inject_oracle(self.local_config['parameters']['generator'], self.oracle)

        # Inject the oracle and the dataset into the minimizer
        inject_dataset(self.local_config['parameters']['minimizer'], self.dataset)
        inject_oracle(self.local_config['parameters']['minimizer'], self.oracle)


    def init(self):
        super().init()

        self.explanation_generator = get_instance_kvargs(self.local_config['parameters']['generator']['class'], 
                                                          {'context':self.context,'local_config': self.local_config['parameters']['generator']})
        
        self.explanation_minimizer = get_instance_kvargs(self.local_config['parameters']['minimizer']['class'], 
                                                          {'context':self.context,'local_config': self.local_config['parameters']['minimizer']})
        

    def explain(self, instance):
        pass