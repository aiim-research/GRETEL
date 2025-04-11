import time

from src.core.explainer_base import Explainer
from src.core.factory_base import get_class, get_instance_kvargs
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import  inject_dataset, inject_oracle
import src.utils.explanations.functions as exp_tools
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation


class GenerateMinimize(Explainer):
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
        # all the components should use the same fold
        self.local_config['parameters']['generator']['parameters']['fold_id'] = self.local_config['parameters']['fold_id']

        # Inject the oracle and the dataset into the minimizer
        inject_dataset(self.local_config['parameters']['minimizer'], self.dataset)
        inject_oracle(self.local_config['parameters']['minimizer'], self.oracle)
        # all the components should use the same fold
        self.local_config['parameters']['minimizer']['parameters']['fold_id'] = self.local_config['parameters']['fold_id']


    def init(self):
        super().init()
        self.logger = self.context.logger

        self.explanation_generator = get_instance_kvargs(self.local_config['parameters']['generator']['class'], 
                                                          {'context':self.context,'local_config': self.local_config['parameters']['generator']})
        
        self.explanation_minimizer = get_instance_kvargs(self.local_config['parameters']['minimizer']['class'], 
                                                          {'context':self.context,'local_config': self.local_config['parameters']['minimizer']})


    def explain(self, instance):

        # Using the generator to obtain an initial explanation
        start_time = time.time()
        initial_explanation = self.explanation_generator.explain(instance)
        # initial_cf  = initial_explanation.counterfactual_instances[0]
        generator_runtime = time.time() - start_time # Getting the runtime of the generator
        initial_explanation._info['runtime'] = generator_runtime # Writing the runtime in the explanation

        # # Getting the predicted label of the initial explanation
        # initial_cf_label = self.oracle.predict(initial_cf)

        # if initial_cf == instance.label:
        #     # the generator was not able to produce a counterfactual
        #     # so we can inmediately return, there is no point in minimizing
        #     self.logger.info(f'The generator could not generate a counterfactual for instance with id {str(instance.id)}')
        #     return initial_explanation
        # else:
        #     self.logger.info(f'The generator generated a counterfactual for instance with id {str(instance.id)}')
        
        # Try to minimize the distance between the counterfactual example and the original instance
        minimum_cf = self.explanation_minimizer.minimize(initial_explanation)

        minimal_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                                    dataset=self.dataset,
                                                                    oracle=self.oracle,
                                                                    explainer=self,
                                                                    input_instance=instance,
                                                                    counterfactual_instances=[minimum_cf])
        
        minimal_explanation._info['generator_explanation'] = initial_explanation
        
        return minimal_explanation
    

    @property
    def name(self):
        # Make the name of the generator_minimizer explainer show the names of the used generator and minimizer
        gen = get_class( self.local_config['parameters']['generator']['class'] ).__name__
        min = get_class( self.local_config['parameters']['minimizer']['class'] ).__name__
        alias = f'GenerateMinimize({gen}-{min})'
        return self.context.get_name(self,alias=alias)