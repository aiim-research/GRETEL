import torch
import copy

from src.core.factory_base import get_instance_kvargs
from src.explainer.per_cls_explainer import PerClassExplainer
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.samplers.abstract_sampler import Sampler
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation

class RSGG(PerClassExplainer):

    def init(self):
        super().init()
        self.sampler: Sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                        self.local_config['parameters']['sampler']['parameters'])
                
    def explain(self, instance):          
        with torch.no_grad():  
            res = super().explain(instance)

            embedded_features, edge_probs = dict(), dict()
            for key, values in res.items():
                # take the node features and edge probabilities
                embedded_features[key] = values[0]
                edge_probs[key] = values[-1]

            cf_instance = self.sampler.sample(instance, self.oracle,
                                              embedded_features=embedded_features,
                                              edge_probabilities=edge_probs)
            
            if cf_instance:
                # Building the explanation instance
                exp = LocalGraphCounterfactualExplanation(context=self.context,
                                                          dataset=self.dataset,
                                                          oracle=self.oracle,
                                                          explainer=self,
                                                          input_instance=instance,
                                                          counterfactual_instances=[cf_instance])
            else:
                # Building the default explanation instance
                exp = LocalGraphCounterfactualExplanation(context=self.context,
                                                          dataset=self.dataset,
                                                          oracle=self.oracle,
                                                          explainer=self,
                                                          input_instance=instance,
                                                          counterfactual_instances=[copy.deepcopy(instance)])
            return exp
            
    
    def check_configuration(self):
        self.set_proto_kls('src.explainer.generative.gans.graph.model.GAN')
        super().check_configuration()
        #The sampler must be present in any case
        init_dflts_to_of(self.local_config,
                         'sampler',
                         'src.utils.samplers.partial_order_samplers.PositiveAndNegativeEdgeSampler',
                         sampling_iterations=500)