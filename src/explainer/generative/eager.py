import torch

from src.core.factory_base import get_instance_kvargs
from src.explainer.per_cls_explainer import PerClassExplainer

from src.utils.cfg_utils import init_dflts_to_of
from src.utils.samplers.abstract_sampler import Sampler

class EAGER(PerClassExplainer):

    def init(self):
        super().init()

        self.sampler: Sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                                    self.local_config['parameters']['sampler']['parameters'])
        self.sampler.dataset = self.dataset
                
    def explain(self, instance):          
        with torch.no_grad():  
            res = super().explain(instance)

            rec_nodes, edge_probs = dict(), dict()
            for key, values in res.items():
                # take the node features and edge probabilities
                rec_nodes[key] = values[0]
                edge_probs[key] = torch.sigmoid(values[-1])
                
            cf_instance = self.sampler.sample(instance, self.oracle,
                                              embedded_features=rec_nodes,
                                              edge_probabilities=edge_probs)
            
        return cf_instance if cf_instance else instance
    
    def check_configuration(self):
        self.set_proto_kls('src.explainer.generative.gans.graph.learnable_edges.model.EdgeLearnableGAN')
        super().check_configuration()
        init_dflts_to_of(self.local_config,
                         'sampler',
                         'src.utils.samplers.bernoulli.Bernoulli',
                         sampling_iterations=2000)