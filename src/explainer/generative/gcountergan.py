import torch

from src.core.factory_base import get_instance_kvargs
from src.explainer.per_cls_explainer import PerClassExplainer
from src.utils.samplers.abstract_sampler import Sampler

from src.utils.cfg_utils import init_dflts_to_of

class GCounteRGAN(PerClassExplainer):


    def init(self):
        super().init()
        self.sampler: Sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                        self.local_config['parameters']['sampler']['parameters'])
        
        self.send_to_device()                

    def explain(self, instance):
            
        with torch.no_grad():
            # here I have a dictionary of key-value pairs like this
            # { 0 : probabilistic adj matrix, 1: probabilistic adj matrix, ... , n: probabilistic adj matrix}
            res = super().explain(instance)
            
            embedded_features, edge_probs = dict(), dict()
            for key, prob_adj_matrix in res.items():
                # take the node features and edge probabilities
                embedded_features[key] = torch.ones(prob_adj_matrix.shape)
                edge_probs[key] = prob_adj_matrix
            
            cf_instance = self.sampler.sample(instance, self.oracle, **{'embedded_features': embedded_features,
                                                                        'edge_probabilities': edge_probs})            
        return cf_instance if cf_instance else instance
    
    def check_configuration(self):
        self.set_proto_kls('src.explainer.generative.gans.image.model.GAN')
        super().check_configuration()
        #The sampler must be present in any case
        init_dflts_to_of(self.local_config,
                         'sampler',
                         'src.utils.samplers.bernoulli.Bernoulli',
                         sampling_iterations=1)
    