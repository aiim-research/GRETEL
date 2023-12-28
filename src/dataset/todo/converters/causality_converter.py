
import numpy as np
from src.dataset.converters.abstract_converter import ConverterAB
from src.dataset.instances.base import DataInstance
from src.dataset.data_instance_causality import CausalDataInstance
from src.dataset.dataset_base import Dataset

            
class DefaultCausalityConverter(ConverterAB):
    
    def __init__(self, causality_dim_choice=10):
        super(DefaultCausalityConverter, self).__init__()
        
        self.name = 'default_causality_converter'
        
        self.causality_dim_choice = causality_dim_choice
        self.data_causality_dims = []
        
    def convert_instance(self, instance: DataInstance) -> CausalDataInstance:
        converted_instance = CausalDataInstance(instance.id,
                                                graph=instance.graph,
                                                graph_label=instance.graph_label,
                                                node_labels=instance.node_labels,
                                                edge_labels=instance.edge_labels,
                                                graph_dgl=instance.graph_dgl)

        if not hasattr(instance, 'features'):
            converted_instance.features = self.__create_dummy_features(instance)
            
        splt = np.linspace(0.15, 1.0, num=self.causality_dim_choice + 1)
        min_1, max_1 = splt[:self.causality_dim_choice], splt[1:]
        
        gen_causality = np.random.choice(self.causality_dim_choice, size=(1)).astype(float)
        self.data_causality_dims.append(gen_causality)
        
        u = int(gen_causality)
        noise_1 = (max_1[u] - min_1[u]) * np.random.random_sample() + min_1[u]
        feat_x1 = noise_1 + 0.5 * np.mean(instance.node_degrees())
        feat_add = feat_x1.repeat(instance.graph.number_of_nodes()).reshape(-1,1)
        
        self.u_dim = len(np.unique(self.data_causality_dims))
        
        converted_instance.features = np.concatenate([feat_add, converted_instance.features], axis=1)
        converted_instance.causality = gen_causality
        
        return converted_instance
    
    def __create_dummy_features(self, instance: DataInstance):
        return np.random.normal(0, 1, (instance.graph.number_of_nodes(), 1))