import numpy as np

from src.core.configurable import Configurable

class BaseManipulator(Configurable):
    
    def __init__(self, context, local_config, dataset):
        self.dataset = dataset
        super().__init__(context, local_config)
        
    def init(self):
        super().init()
        self.manipulated = False
        self._process()
         
    def _process(self):
        for instance in self.dataset.instances:
            node_features_map = self.node_info(instance)
            edge_features_map = self.edge_info(instance)
            graph_features_map = self.graph_info(instance)
            self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
            # overriding the features
            # resize in num_nodes x feature dim
            instance.node_features = self.__process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
            instance.edge_features = self.__process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
            instance.graph_features = self.__process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

           

    def _process_instance(self,instance):
        node_features_map = self.node_info(instance)
        edge_features_map = self.edge_info(instance)
        graph_features_map = self.graph_info(instance)
        self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
        # overriding the features
        # resize in num_nodes x feature dim
        instance.node_features = self.__process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
        instance.edge_features = self.__process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
        instance.graph_features = self.__process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

       
    def node_info(self, instance):
        return {}
    
    def graph_info(self, instance):
        return {}
    
    def edge_info(self, instance):
        return {}
    
    def manipulate_features_maps(self, feature_values):
        if not self.manipulated:
            node_features_map, edge_features_map, graph_features_map = feature_values
            self.dataset.node_features_map = self.__process_map(node_features_map, self.dataset.node_features_map)
            self.dataset.edge_features_map = self.__process_map(edge_features_map, self.dataset.edge_features_map)
            self.dataset.graph_features_map = self.__process_map(graph_features_map, self.dataset.graph_features_map)
            self.manipulated = True
    
    def __process_map(self, curr_map, dataset_map):
        _max = max(dataset_map.values()) if dataset_map.values() else -1
        for key in curr_map:
            if key not in dataset_map:
                _max += 1
                dataset_map[key] = _max
        return dataset_map
    
    def __process_features(self, features, curr_map, dataset_map):
        if curr_map:
            if not isinstance(features, np.ndarray):
                features = np.array([])
            try:
                old_feature_dim = features.shape[1]
            except IndexError:
                old_feature_dim = 0
            # If the feature vector doesn't exist, then
            # here we're creating it for the first time
            if old_feature_dim:
                features = np.pad(features,
                                pad_width=((0, 0), (0, len(dataset_map) - old_feature_dim)),
                                constant_values=0)
            else:
                features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))
                
            for key in curr_map:
                index = dataset_map[key]
                features[:, index] = curr_map[key]
                            
        return features