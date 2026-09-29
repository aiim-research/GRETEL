import pickle


store_path = 'data/datasets/backup/TreeCycles-6482e6d0d05bacea26631eef5a1144f8'
with open(store_path, 'rb') as f:
    dump = pickle.load(f)
    '''self.instances = dump['instances']
    self.splits = dump['splits']
    #self.local_config = dump['config']
    self.node_features_map = dump['node_features_map']
    self.edge_features_map = dump['edge_features_map']
    self.graph_features_map = dump['graph_features_map']
    self._num_nodes = dump['num_nodes']
    self._class_indices = dump['class_indices'] '''

print(dump)