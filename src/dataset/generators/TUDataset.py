import torch
from os.path import join,exists
from os import makedirs

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

from torch_geometric.datasets import TUDataset as downloader
import torch_geometric.datasets.tu_dataset as tu
from torch_geometric.data import Data
try:
    from torch_geometric.data import DataEdgeAttr
    torch.serialization.add_safe_globals([tu.TUDataset, Data, DataEdgeAttr])
except ImportError:
    torch.serialization.add_safe_globals([tu.TUDataset, Data])

class TUDataset(Generator):

    def prepare_data(self):

        base_path = join(self.context.working_store_path,self.dataset_name)
        self.context.logger.info("Dataset Data Path:\t" + base_path)

        if not exists(base_path):
            self.context.logger.info("Downloading " + self.dataset_name + "...")
            makedirs(base_path, exist_ok=True)
            dataset = downloader(base_path, name=self.dataset_name, use_node_attr=True, use_edge_attr=True)
            if not exists(join(base_path, f'{self.dataset_name}.pkl')):
                torch.save(dataset, join(base_path, f'{self.dataset_name}.pkl'))
                self.context.logger.info(f"Saved dataset {self.dataset_name} in {join(base_path, f'{self.dataset_name}.pkl')}.")
        return base_path        
       
    
    def init(self):
        
        self.dataset_name = self.local_config['parameters']['alias']
        base_path = self.prepare_data()
        # read the dataset and process it
        self.read_file = join(base_path, f'{self.dataset_name}.pkl')
        self.generate_dataset()

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.populate()

    def populate(self):
       data = torch.load(self.read_file, weights_only=False)

       features_map = {f'attribute_{i}': i for i in range(data[0].x.size(1))}
       self.dataset.node_features_map = features_map

       # TODO edge_map, graph_map

       for id, instance in enumerate(data):
            adj_matrix = torch.zeros((instance.x.size(0), instance.x.size(0)), dtype=torch.float)
            adj_matrix[instance.edge_index[0], instance.edge_index[1]] = 1.0 

            edge_features = None
            try:
                edge_features = instance.edge_weights.numpy()
            except AttributeError:
                self.context.logger.info(f'Instance id = {id} does not have edge features.')

            self.dataset.instances.append(GraphInstance(id=id, 
                                                        label=instance.y.item(), 
                                                        data=adj_matrix.numpy(),
                                                        graph_features=None,
                                                        node_features=instance.x.numpy(),
                                                        edge_features=edge_features
                                                        ))