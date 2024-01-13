import pickle
from typing import List
import numpy as np

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Subset
from src.dataset.manipulators.base import BaseManipulator
from torch_geometric.loader import DataLoader
from src.core.factory_base import get_instance_kvargs

from src.core.savable import Savable
from src.dataset.instances.base import DataInstance
from src.core.factory_base import get_class
from src.utils.context import clean_cfg


class Dataset(Savable):
    
    def init(self):
        super().init()
        ################### PREAMBLE ###################
        self.instances: List[DataInstance] = []
        
        self.node_features_map = {}
        self.edge_features_map = {}
        self.graph_features_map = {}
        
        self.splits = []
        self._torch_repr = None
        self._class_indices = {}
        
        self._num_nodes = None
        self._num_nodes_values = None
        #################################################
    
        
    def create(self):
        self.generator = get_instance_kvargs(self.local_config['parameters']['generator']['class'],
                                                {
                                                    "context": self.context, 
                                                    "local_config": self.local_config['parameters']['generator'],
                                                    "dataset": self
                                                })
        self._inject_dataset()
            
        self.manipulators: List[BaseManipulator] = self._create_manipulators()
            
        self.generate_splits(n_splits=self.local_config['parameters']['n_splits'],
                             shuffle=self.local_config['parameters']['shuffle'])

       
    def _inject_dataset(self):
        for instance in self.instances:
            instance._dataset = self
            
    def _create_manipulators(self):
        manipulator_instances = []
        for manipulator in self.local_config['parameters']['manipulators']:
            self.context.logger.info("Apply: "+manipulator['class'])
            manipulator_instances.append(get_instance_kvargs(manipulator['class'],
                                {
                                    "context": self.context,
                                    "local_config": manipulator,
                                    "dataset": self
                                }))
            
        return manipulator_instances
    
    def __len__(self):
        return len(self.get_data())

    def get_data(self):
        return self.instances
    
    def get_instance(self, i: int):
        return self.instances[i]
    
    def get_instances_by_class(self,cls):
        if not self._inst_by_cls:
            self._inst_by_cls= []

        if not self._inst_by_cls[cls]:            
            idx = self.class_indices[cls]
            self._inst_by_cls[cls]=[self.instances[i] for i in idx]
        return self._inst_by_cls[cls]
    
    def num_node_features(self):
        return len(self.node_features_map)
    
    def num_edge_features(self):
        return len(self.edge_features_map)
    
    def num_graph_features(self):
        return len(self.graph_features_map)
    
    def class_indices(self):
        if not self._class_indices:
            for i, inst in enumerate(self.instances):
                self._class_indices[inst.label] = self._class_indices.get(inst.label, []) + [i]
        return self._class_indices
    
    def manipulate(self, instance: DataInstance):
        for manipulator in self.manipulators:
            manipulator._process_instance(instance)
    
    @property        
    def num_classes(self):
        return len(self.class_indices())
    
    @property
    def num_nodes(self):
        if not self._num_nodes:
            self._num_nodes = np.min(self.num_nodes_values)
        return self._num_nodes
    
    @property
    def num_nodes_values(self):
        if not self._num_nodes_values:
            self._num_nodes_values = []
            for inst in self.instances:
                self._num_nodes_values.append(len(inst.data))
        return self._num_nodes_values
    
    def get_split_indices(self, fold_id=-1):
        if fold_id == -1:
            #NOTE: i am bit worried that it might be weak if you have sparse indices
            return {'train': list(range(0, len(self.instances))), 'test': list(range(0, len(self.instances))) }
        else:
            return self.splits[fold_id]
    
    def generate_splits(self, n_splits=10, shuffle=True):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        spl = kf.split([g for g in self.instances], [g.label for g in self.instances])
        for train_index, test_index in spl:
            self.splits.append({'train': train_index.tolist(), 'test': test_index.tolist()})
            
    def get_torch_loader(self,
                         fold_id: int=-1,
                         batch_size: int=4, 
                         usage: str='train', 
                         kls: int=-1, 
                         dataset_kls: str='src.dataset.utils.dataset_torch.TorchGeometricDataset',
                         **kwargs):
        """
            Retrieves a DataLoader for Torch instances, facilitating batch processing.

            Parameters:
                - fold_id (int, optional): The fold identifier. Defaults to -1, indicating all folds.
                - batch_size (int, optional): The number of instances in each batch. Defaults to 4.
                - usage (str, optional): The usage type, e.g., 'train' or 'test'. Defaults to 'train'.
                - kls (int, optional): The class identifier. Defaults to -1, indicating all classes.
                - dataset_kls (str, optional): The class name of the (torch) Dataset or (torch_geometric) Dataset to use.
                                            Defaults to 'src.dataset.utils.dataset_torch.TorchGeometricDataset'.
                - **kwargs: Additional keyword arguments to be passed when creating the Dataset instance.

            Returns:
                torch_geometric.loader.DataLoader: A DataLoader configured for the specified Torch instances.

            Usage Example:
                ```
                dataloader = get_torch_loader(fold_id=0, batch_size=32, usage='train', kls=1, dataset_kls='{path-to-dataset-class}')
                ```

            Notes:
                - The DataLoader is created using instances obtained from the `get_torch_instances` method.
                - Batching is performed with the specified `batch_size`.
                - Data shuffling and dropping the last incomplete batch are enabled by default for training.
                - The function leverages the `DataLoader` class from the `torch_geometric.loader` module.
        """
        instances = self.get_torch_instances(fold_id=fold_id, usage=usage, kls=kls, dataset_kls=dataset_kls)
        return DataLoader(instances, batch_size=batch_size, shuffle=True, drop_last=True)
    
    def get_torch_instances(self,
                            fold_id: int=-1,
                            usage: str='train',
                            kls: int=-1,
                            dataset_kls: str='src.dataset.utils.dataset_torch.TorchGeometricDataset',
                            **kwargs):
        """
            Retrieves a subset of Torch instances for a specific fold, usage, and dataset class.

            Parameters:
                - fold_id (int, optional): The fold identifier. Defaults to -1, indicating all folds.
                - usage (str, optional): The usage type, e.g., 'train' or 'test'. Defaults to 'train'.
                - kls (int, optional): The class identifier. Defaults to -1, indicating all classes.
                - dataset_kls (str, optional): The class name of the (torch) Dataset or (torch_geometric) Dataset to use.
                                            Defaults to 'src.dataset.utils.dataset_torch.TorchGeometricDataset'.
                - **kwargs: Additional keyword arguments to be passed when creating the Dataset instance.

            Returns:
                torch.utils.data.Subset: A subset of Torch instances based on the specified parameters.

            Usage Example:
                ```
                subset = get_torch_instances(fold_id=0, usage='train', kls=1, dataset_kls='{path-to-dataset-class}')
                ```

            Notes:
                - If `kls` is specified, instances not belonging to the specified class will be excluded.
                - The function relies on the presence of `get_split_indices` and `class_indices` methods in the parent class.
                - The Dataset instance is created using the specified class name (`dataset_kls`) and additional arguments (`**kwargs`).
        """
        # Retrieve the indices for the specified fold and usage
        indices = self.get_split_indices(fold_id)[usage]
        # If a specific class (kls) is provided, filter out instances not belonging to that class
        if kls != -1:
            indices = list(set(indices).difference(set(self.class_indices()[kls])))
        # Create an instance of the specified dataset class with the given instances and additional arguments
        self._torch_repr = get_class(dataset_kls)(self.instances, **kwargs)
        # Return a Subset of the dataset instances based on the filtered indices
        return Subset(self._torch_repr.instances, indices)
    
    def read(self):
        if self.saved():
            store_path = self.context.get_path(self)
            with open(store_path, 'rb') as f:
                dump = pickle.load(f)
                self.instances = dump['instances']
                self.splits = dump['splits']
                #self.local_config = dump['config']
                self.node_features_map = dump['node_features_map']
                self.edge_features_map = dump['edge_features_map']
                self.graph_features_map = dump['graph_features_map']
                self._num_nodes = dump['num_nodes']
                self._class_indices = dump['class_indices']
                self.manipulators = dump['manipulators']

            #TODO: Attach the dataset back to all the instances
            self._inject_dataset()
            
                
    def write(self):
        store_path = self.context.get_path(self)
        
        dump = {
            "instances" : self.instances,
            "splits": self.splits,
            "config": clean_cfg(self.local_config), 
            "node_features_map": self.node_features_map,
            "edge_features_map": self.edge_features_map,
            "graph_features_map": self.graph_features_map,
            "num_nodes": self._num_nodes,
            "class_indices": self._class_indices,
            "manipulators": self.manipulators      
        }
        
        with open(store_path, 'wb') as f:
            pickle.dump(dump, f)
            
            
    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        if 'generator' not in local_config['parameters']:
            raise ValueError(f'''The "generator" parameter needs to be specified in {self}''')
        
        if 'manipulators' not in local_config['parameters']: # or not len(local_config['parameters']['manipulators']):
            local_config['parameters']['manipulators'] = []
        
        #local_config['parameters']['manipulators'].append(build_default_config_obj("src.dataset.manipulators.centralities.NodeCentrality"))
        #local_config['parameters']['manipulators'].append(build_default_config_obj("src.dataset.manipulators.weights.EdgeWeights"))
            
        local_config['parameters']['n_splits'] = local_config['parameters'].get('n_splits', 10)
        local_config['parameters']['shuffle'] = local_config['parameters'].get('shuffle', True)
        
    
    @property
    def name(self):
        alias = get_class( self.local_config['parameters']['generator']['class'] ).__name__
        return self.context.get_name(self,alias=alias)