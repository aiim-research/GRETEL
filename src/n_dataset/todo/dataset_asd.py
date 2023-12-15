from src.dataset.instances.base import DataInstance
from src.n_dataset.dataset_base import Dataset

import networkx as nx
import numpy as np
import os

class ASDLoader(Loader):

    def __init__(self, id, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self.instances = []
        self.name = 'autism_dataset'

    def read_adjacency_matrices(self, dataset_path):
        """
        Reads the dataset from the adjacency matrices
        """
        
        # Path to the instances of the "Typical development class"
        td_class_path = os.path.join(dataset_path, 'td')

        # Path to the instances of the "Autism Spectrum Disorder class"
        asd_class_path = os.path.join(dataset_path, 'asd')
        
        paths = [td_class_path, asd_class_path]

        instance_id = 0
        graph_label = 0
        # For each class folder
        for path in paths:
            for filename in os.listdir(path):
                # avoiding files not related to the dataset
                if 'DS_Store' not in filename:
                    # Reading the adjacency matrix
                    with open(os.path.join(path, filename), 'r') as f:
                        if filename[-3:]=='csv':
                            l = [[int(float(num)) for num in line.split(',')] for line in f] # if .csv 
                        else:
                            l = [[int(num) for num in line.split(' ')] for line in f] # if .txt

                        # Creating the instance
                        inst = DataInstance(instance_id)
                        instance_id += 1    
                        inst.name = filename.split('.')[0]
                        inst.graph_label = graph_label
                        # transforming the numpy array into a graph
                        g_array = np.array(l, dtype=np.int32)
                        inst.from_numpy_array(g_array)
                        
                        # Adding the instance to the instances list
                        self.instances.append(inst)
            graph_label +=1