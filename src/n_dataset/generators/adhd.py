from os import listdir
from os.path import isfile, join

import numpy as np
import networkx as nx

from src.n_dataset.instances.graph import GraphInstance
from src.n_dataset.generators.base import Generator


class ADHD(Generator):

    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        # Path to the instances of the "Attention Deficit Hyperactivity Disorder class"
        self.adhd_class_path = join(base_path, 'adhd_dataset')  
        self._td_file_path = join(base_path, 'td')
        self.generate_dataset()

    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.read_adjacency_matrices()
    
    def read_adjacency_matrices(self):
        """
        Reads the dataset from the adjacency matrices
        """

        instance_id = 0
        label = 0

        for filename in listdir(self._td_file_path):
            with open(join(self._td_file_path, filename), 'r') as f:
                instance = [[int(num) for num in line.split(' ')] for line in f]
                self.dataset.instances.append(GraphInstance(instance_id, label=label, data=np.array(instance, dtype=np.int32)))
                instance_id += 1

        label += 1

        for filename in listdir(self.adhd_class_path):
            # Reading the graph files
            graph_path = join(self.adhd_class_path, filename)
            adjlist_file = None
            for file in listdir(graph_path):
                if file[-11:]=='_label.json':
                    with open(join(graph_path, file), 'r') as f:
                        graph_label = int(f.readline())
                elif file[-8:] == '.adjlist':
                    adjlist_file = file

            # Reading the adjacency matrix
            adjlist = join(graph_path, adjlist_file)
            npdata = _nd_array_from_adjlist(adjlist)

            # Creating the instance
            inst = GraphInstance(instance_id, label, npdata)
            instance_id += 1    
            
            # Adding the instance to the instances list
            self.dataset.instances.append(inst)

        

            
def _nd_array_from_adjlist(path: str):
        with open(path, 'r') as f:
            data = []
            while f.readable():
                line = f.readline()
                if not line:
                    break
                if len(line) > 0 and len(line.strip()) > 0 and line[0] != '#':
                    splitted = line.split(' ')
                    vertex_id = int(splitted[0])
                    vertex_arist_lenght = int(splitted[1])
                    data.append([])
                    for _ in range(vertex_arist_lenght):
                        if not f.readable():
                            break
                        edge = int(f.readline().split(' ')[0])
                        data[vertex_id].append(edge)
            result = np.zeros((len(data),len(data)))
            for i,vertex in enumerate(data):
                for edge in vertex:
                    result[i][edge] = 1
            return result