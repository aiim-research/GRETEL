import networkx as nx
import numpy as np

from src.n_dataset.generators.base import Generator
from src.n_dataset.instances.graph import GraphInstance


class TreeCyclesFixed(Generator):
    
    def init(self):        
        self.num_instances = self.local_config['parameters']['num_instances']
        self.num_nodes_per_instance = self.local_config['parameters']['num_nodes_per_instance']
        self.num_cycles = self.local_config['parameters']['num_cycles']
        self.cycle_size = self.local_config['parameters']['cycle_size']
        self.generate_dataset()
        
    def check_configuration(self):
        super().check_configuration
        local_config=self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 1000)
        local_config['parameters']['num_nodes_per_instance'] = local_config['parameters'].get('num_nodes_per_instance', 300)
        
        if 'cycle_size' not in local_config['parameters']:
            raise Exception("The size of each cycle must be given.")
        
        if 'num_cycles' not in local_config['parameters']:
            raise Exception("The number of cycles must be given.")

        if local_config['parameters']['cycle_size'] < 3:
            raise ValueError("The size of each cycle must be at least 3.")

        if local_config['parameters']['num_cycles'] * local_config['parameters']['cycle_size'] > local_config['parameters']['num_nodes_per_instance']:
            raise ValueError("The size of each cycle multiplied by the number of cycles must be less than or equal to the number of nodes in the graph.")

    def generate_dataset(self):
               
        for i in range(self.num_instances):
            # Randomly determine if the graph is going to contain cycles or just be a tree
            has_cycles = np.random.randint(0,2) # 2 excluded
            # If the graph will contain cycles
            if(has_cycles):
                cycles = []
                left = self.num_nodes_per_instance - self.num_cycles * self.cycle_size
                            
                for _ in range(self.num_cycles):
                    cycles.append(nx.cycle_graph(self.cycle_size))
                
                tc_graph = self._join_graphs_as_adj(nx.random_tree(n=left), cycles)     
                self.dataset.instances.append(GraphInstance(id=i, data=tc_graph, label=1))
            else:
                # Generating a random tree containing all the nodes of the instance
                t_graph = nx.random_tree(n=self.num_nodes_per_instance)
                self.dataset.instances.append(GraphInstance(id=i, data=nx.to_numpy_array(t_graph), label=0))

            self.context.logger.info("Generated instance with id:"+str(i))
        
    
    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def _join_graphs_as_adj(self, base, others):
        Ab = nx.to_numpy_array(base)
        A = Ab
        for other in others:
            Ao = nx.to_numpy_array(other)
            t_node = np.random.randint(0,len(Ab))
            s_node = len(A) + np.random.randint(0,len(Ao))
            A = np.block([[A,np.zeros((len(A),len(Ao)))],[np.zeros((len(Ao),len(A))), Ao]])            
            A[t_node,s_node]=1
            A[s_node,t_node]=1

        return A
