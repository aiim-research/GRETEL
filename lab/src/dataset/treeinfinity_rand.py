import networkx as nx
import numpy as np

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


class TreeInfinityCycles(Generator):

    def init(self):       
        self.num_instances = self.local_config['parameters']['num_instances']
        self.num_nodes_per_instance = self.local_config['parameters']['num_nodes_per_instance']
        self.infinity_cycle_length = self.local_config['parameters'].get('infinity_cycle_length', 6)
        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 1000)
        local_config['parameters']['num_nodes_per_instance'] = local_config['parameters'].get('num_nodes_per_instance', 300)
        local_config['parameters']['infinity_cycle_length'] = local_config['parameters'].get('infinity_cycle_length', 6)
        
        assert(int(local_config['parameters']['infinity_cycle_length']) // 2 >= 3)

    def generate_dataset(self):     
                  
        for i in range(self.num_instances):
            # Generating a random tree containing all the nodes of the instance
            t_graph = nx.to_numpy_array(nx.random_tree(n=self.num_nodes_per_instance))
            
            # Randomly determine if the graph is going to contain an infinity-shaped cycle
            has_infinity_cycle = np.random.randint(0, 2)  # 2 excluded

            if has_infinity_cycle:
                # Adding an infinity-shaped cycle
                infinity_cycle = self.__infinity_cycle()
                t_graph = self._join_graphs_as_adj(t_graph, [infinity_cycle])
                label = 1  # Graph contains an infinity-shaped cycle
            else:
                label = 0  # Graph is a random tree

            self.dataset.instances.append(GraphInstance(id=i, data=t_graph, label=label, dataset=self.dataset))

            self.context.logger.info(f"Generated instance with id {i} and label={label}")
    
    
    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def __infinity_cycle(self):
        cycle_length = self.infinity_cycle_length // 2
        # create the two halves of the infinity shape
        g = nx.cycle_graph(cycle_length)
        h = nx.cycle_graph(cycle_length)
        # rename the nodes of the second half
        h = nx.relabel_nodes(h, {i : cycle_length + i for i in range(max(h.nodes)+1)})
        # get the last node of g and h to connect to the "bridge" node that
        # will close the infinity shape
        max_nodes_g, max_nodes_h = max(g.nodes), max(h.nodes)
        # add the bridge to g and h
        common_node = self.infinity_cycle_length
        g.add_node(common_node)
        h.add_node(common_node)
        # connect the bridge with max_nodes_g and max_nodes_h
        g.add_edge(max_nodes_g, common_node)
        h.add_edge(max_nodes_h, common_node)
        # concatenate the two subgraphs
        inf_cycle = nx.compose(g,h)
        return nx.to_numpy_array(inf_cycle)
        

    def _join_graphs_as_adj(self, base, others):
        Ab = base
        A = Ab
        for other in others:
            Ao = other
            t_node = np.random.randint(0, len(Ab))
            s_node = len(A) + np.random.randint(0, len(Ao))
            A = np.block([[A, np.zeros((len(A), len(Ao)))], [np.zeros((len(Ao), len(A))), Ao]])            
            A[t_node, s_node] = 1
            A[s_node, t_node] = 1

        return A