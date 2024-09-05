import numpy as np
import networkx as nx

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


class TringlesSquares(Generator):

    def init(self):       
        self.num_instances = self.local_config['parameters']['num_instances']
        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 100)
        

    def generate_dataset(self):     
                  
        for i in range(self.num_instances):
            # Randomly determine if the graph is going to be a triangle or a square
            triangle = np.random.randint(0, 2)  # 2 excluded

            if triangle:
                # Generating a triangle
                G = nx.cycle_graph(3)

                assert len(G.nodes) == 3 and len(G.edges) == 3 # The structure is not a triangle

                g = nx.to_numpy_array(G)
                label = 0  # Graph is a triangle
            else:
                G = nx.cycle_graph(4)

                assert len(G.nodes) == 4 and len(G.edges) == 4 # the structure is not an square

                g = nx.to_numpy_array(G) 
                label = 1  # Graph is an square

            self.dataset.instances.append(GraphInstance(id=i, data=g, label=label, dataset=self.dataset))

            self.context.logger.info(f"Generated instance with id {i} and label={label}")
    
    
    def get_num_instances(self):
        return len(self.dataset.instances)
        

