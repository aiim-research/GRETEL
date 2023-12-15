import networkx as nx
import numpy as np

from src.n_dataset.instances.graph import GraphInstance
from src.n_dataset.generators.treecycles_fixed import TreeCyclesFixed


class TreeCyclesFixedWithRatio(TreeCyclesFixed):
        
    def check_configuration(self):
        local_config=self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 1000)
        local_config['parameters']['num_nodes_per_instance'] = local_config['parameters'].get('num_nodes_per_instance', 300)
        local_config['parameters']['ratio_nodes_in_cycles'] = local_config['parameters'].get('ratio_nodes_in_cycles', 0.3)
                
        nodes_in_cycle = int(local_config['parameters']['ratio_nodes_in_cycles'] * local_config['parameters']['num_nodes_per_instance'])
        if 'num_cycles' not in local_config['parameters']:
            if 'cycle_size' not in local_config['parameters']:
                local_config['parameters']['cycle_size'] = 3
            local_config['parameters']['num_cycles'] = nodes_in_cycle // local_config['parameters']['cycle_size']
        elif 'cycle_size' not in local_config['parameters']:
            local_config['parameters']['cycle_size'] = nodes_in_cycle // local_config['parameters']['num_cycles']

        if local_config['parameters']['cycle_size'] < 3:
            local_config['parameters']['cycle_size'] = 3

        if local_config['parameters']['num_cycles'] * local_config['parameters']['cycle_size'] > local_config['parameters']['num_nodes_per_instance']:
            local_config['parameters']['cycle_size'] = 3
            local_config['parameters']['num_cycles'] = np.random.randint(0, local_config['parameters']['num_nodes_per_instance'] // 3 + 1)

        super().check_configuration
