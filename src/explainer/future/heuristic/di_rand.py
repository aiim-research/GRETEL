import random
import itertools
import numpy as np
import copy


from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation


class DIRand(Explainer):
    """iRand stands for Iterative Random Explainer, the logic of the explainers is to 
    """
            
    def init(self):
        super().init()

        self.perturbation_percentage = self.local_config['parameters']['p']
        self.tries = self.local_config['parameters']['t']

        self.direction = self.local_config['parameters'].get('direction', 0)

        
    def explain(self, instance):
        l_input_inst = self.oracle.predict(instance)
        nodes = instance.data.shape[0]

        # Create a copy of the adjacency matrix in integer format without modifying instance.data
        adj_matrix = instance.data.astype(int)  # Ensure integer type copy

        # Select edges based on direction, ensuring no self-loops
        if self.direction == -1:  # Remove edges (only existing ones)
            candidate_edges = np.array([(i, j) for i, j in itertools.product(range(nodes), repeat=2) if i != j and adj_matrix[i, j] == 1]) if instance.directed else \
                            np.array([(i, j) for i, j in itertools.combinations(range(nodes), 2) if adj_matrix[i, j] == 1])
        
        elif self.direction == 1:  # Add edges (only missing ones)
            candidate_edges = np.array([(i, j) for i, j in itertools.product(range(nodes), repeat=2) if i != j and adj_matrix[i, j] == 0]) if instance.directed else \
                            np.array([(i, j) for i, j in itertools.combinations(range(nodes), 2) if adj_matrix[i, j] == 0])
        
        else:  # Perturb edges (both adding and removing)
            candidate_edges = np.array([(i, j) for i, j in itertools.product(range(nodes), repeat=2) if i != j]) if instance.directed else \
                            np.array([(i, j) for i, j in itertools.combinations(range(nodes), 2)])

       # Check if we have valid edges to modify
        if len(candidate_edges) == 0:
            return LocalGraphCounterfactualExplanation(
                context=self.context,
                dataset=self.dataset,
                oracle=self.oracle,
                explainer=self,
                input_instance=instance,
                counterfactual_instances=[copy.deepcopy(instance)],
            )

        # Maximum number of modifications
        k = int(len(candidate_edges) * self.perturbation_percentage)

        # Iteratively increase modifications
        for i in range(1, k + 1):  
            for _ in range(self.tries):
                cf_cand_matrix = adj_matrix.copy()

                # Sample edges to modify
                sampled_edges = candidate_edges[np.random.choice(len(candidate_edges), size=i, replace=False)]

                # Apply changes using ^= 1
                cf_cand_matrix[sampled_edges[:, 0], sampled_edges[:, 1]] ^= 1

                if not instance.directed:
                    cf_cand_matrix[sampled_edges[:, 1], sampled_edges[:, 0]] ^= 1  # Mirror update

                # Enforce constraints: prevent unwanted modifications
                if self.direction == -1:  # Only remove edges
                    cf_cand_matrix = np.where(instance.data == 1, cf_cand_matrix, adj_matrix)
                elif self.direction == 1:  # Only add edges
                    cf_cand_matrix = np.where(instance.data == 0, cf_cand_matrix, adj_matrix)

                # Create counterfactual candidate
                cf_instance = GraphInstance(
                    id=instance.id,
                    label=0,  # Temporary label, will be updated
                    data=cf_cand_matrix,
                    node_features=instance.node_features
                )

                # Check if counterfactual is valid
                l_cf_cand = self.oracle.predict(cf_instance)
                if l_input_inst != l_cf_cand:
                    cf_instance.label = l_cf_cand  # Update label if class changes

                    return LocalGraphCounterfactualExplanation(
                        context=self.context,
                        dataset=self.dataset,
                        oracle=self.oracle,
                        explainer=self,
                        input_instance=instance,
                        counterfactual_instances=[cf_instance],
                    )

        # If no counterfactual is found, return the original instance
        return LocalGraphCounterfactualExplanation(
            context=self.context,
            dataset=self.dataset,
            oracle=self.oracle,
            explainer=self,
            input_instance=instance,
            counterfactual_instances=[copy.deepcopy(instance)],
        )
    

    def real_fit(self):
        pass

    
    def check_configuration(self):
        super().check_configuration()

        if not 'p' in self.local_config['parameters']:
            self.local_config['parameters']['p'] = 0.1

        if not 't' in self.local_config['parameters']:
            self.local_config['parameters']['t'] = 3

    def write(self):
        pass
      
    def read(self):
        pass
    