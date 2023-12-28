import random
import itertools
import numpy as np
import copy


from src.core.explainer_base import Explainer


class IRandExplainer(Explainer):
    """iRand stands for Iterative Random Explainer, the logic of the explainers is to 
    """
            
    def init(self):
        super().init()

        self.perturbation_percentage = self.local_config['parameters']['p']
        self.tries = self.local_config['parameters']['t']
        self.fold_id = -1
        
        
    def explain(self, instance):
        l_input_inst = self.oracle.predict(instance)

        nodes = instance.data.shape[0]

        # all edges (direct graph)
        all_edges = list(itertools.product(list(range(nodes)), repeat=2))
        # filter for only undirected edges
        new_edges = list()
        for edge in all_edges:
            if ((edge[1], edge[0]) not in new_edges) and edge[0] != edge[1]:
                new_edges.append(list(edge))
        new_edges = np.array(new_edges)
        
        # Calculate the maximun percent of edges to modify
        k = int(len(new_edges) * self.perturbation_percentage)

        # Creating the instance to return (initially a copy of the original instance)
        result = copy.deepcopy(instance)

        # increase the number of random modifications
        for i in range(1, k):
            # how many attempts at a current modification level
            for j in range(0, self.tries):
                cf_cand_matrix = np.copy(instance.data)
                # sample according to perturbation_percentage
                sample_index = np.random.choice(list(range(len(new_edges))), size=i)
                sampled_edges = new_edges[sample_index]

                # switch on/off the sampled edges
                cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]] = 1 - cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]]
                cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]] = 1 - cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]]
            
                # build the counterfactaul candidates instance 
                result.data = cf_cand_matrix
                # TODO this resetting of the nx representation it is not very robust
                result._nx_repr = None

                # if a counterfactual was found return that
                l_cf_cand = self.oracle.predict(result)
                if l_input_inst != l_cf_cand:
                    result.label = l_cf_cand
                    return result
                else:
                    result.data = instance.data
        
        # If no counterfactual was found return the original instance by convention
        return instance
    

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
    