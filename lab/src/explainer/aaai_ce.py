from src.core.explainer_base import Explainer
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.utils.cfg_utils import retake_oracle
from copy import deepcopy
import numpy as np

class OvershootCounterfactualExplainer(Explainer):
    
    def init(self):
        self.oracle = retake_oracle(self.local_config)
        
        local_params = self.local_config['parameters']
        self.overshoot_factor = local_params['overshoot_factor']
        self.step_size = local_params['step_size']
        self.max_iterations = local_params['max_iterations']
        self.perturbation = local_params['perturbation']
        self.threshold = local_params['threshold']
        
        assert ((isinstance(self.max_iterations, float) or isinstance(self.max_iterations, int)) and self.max_iterations >= 1)
        assert ((isinstance(self.step_size, float) or isinstance(self.step_size, int)) and self.step_size > 0)
        assert ((isinstance(self.perturbation, float) or isinstance(self.perturbation, int)) and self.perturbation > 0)
        assert ((isinstance(self.threshold, float) or isinstance(self.perturbation, int)) and self.threshold >= 0)

    def explain(self, instance):
        self.original_pred = self.oracle.predict(instance)
        return self.__shorten_distance(instance, self.__overshoot(instance))
     
    def __overshoot(self, instance: DataInstance) -> DataInstance:
        self.context.logger.info(f'Instance {instance.id} predicted as {self.original_pred}')
        
        pos, neg = self.__overshoot_matrix(instance.data)
                
        cf_candidates = [
            GraphInstance(id=-1, label='dummy', data=pos), 
            GraphInstance(id=-2, label='dummy', data=neg)
        ]
        
        for cf in cf_candidates:
            instance._dataset.manipulate(cf)

        for _ in range(self.max_iterations):
            predictions = [self.oracle.predict(inst) == self.original_pred for inst in cf_candidates]
            # if all predictions are equal, then I need to overshoot more
            if np.all(predictions):
                new_candidates = []
                for cf_inst in cf_candidates:
                    pos, neg = self.__overshoot_matrix(cf_inst.data)
                    pos_instance = GraphInstance(id=-1, label='dummy', data=pos)
                    neg_instance = GraphInstance(id=-2, label='dummy', data=neg)
                    # calculate the features according to the dataset manipulators
                    instance._dataset.manipulate(pos_instance)
                    instance._dataset.manipulate(neg_instance)
                    # append these candidates
                    new_candidates += [pos_instance, neg_instance]
                if len(new_candidates):
                    cf_candidates.clear()
                    cf_candidates = deepcopy(new_candidates)
            else:
                cf_inst = cf_candidates[predictions.index(False)]
                self.context.logger.info(f'For instance {instance.id} I found a counterfactual with cls = {self.oracle.predict(cf_inst)}.')
                return cf_inst
            
        return instance
    
    def __shorten_distance(self, instance: DataInstance, overshoot_instance: DataInstance):
        for _ in range(self.max_iterations):
            # I save the deepcopy because I need to return immediately
            # before I cross the decision boundary
            data = self.__pull_matrix(overshoot_instance.data, instance.data)
            retr_instance = GraphInstance(id=-1, label='dummy', data=data)
            instance._dataset.manipulate(retr_instance)
            # until the class is different from the original one, then I can pull towards
            # the decision boundary. I return immediately after crossing it to the original
            # class vector space
            if self.oracle.predict(retr_instance) != self.original_pred:
                return retr_instance
        self.context.logger.info(f'Overshot data pulled --> cls = {self.oracle.predict(overshoot_instance)}')
        return overshoot_instance
    
    def __pull_matrix(self, A, B):
        res = deepcopy(A)
        # Find the indices where A and B differ
        diff_indices = np.where(res != B)
        if diff_indices[0].size > 0:
            # Flip the bit at the first differing index
            first_diff_index = diff_indices[0][0], diff_indices[1][0]
            res[first_diff_index] = B[first_diff_index]
        return res
        
    def __overshoot_matrix(self, matrix):        
        return self.__add_edge(matrix), self.__remove_edge(matrix)
    
    def __add_edge(self, matrix):
        non_existing_edges = np.argwhere(matrix == 0)
        i, j = non_existing_edges[np.random.choice(len(non_existing_edges))]
        perturbed_matrix = matrix.copy()
        perturbed_matrix[i, j] = 1
        return perturbed_matrix
    
    def __remove_edge(self, matrix):
        # Generate a random value
        existing_edges = np.argwhere(matrix == 1)
        i, j = existing_edges[np.random.choice(len(existing_edges))]
        # Remove an edge with the specified probability
        perturbed_matrix = matrix.copy()
        perturbed_matrix[i, j] = 0
        return perturbed_matrix
    
    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        
        if 'overshoot_factor' not in local_config['parameters']:
            local_config['parameters']['overshoot_factor'] = 2
        
        if 'step_size' not in local_config['parameters']:
            local_config['parameters']['step_size'] = 0.2
            
        if 'max_iterations' not in local_config['parameters']:
            local_config['parameters']['max_iterations'] = 10
            
        if 'perturbation' not in local_config['parameters']:
            local_config['parameters']['perturbation'] = 1
            
        if 'threshold' not in local_config['parameters']:
            local_config['parameters']['threshold'] = 0.3
               
        self.fold_id = self.local_config['parameters'].get('fold_id',-1)


                