import copy
import sys
import numpy as np
import random
from typing import List

from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.manipulation.methods import average_smoothing, feature_aggregation, heat_kernel_diffusion, laplacian_regularization, random_walk_diffusion, weighted_smoothing
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.comparison import get_all_edge_differences, get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric


class RandomPlus(ExplanationMinimizer):

    def check_configuration(self):
        super().check_configuration()

        if 'max_oc' not in self.local_config['parameters']:
            self.local_config['parameters']['max_oc'] = 2000

        if 'changes_batch_size' not in self.local_config['parameters']:
            self.local_config['parameters']['changes_batch_size'] = 5

    
    def init(self):
        super().init()
        self.distance_metric = GraphEditDistanceMetric()  
        self.max_oc = self.local_config['parameters']['max_oc']
        self.changes_batch_size = self.local_config['parameters']['changes_batch_size']
        
        self.methods = [
            lambda data, features: average_smoothing(data, features, iterations=1),
            lambda data, features: weighted_smoothing(data, features, iterations=1),
            lambda data, features: laplacian_regularization(data, features, lambda_reg=0.01, iterations=1),
            lambda data, features: feature_aggregation(data, features, alpha=0.5, iterations=1),
            lambda data, features: heat_kernel_diffusion(data, features, t=0.5),
            lambda data, features: random_walk_diffusion(data, features, steps=1)
        ]


    def minimize(self, explaination: LocalGraphCounterfactualExplanation) -> DataInstance:
        instance = explaination.input_instance
        input_label = self.oracle.predict(instance)

        min_ctf = explaination.counterfactual_instances[0]
        print("random instance -> " + str(self.oracle.predict(instance)))
        print("random min_ctf -> " + str(self.oracle.predict(min_ctf)))
        # min_ctf_dist = self.distance_metric.evaluate(instance, min_ctf, self.oracle)
        # for ctf_candidate in explaination.counterfactual_instances:
        #     candidate_label = self.oracle.predict(ctf_candidate)

        #     if input_label != candidate_label:
        #         ctf_distance = self.distance_metric.evaluate(instance, ctf_candidate, self.oracle)
                
        #         if ctf_distance < min_ctf_dist:
        #             min_ctf_dist = ctf_distance
        #             min_ctf = ctf_candidate
                    
        cf_instance = min_ctf
        # Get the changes between the original graph and the initial counterfactual
        changed_edges, _, _ = get_all_edge_differences(instance, [cf_instance])

        # apply the backward search to minimize the counterfactual
        minimal_cf = self.oblivious_backward_search(instance, 
                                                    cf_instance, 
                                                    changed_edges, 
                                                    k=self.changes_batch_size,
                                                    maximum_oracle_calls=self.max_oc)

        # Return the minimal counterfactual
        return minimal_cf


    def oblivious_backward_search(self, instance, cf_instance, changed_edges, k=5, maximum_oracle_calls=2000):
        '''
        This method tries to reduce the size of a counterfactual instance by randomly reverting some of the changes, 
        made to the original instance, while mantaining the correctness
        '''
        initial_changed_edges = len(changed_edges)
        reduction_success = False
        gc = np.copy(cf_instance.data)
        features = np.copy(cf_instance.graph_features)
        # d = self.distance_metric.distance(instance.data,gc)
        random.shuffle(changed_edges)
        oracle_calls_count=0

        # while(oracle_calls_count < maximum_oracle_calls and len(changed_edges) > 0 and d > 1):
        while(oracle_calls_count < maximum_oracle_calls and len(changed_edges) > 0):
            # Create a working copy of the CF adjacency matrix to revert some changes
            gci = np.copy(gc)

            # Select some edges to revert
            ki = min(k, len(changed_edges))
            edges_i = [changed_edges.pop(0) for _ in range(ki)]
            
            # Revert the changes on the selected edges
            for i,j in edges_i:
                gci[i][j] = abs(1 - gci[i][j])

                # If the graph is undirected we need to undo the symmetrical edge too 
                if not instance.is_directed:
                    gci[j][i] = abs(1 - gci[j][i])

            found = False
            
            for method in self.methods:
                node_features = method(gci, instance.node_features)
                
                if node_features is None:
                    print("node_features is None")
                elif isinstance(node_features, list) and len(node_features) == 0:
                    print("node_features is an empty list")
            
                reduced_cf_inst = GraphInstance(id=instance.id, 
                                                label=0, 
                                                data=gci, 
                                                directed=instance.directed, 
                                                node_features=node_features)
                reduced_cf_inst.label = self.oracle.predict(reduced_cf_inst)
                oracle_calls_count += 1

                instance_label = self.oracle.predict(instance)
            
                if reduced_cf_inst.label != instance_label: # If the reduced instance is still a counterfactual
                    found = True
                    break
            
            if found:
                reduction_success = True
                gc = np.copy(gci)
                features = np.copy(node_features)
                k+=1
                final_changed_edges = len(changed_edges)
            else: # If the reduced instance is no longer a counterfactual
                if k>1:
                    # Reduce the amount of changes to perform in the next iteration
                    k-=1
                    changed_edges = changed_edges + edges_i
                else:
                    # \if the size of the batch of changes to perform is already 1
                    # Change the edge to pick
                    changed_edges = changed_edges + edges_i
        
        result_cf = GraphInstance(id=instance.id, 
                                  label=0, 
                                  data=gc, 
                                  directed=instance.directed, 
                                  node_features=features)
        
        result_cf.label = self.oracle.predict(result_cf)
        
        # print(self.oracle.predict(instance))

        if reduction_success:
            self.logger.info(f'The counterfactual for {str(instance.id)} was reduced ({str(initial_changed_edges)} -> {str(final_changed_edges)})')
        else:
            self.logger.info(f'The counterfactual for {str(instance.id)} was not reduced')

        return result_cf
    
    
    def write(self):
        pass

    def read(self):
        pass