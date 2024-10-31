import copy
import sys
import numpy as np
import random
from typing import List

from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.comparison import get_all_edge_differences, get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric


class Random(ExplanationMinimizer):

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


    def minimize(self, explaination: LocalGraphCounterfactualExplanation) -> DataInstance:
        instance = explaination.input_instance
        input_label = self.oracle.predict(instance)

        min_ctf = explaination.counterfactual_instances[0]
        min_ctf_dist = self.distance_metric.evaluate(instance, min_ctf, self.oracle)
        for ctf_candidate in explaination.counterfactual_instances:
            candidate_label = self.oracle.predict(ctf_candidate)

            if input_label != candidate_label:
                ctf_distance = self.distance_metric.evaluate(instance, ctf_candidate, self.oracle)
                
                if ctf_distance < min_ctf_dist:
                    min_ctf_dist = ctf_distance
                    min_ctf = ctf_candidate
                    
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
        reduction_success = False
        gc = np.copy(cf_instance.data)
        # d = self.distance_metric.distance(instance.data,gc)
        random.shuffle(changed_edges)
        oracle_calls_count=0

        # while(oracle_calls_count < maximum_oracle_calls and len(changed_edges) > 0 and d > 1):
        while(oracle_calls_count < maximum_oracle_calls and len(changed_edges) > 0):
            # Create a working copy of the CF adjacency matrix to revert some changes
            gci = np.copy(gc)

            # Select some edges to revert
            ki = min(k, len(changed_edges))
            edges_i = [changed_edges.pop(0) for i in range(ki)]
            
            # Revert the changes on the selected edges
            for i,j in edges_i:
                gci[i][j] = abs(1 - gci[i][j])

                # If the graph is undirected we need to undo the symmetrical edge too 
                if not instance.is_directed:
                    gci[j][i] = abs(1 - gci[j][i])

            reduced_cf_inst = GraphInstance(id=instance.id, label=0, data=gci, directed=instance.directed)
            self.dataset.manipulate(reduced_cf_inst)
            reduced_cf_inst.label = self.oracle.predict(reduced_cf_inst)
            oracle_calls_count += 1

            if reduced_cf_inst.label != instance.label: # If the reduced instance is still a counterfactual
                reduction_success = True
                gc = np.copy(gci)
                k+=1
            else: # If the reduced instance is no longer a counterfactual
                if k>1:
                    # Reduce the amount of changes to perform in the next iteration
                    k-=1
                    changed_edges = changed_edges + edges_i
                else:
                    # \if the size of the batch of changes to perform is already 1
                    # Change the edge to pick
                    changed_edges = changed_edges + edges_i

        result_cf = GraphInstance(id=instance.id, label=0, data=gc, directed=instance.directed)
        self.dataset.manipulate(result_cf)
        result_cf.label = self.oracle.predict(result_cf)

        if reduction_success:
            self.logger.info(f'The counterfactual for {str(instance.id)} was reduced')
        else:
            self.logger.info(f'The counterfactual for {str(instance.id)} was not reduced')

        return result_cf
    
    
    def write(self):
        pass

    def read(self):
        pass