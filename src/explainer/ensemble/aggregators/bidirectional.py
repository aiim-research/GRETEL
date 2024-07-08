import copy
import sys
import numpy as np
import random
from typing import List

from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset
from src.dataset.instances.graph import GraphInstance
from src.dataset.instances.graph import GraphInstance
from src.dataset.instances.base import DataInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.utils.utils import pad_adj_matrix
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
import src.utils.explanations.functions as exp_tools


class ExplanationBidirectionalSearch(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()
        self.logger = self.context.logger

        if 'oc_limit' not in self.local_config['parameters']:
            self.local_config['parameters']['oc_limit'] = 2000

        if 'change_batch' not in self.local_config['parameters']:
            self.local_config['parameters']['change_batch'] = 5

        if 'action_prob' not in self.local_config['parameters']:
            self.local_config['parameters']['action_prob'] = 0.5


    def init(self):
        super().init()

        self.oc_limit = self.local_config['parameters']['oc_limit']
        self.k = self.local_config['parameters']['change_batch']
        self.p = self.local_config['parameters']['action_prob']


    def real_aggregate(self, explanations: List[LocalGraphCounterfactualExplanation]) -> LocalGraphCounterfactualExplanation:
        input_inst = explanations[0].input_instance
        cf_instances = exp_tools.unpack_cf_instances(explanations)

        e_add = []
        e_rem = []
        change_edges, min_changes, change_freq_matrix = self.get_all_edge_differences(input_inst, cf_instances)
        for x, y in change_edges:
            if input_inst.data[x,y] > 0:
                e_rem.append((x,y)) # Add existing edges to the remove list
            else:
                e_add.append((x,y)) # Add non-existing edges to the add list
        
        # Try to get a first counterfactual with the greedy "Oblivious Forward Search"
        initial_cf, used_oc = self.oblivious_forward_search(instance=input_inst, 
                                                            e_add=e_add,
                                                            e_rem=e_rem,
                                                            k=self.k,
                                                            maximum_oracle_calls=self.oc_limit,
                                                            p=self.p)
        
        # If the first step was unable to find a counterfactual
        if initial_cf.label == input_inst.label:
            self.logger.info(f'No counterfactual was found for instance {input_inst.id} in the forward search')
            no_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                             dataset=self.dataset,
                                                             oracle=self.oracle,
                                                             explainer=None, # Will be added later by the ensemble
                                                             input_instance=input_inst,
                                                             counterfactual_instances=[copy.deepcopy(input_inst)])
            return no_explanation
        
        # If a counterfactual was found in the first step then try to reduce the number of changes in the second step
        changed_edges, _, _ = self.get_all_edge_differences(input_inst, [initial_cf])
        final_cf  = self.oblivious_backward_search(instance=input_inst,
                                                   cf_instance=initial_cf,
                                                   changed_edges=changed_edges,
                                                   k=self.k,
                                                   maximum_oracle_calls=self.oc_limit - used_oc)
        
        aggregated_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                                    dataset=self.dataset,
                                                                    oracle=self.oracle,
                                                                    explainer=None, # Will be added later by the ensemble
                                                                    input_instance=input_inst,
                                                                    counterfactual_instances=[final_cf])

        return aggregated_explanation

        
    def oblivious_forward_search(self, instance, e_add, e_rem, k=5, maximum_oracle_calls=2000, p=0.5):
        '''
        This method performs a random search trying to generate a counterfactual by testing 
        multiple combinations of edge modifications
        '''
        oracle_calls_count=0
        instance_lbl = self.oracle.predict(instance)
        
        # Candidate counterfactual
        cf_candidate_matrix = np.copy(instance.data)

        # randomize and remove duplicate
        random.shuffle(e_add)
        random.shuffle(e_rem)
        
        # Start the search
        while(oracle_calls_count < maximum_oracle_calls): # While the maximum number of oracle calls is not exceeded
            k_i=0
            while(k_i<k): # Made a number of changes (edge adds/removals) no more than k
                if random.random() < p:
                    if (len(e_rem) > 0):
                        # remove
                        i,j = e_rem.pop(0)

                        if instance.directed:
                            cf_candidate_matrix[i][j]=0
                        else:
                            cf_candidate_matrix[i][j]=0
                            cf_candidate_matrix[j][i]=0

                        e_add.append((i,j))
                        random.shuffle(e_add)
                        k_i+=1
                else:
                    if (len(e_add) > 0):
                        # add
                        i,j = e_add.pop(0)

                        if instance.directed:
                            cf_candidate_matrix[i][j]=1
                        else:
                            cf_candidate_matrix[i][j]=1
                            cf_candidate_matrix[j][i]=1

                        e_rem.append((i,j))
                        random.shuffle(e_rem)
                        k_i+=1

            cf_candidate_inst = GraphInstance(id=instance.id, label=0, data=cf_candidate_matrix)
            self.dataset.manipulate(cf_candidate_inst)
            cf_candidate_inst.label = self.oracle.predict(cf_candidate_inst)
            oracle_calls_count += 1 # Increase the oracle calls counter

            if cf_candidate_inst.label != instance_lbl:
                # A counterfactual was found
                return cf_candidate_inst, oracle_calls_count
        
        # return the original graph if no counterfactual was found
        return copy.deepcopy(instance), oracle_calls_count


    def oblivious_backward_search(self, instance, cf_instance, changed_edges, k=5, maximum_oracle_calls=2000):
        '''
        This method tries to reduce the size of a counterfactual instance by randomly reverting some of the changes, 
        made to the original instance, while mantaining the correctness
        '''
        gc = np.copy(cf_instance.data)
        # d = self.distance_metric.distance(instance.data,gc)
        random.shuffle(changed_edges)
        oracle_calls_count=0

        # while(oracle_calls_count < maximum_oracle_calls and len(changed_edges) > 0 and d > 1):
        while(oracle_calls_count < maximum_oracle_calls and len(changed_edges) > 0):
            # Create a working copy of the CF adjacency matrix to revert some changes
            gci = np.copy(gc)

            # Select some edges to revert
            ki = min(k,len(changed_edges))
            edges_i = [changed_edges.pop(0) for i in range(ki)]
            
            # Revert the changes on the selected edges
            for i,j in edges_i:
                gci[i][j] = abs(1 - gci[i][j])
                gci[j][i] = abs(1 - gci[j][i])

            reduced_cf_inst = GraphInstance(id=instance.id, label=0, data=gci)
            self.dataset.manipulate(reduced_cf_inst)
            reduced_cf_inst.label = self.oracle.predict(reduced_cf_inst)
            oracle_calls_count += 1

            if reduced_cf_inst.label != instance.label: # If the reduced instance is still a counterfactual
                self.logger.info(f'The counterfactual for {instance.id} was reduced')
                gc = np.copy(gci)
                k+=1
            else: # If the reduced instance is no longer a counterfactual
                if k>1:
                    # Reduce the amount of changes to perform in the next iteration
                    k-=1
                    changed_edges = changed_edges + edges_i

        result_cf = GraphInstance(id=instance.id, label=0, data=gc, directed=instance.directed)
        self.dataset.manipulate(result_cf)
        result_cf.label = self.oracle.predict(reduced_cf_inst)

        return result_cf
    