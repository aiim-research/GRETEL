import random
import itertools
import numpy as np
import copy

from src.core.explainer_base import Explainer
from src.core.factory_base import get_class, get_instance_kvargs
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import  inject_dataset, inject_oracle, init_dflts_to_of
import src.utils.explanations.functions as exp_tools
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.instances.graph import GraphInstance


class OFS(Explainer):
    "This explainer implements the ObliviousForwardSearch proposed by Abrate and Bonchi as part of OBS"

    def check_configuration(self):
        super().check_configuration()

        self.local_config['parameters']['max_oc'] = self.local_config['parameters'].get('max_oc', 2000)

        self.local_config['parameters']['changes_batch_size'] = self.local_config['parameters'].get('changes_batch_size', 5)

        self.local_config['parameters']['p'] = self.local_config['parameters'].get('p', 0.5)

        dst_metric='src.explainer.heuristic.obs_dist.ObliviousBidirectionalDistance'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()
        self.logger = self.context.logger
        self.max_oc = self.local_config['parameters']['max_oc']
        self.changes_batch_size = self.local_config['parameters']['changes_batch_size']
        self.p = self.local_config['parameters']['p']

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        

    def explain(self, instance):
        # Searching for a counterfactual instance
        cf_inst = self.oblivious_forward_search(instance)

        # Wrapping the counterfactual in an explanation
        exp = LocalGraphCounterfactualExplanation(context=self.context,
                                                  dataset=self.dataset,
                                                  oracle=self.oracle,
                                                  explainer=self,
                                                  input_instance=instance,
                                                  counterfactual_instances=[cf_inst])

        # Returning the explanation
        return exp
    

    def oblivious_forward_search(self, instance):
        '''
        Oblivious Forward Search as implemented by Abrate and Bonchi
        '''
        dim = len(instance.data)
        l=0
        
        # Candidate counterfactual
        g_c = np.copy(instance.data)
        r = instance.label

        # Create add and remove sets of edges
        g_add = []
        g_rem = []
        for i in range(dim):
            for j in range(i,dim):
                if i!=j:
                    if g_c[i][j]>0.5: # Add existing edges to the remove list
                        g_rem.append((i,j))
                    else:
                        g_add.append((i,j)) # Add non-exisitng edges to the add list

        # randomize and remove duplicate
        random.shuffle(g_add)
        random.shuffle(g_rem)
        
        # Start the search
        while(l < self.max_oc): # While the maximum number of oracle calls is not exceeded
            ki=0
            while(ki < self.changes_batch_size): # Made a number of changes (edge adds/removals) no more than changes_batch_size
                if random.random() < self.p:
                    if (len(g_rem) > 0):
                        # remove
                        i,j = g_rem.pop(0)
                        g_c[i][j]=0
                        g_c[j][i]=0
                        g_add.append((i,j))
                        #random.shuffle(g_add)
                        ki+=1
                else:
                    if (len(g_add) > 0):
                        # add
                        i,j = g_add.pop(0)
                        g_c[i][j]=1
                        g_c[j][i]=1
                        g_rem.append((i,j))
                        #random.shuffle(g_rem)
                        ki+=1
            ki=0

            current_inst = GraphInstance(id=instance.id, 
                                 label=0, 
                                 data=g_c,
                                 node_features=instance.node_features)

            r = self.oracle.predict(current_inst)
            l += 1 # Increase the oracle calls counter

            if r != instance.label:
                # A counterfactual was found
                d = self.distance_metric.distance(instance.data, g_c)
                return current_inst
        
        # Return the original graph if no counterfactual was found
        return copy.deepcopy(instance)
    