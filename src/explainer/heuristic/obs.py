import random
import itertools
import numpy as np
import copy

from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import init_dflts_to_of
from src.dataset.instances.graph import GraphInstance


class ObliviousBidirectionalSearchExplainer(Explainer):
    """
    An implementation of the Counterfactual Explainer proposed in the paper "Abrate, Carlo, and Francesco Bonchi. 
    "Counterfactual Graphs for Explainable Classification of Brain Networks." 
    Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021."
    """

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.explainer.heuristic.obs_dist.ObliviousBidirectionalDistance'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)
        


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        
    def explain(self, instance):

        # Get the label of the original instance
        l_inst = self.oracle.predict(instance)
        # Get the label of the counterfactual (just for binary classification problems)
        l_counterfactual = 1 - l_inst

        instance_matrix = instance.data
        # Try to get a first counterfactual with the greedy "Oblivious Forward Search"
        ged, counterfactual, oracle_calls = self.oblivious_forward_search(instance, 
                                                                          instance_matrix,
                                                                          l_counterfactual)

        final_counterfactual, edit_distance, oracle_calls, info = self.oblivious_backward_search(instance, 
                                                                                        instance_matrix, 
                                                                                        counterfactual, 
                                                                                        l_counterfactual)

        # Converting the final counterfactual into a DataInstance
        result = GraphInstance(id=instance.id, 
                               label=0, 
                               data=final_counterfactual, 
                               node_features=instance.node_features)

        return result



    # 3rd party adapted /////////////////////////////////////////////////////////////////////////////
        
    def oblivious_forward_search(self, instance, g_o, y_bar, k=5, lambda_g=2000, p_0=0.5):
        '''
        Oblivious Forward Search as implemented by Abrate and Bonchi
        '''
        dim = len(g_o)
        l=0
        
        # Candidate counterfactual
        g_c = np.copy(g_o)
        r = abs(1-y_bar)

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
        while(l<lambda_g): # While the maximum number of oracle calls is not exceeded
            ki=0
            while(ki<k): # Made a number of changes (edge adds/removals) no more than k
                if self._bernoulli(p_0):
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

            inst = GraphInstance(id=instance.id, 
                                 label=0, 
                                 data=g_c,
                                 node_features=instance.node_features)

            r = self.oracle.predict(inst)
            l += 1 # Increase the oracle calls counter

            if r==y_bar:
                #print('- A counterfactual is found!')
                d = self.distance_metric.distance(g_o,g_c)
                return d,g_c,l
            if len(g_rem)<1:
                print('no more remove')
        
        # m-comment: return the original graph if no counterfactual was found
        return 0, g_o, l


    def oblivious_backward_search(self, instance, g ,gc1 ,y_bar ,k=5 , l_max=2000):
        '''
        '''
        info = []

        gc = np.copy(gc1)
        edges = self._get_change_list(g,gc)
        d = self.distance_metric.distance(g,gc)
        random.shuffle(edges)
        li=0
        while(li<l_max and len(edges)>0 and d>1):
            ki = min(k,len(edges))
            gci = np.copy(gc)
            edges_i = [edges.pop(0) for i in range(ki)]
            for i,j in edges_i:
                if gci[i][j]>0.5:
                    gci[i][j] = 0
                    gci[j][i] = 0
                else:
                    gci[i][j] = 1
                    gci[j][i] = 1

            inst = GraphInstance(id=instance.id, 
                                 label=0, 
                                 data=gci,
                                 node_features=instance.node_features)
            
            r = self.oracle.predict(inst)
            li += 1

            if r==y_bar:
                gc = np.copy(gci)
                d = self.distance_metric.distance(g,gc)
                info.append((r,d,li,ki))
                k+=1
            else:
                d = self.distance_metric.distance(g,gc)
                info.append((r,d,li,ki))

                if k>1:
                    k-=1
                    edges = edges + edges_i

        return gc, self.distance_metric.distance(g,gc), li, info
    

    # Ancillary functions//////////////////////////////////////////////////////////////////////////////

   
    def _bernoulli(self, p):
        ''' p is the probability of removing an edge.
        '''
        return True if random.random() < p else False


    def _get_change_list(self, g1,g2):
        edges = []
        g_diff = abs(g1-g2)
        dim_g = len(g1)
        for i in range(dim_g):
            for j in range(i,dim_g):
                if g_diff[i][j]==1:
                    edges.append((i,j))
        return edges

