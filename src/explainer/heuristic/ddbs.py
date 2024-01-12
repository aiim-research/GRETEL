import random
import itertools
import numpy as np
import copy


from src.dataset.instances.graph import GraphInstance
from src.dataset.dataset_base import Dataset
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.core.trainable_base import Trainable

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset


class DataDrivenBidirectionalSearchExplainer(Explainer):
    """
    An implementation of the Counterfactual Explainer proposed in the paper "Abrate, Carlo, and Francesco Bonchi. 
    "Counterfactual Graphs for Explainable Classification of Brain Networks." 
    Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021."
    """

    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)
       

    def explain(self, instance):
        """
        Uses a combination of Data-driven Forward Search with Data-driven Backward Search as described in "Abrate, Carlo, and Francesco Bonchi. 
        "Counterfactual Graphs for Explainable Classification of Brain Networks." Proceedings of the 27th ACM SIGKDD Conference on Knowledge
        Discovery & Data Mining. 2021."
        """
        edges_prob = self.get_edge_probabilities(self.dataset, self.oracle)

        instance_label = self.oracle.predict(instance)
        counterfactual_label = 1 - instance_label  # this is only true for binary classification problems
        original_graph = instance.data

        ged, counterfactual, oracle_calls = self.DFS(instance, original_graph, counterfactual_label, edges_prob)

        final_counterfactual, ged, oracle_calls, info = self.bb_prob_2(instance, original_graph, counterfactual, counterfactual_label, edges_prob)

        result = GraphInstance(id=instance.id, 
                                label=0, 
                                data=final_counterfactual,
                                node_features=instance.node_features)

        return result
    

    # Ancillary functions/////////////////////////////////////////////////////////////////////////////

    def get_edge_probabilities(self, dataset, oracle):
        """
        This method is meant to be used with DFS
        """
        # Nodes frequency
        #create the two matices as the count of the frequency of each edge to be in a graph of the dataset
        dim_g = len(dataset.instances[0].data)

        g_0 = np.zeros((dim_g,dim_g))
        g_1 = np.zeros((dim_g,dim_g))

        for inst in dataset.instances:
            g = np.copy(inst.data)

            y_hat = oracle.predict(inst)
            if y_hat==0:
                g_0 = np.add(g_0,g)
            else:
                g_1 = np.add(g_1,g)

        g_01 = g_0-g_1
        g_10 = g_1-g_0

        min_01 = g_01.min()
        max_01 = g_01.max()
        g01 = np.ones((dim_g,dim_g))+(g_01-min_01)/(max_01-min_01)

        min_10 = g_10.min()
        max_10 = g_10.max()
        g10 = np.ones((dim_g,dim_g))+(g_10-min_10)/(max_10-min_10)

        prob_initial = {0:g01/g01.sum(),1:g10/g10.sum()}

        g00 = np.ones((dim_g,dim_g))
        uniform_initial_prop = {0:g00/g00.sum(),1:g00/g00.sum()}

        return prob_initial
    

    def DFS_select(self, g,edges, y_bar, ki, edges_prob, p_0=0.5):
        '''
        '''
        edges_prob_rem = np.array([])
        edges_prob_add = np.array([])
        edges_add = []
        edges_rem = []
        e = []
        dim = len(g)
        for i in range(dim):
            for j in range(dim):
                if (i,j) not in edges:
                    if g[i][j]>0:
                        edges_prob_rem = np.append(edges_prob_rem,edges_prob[1-y_bar][i][j])
                        edges_rem.append((i,j))
                    else:
                        edges_prob_add = np.append(edges_prob_add,edges_prob[y_bar][i][j])
                        edges_add.append((i,j))
        edges_prob_add = edges_prob_add/edges_prob_add.sum()
        edges_prob_rem = edges_prob_rem/edges_prob_rem.sum()
        #print('-- ',len(edges_rem),len(edges_add),len(edges))
        edges_i = []
        kii=0
        while(kii<ki):
            kii+=1
            if self._bernoulli(p_0) and len(edges_add)>0:
                #add
                n = np.random.choice(range(len(edges_add)), size=1, p=edges_prob_add)[0]
                i,j = edges_add[n]
                g[i][j]=1
                g[j][i]=1
            elif len(edges_rem)>0:
                #remove
                n = np.random.choice(range(len(edges_rem)), size=1, p=edges_prob_rem)[0]
                i,j = edges_rem[n]
                g[i][j]=0
                g[j][i]=0
            edges.append((i,j))
        return g,edges
    

    def DFS(self, instance, g, y_bar, edges_prob, k=10, l_max=2000):
        '''
        '''
        info = []
        gc = np.copy(g)
        d = self._edit_distance(g,gc)
        li=0
        edges=[]
        while(li<l_max):
            gc,edges = self.DFS_select(gc,edges,y_bar,k,edges_prob,)

            inst = GraphInstance(id=instance.id, 
                                 label=0, 
                                 data=gc,
                                 node_features=instance.node_features)
            
            # inst = copy.deepcopy(instance)
            # inst.data = gc
            # inst._nx_repr = None
            r = self.oracle.predict(inst)
            
            li += 1
            if r==y_bar:
                d = self._edit_distance(g, gc)
                return d, gc, li
                
        return 0, gc, li
    

    def bb_prob_2(self, instance, g , gc1, y_bar, edges_prob, k=5, l_max=2000):
        '''
        '''
        info = []
        gc = np.copy(gc1)
        edges = self._get_change_list(g, gc)
        d = self._edit_distance(g,gc)
        li=0

        while(li<l_max and len(edges)>0 and d>1):
            ki = min(k,len(edges))
            gci,new_edges = self.get_prob_edges(gc,edges,y_bar,k,edges_prob)

            # Modified code ////////////////////////////////////////////////
            # r = oracle(gci)
            inst = GraphInstance(id=instance.id, 
                                 label=0, 
                                 data=gci,
                                 node_features=instance.node_features)
            
            # inst = copy.deepcopy(instance)
            # inst.data = gci
            # inst._nx_repr = None
            r = self.oracle.predict(inst)
            # //////////////////////////////////////////////////////////////
            li += 1

            if r==y_bar and self._edit_distance(gci,gc)>0:
                gc = np.copy(gci)
                d = self._edit_distance(g,gc)
                edges = self._get_change_list(g,gc)
                #print('ok --> ',r,d,li,k)
                info.append((r,d,li,ki))
                k+=1
            else:
                #print('no --> ',r,d,li,k)
                d = self._edit_distance(g,gc)
                info.append((r,d,li,ki))
                if k>1:
                    k-=1
                else:
                    edges.remove(new_edges[0])

        return gc, self._edit_distance(g, gc), li, info
    

    def get_prob_edges(self, gc, edges, y_bar, ki, edges_prob, p_0=0.5):
        '''
        '''
        gci = np.copy(gc)
        edges_prob_rem = np.array([])
        edges_prob_add = np.array([])
        edges_add = []
        edges_rem = []
        e = []

        for e in edges:
            i,j = e
            if gc[i][j]>0:
                edges_prob_rem = np.append(edges_prob_rem,edges_prob[1-y_bar][i][j])
                edges_rem.append((i,j))
            else:
                edges_prob_add = np.append(edges_prob_add,edges_prob[y_bar][i][j])
                edges_add.append((i,j))
        edges_prob_add = edges_prob_add/edges_prob_add.sum()
        edges_prob_rem = edges_prob_rem/edges_prob_rem.sum()
        #print('-- ',len(edges_rem),len(edges_add),len(edges))
        edges_i = []
        kii=0
        while(kii<ki):
            kii+=1
            if self._bernoulli(p_0) and len(edges_add)>0:
                #add
                n = np.random.choice(range(len(edges_add)), size=1, p=edges_prob_add)[0]
                i,j = edges_add[n]
                gci[i][j]=1
                gci[j][i]=1
            elif len(edges_rem)>0:
                #remove
                n = np.random.choice(range(len(edges_rem)), size=1, p=edges_prob_rem)[0]
                i,j = edges_rem[n]
                gci[i][j]=0
                gci[j][i]=0
            edges_i.append((i,j))
        new_edges = edges_add+edges_rem
        return gci,new_edges
    

    def _edit_distance(self, g_1,g_2):
        '''Returns a particular version of Graph Edit Distance when only edge changes are considered and
        the graphs are already matched
        '''
        return self._tot_edges(abs(g_1-g_2))


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


    def _tot_edges(self, g):
        '''Returns the total number of edges for undirected graphs
        '''
        return sum([sum(el) for el in g])/2
