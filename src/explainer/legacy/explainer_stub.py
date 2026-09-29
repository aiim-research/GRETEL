import copy
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
import networkx as nx

from src.dataset.instances.graph import GraphInstance
from src.dataset.manipulators.centralities import NodeCentrality
from src.dataset.manipulators.weights import EdgeWeights


class ExplainerStub(Trainable, Explainer):
    def init(self):
      super().init()  
      self.weights_man = EdgeWeights(self.context,{},copy.deepcopy(self.dataset))
      self.centr_man = NodeCentrality(self.context,{},copy.deepcopy(self.dataset))    
     
    
    def real_fit(self):
        for g in self.dataset.instances:
           self.oracle.predict(g)

    def write(self):
        pass
    
    #TODO: Idea: having Oracle-Acc-based measures of any kind (different from mes/acc):
    # e.g.  'Oracle_Calls':     [1, 1, 1, 1, 1, 1, 3, 12, 17, 16, 2, 5] -> 9.16 vs 5.08
    #       'Correctness':      [0, 0, 0, 0, 0, 0, 1,  1,  1,  1, 1, 1] -> 1 vs 0.5
    #       'Oracle_Accuracy':  [0, 0, 0, 0, 0, 0, 1,  1,  1,  1, 1, 1] -> 0.5 
    # Impl.   cacc = o_calls * o_acc; 
    #        mes_acc = np.mean(cacc[cacc != 0])

    def explain(self, instance):
        for _ in range(100):            
            cg = nx.connected_watts_strogatz_graph(instance.num_nodes, int(instance.num_nodes*0.1), 0.2, tries=100)
            cg = GraphInstance(0,  1 - instance.label, nx.to_numpy_array(cg))
            self.weights_man._process_instance(cg)
            self.centr_man._process_instance(cg)
            if(instance.label+self.oracle.predict(cg)==1):
                break
        
        return cg