from src.embedder.graph2vec.estimator import Graph2Vec
from src.core.embedder_base import Embedder
from src.utils.cfg_utils import default_cfg, empty_cfg_for, init_dflts_to_of

class Graph2VecEmbedder(Embedder):
        
    def init(self):
        self.model = Graph2Vec(**self.local_config['parameters']['model']['parameters'])
            
    def real_fit(self):
        #TODO: Currently the fold are not managed
        # Copies of the graphs are provided because the _check_graphs function modifies the edges of the passed graph
        graphs = [i.get_nx() for i in self.dataset.get_data()]
        self.model.fit(graphs)

    def get_embeddings(self):
        return self.model.get_embedding()

    def get_embedding(self, instance):
        # A Copy of the graph is provided because the _check_graphs function modifies the edges of the passed graph
        return self.model.infer(instance.get_nx()).reshape(1, -1)    
    
    @default_cfg
    def grtl_default(kls, fold_id):
        #TODO: add the fold_id  logic so do not inject it for the moment
        self_kls = "src.embedder.graph2vec.model.Graph2VecEmbedder"
        self_cfg = empty_cfg_for(self_kls) #Init an empty cfg for this class

        sub_kls ='src.embedder.graph2vec.estimator.Graph2Vec'        
        init_dflts_to_of(self_cfg, 'model', sub_kls) #Init the default accordingly to the nested Estimator
        return self_cfg