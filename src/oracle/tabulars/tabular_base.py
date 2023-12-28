from src.core.oracle_base import Oracle
from src.utils.cfg_utils import add_init_defaults_params, get_dflts_to_of, init_dflts_to_of, inject_dataset

class TabularOracle(Oracle):#TODO: Made it Abstract class
    def init(self):
        super().init()
        embedder_snippet = self.local_config['parameters']['embedder'] 
        if 'retrain' not in embedder_snippet['parameters']:
            embedder_snippet['parameters']['retrain'] = self.local_config['parameters'].get('retrain', False)
            
        self.embedder = self.context.factories['embedders'].get_embedder(embedder_snippet,self.dataset)

        
    def real_fit(self):
        inst_vectors = self.embedder.get_embeddings()

        if self.fold_id == -1:
            # Training with the entire dataset
            x = inst_vectors
            y = [ i.label for i in self.dataset.instances]
        else:
            # Training with an specific split
            spt = self.dataset.get_split_indices()[self.fold_id]['train']
            x = [inst_vectors[i] for i in spt]
            y = [self.dataset.get_instance(i).label for i in spt]
        self.model.fit(x, y)

    def _real_predict(self, data_instance):
        data_instance = self.embedder.get_embedding(data_instance)
        return self.model.predict(data_instance).squeeze()
        
    def _real_predict_proba(self, data_instance):
        data_instance = self.embedder.get_embedding(data_instance)
        return self.model.predict_proba(data_instance).squeeze()
    
    def check_configuration(self):
        super().check_configuration()
        emb_kls = "src.embedder.graph2vec.model.Graph2VecEmbedder"

        get_dflts_to_of (self.local_config, 'embedder', emb_kls, fold_id = self.fold_id) # Get empty or Default and pass the fold_id
        #inject_dataset(self.local_config['parameters']['embedder'], self.dataset)      # Inject the dataset to the embedder
    
    