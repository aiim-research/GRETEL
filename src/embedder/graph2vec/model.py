from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from src.core.embedder_base import Embedder
from src.embedder.graph2vec.graph2vec_model import WeisfeilerLehmanMachine
from src.n_dataset.instances.graph import GraphInstance
import numpy as np

class Graph2VecEmbedder(Embedder):
    def init(self):
        self.wl_iterations = self.local_config['parameters']['wl_iterations']
        self.dimensions = self.local_config['parameters']['dimensions']
        self.min_count = self.local_config['parameters']['min_count']
        self.down_sampling = self.local_config['parameters']['down_sampling']
        self.workers = self.local_config['parameters']['workers']
        self.epochs = self.local_config['parameters']['epochs']
        self.learning_rate = self.local_config['parameters']['learning_rate']
        self.seed = self.local_config['parameters']['seed']
        self.selected_feature = self.local_config['parameters']['selected_feature']
        self.embedding = None

    # todo support more than one feature
    def get_embeddings(self):
        return np.array(self.embedding)

    def get_embedding(self, instance: GraphInstance):
        features = self._get_instace_features(instance)
        document = WeisfeilerLehmanMachine(instance, features, self.wl_iterations)
        document = TaggedDocument(words=document.extracted_features, tags=[str(0)])
        self.model.random.seed(self.seed) # Force the seed to eliminate randomness
        return np.array(self.model.infer_vector(document.words)).reshape(1, -1)

    def real_fit(self):
        documents = [
            WeisfeilerLehmanMachine(
                graph, self._get_instace_features(graph), self.wl_iterations
            )
            for graph in self.dataset.instances
        ]
        documents = [
            TaggedDocument(words=doc.extracted_features, tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )
        self.model.random.seed(self.seed) # Force the seed to eliminate randomness
        self.embedding = [self.model.docvecs[str(i)] for i, _ in enumerate(documents)]

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['wl_iterations'] =  self.local_config['parameters'].get('wl_iterations', 32)
        self.local_config['parameters']['dimensions'] =  self.local_config['parameters'].get('dimensions', 128)
        self.local_config['parameters']['min_count'] =  self.local_config['parameters'].get('min_count', 5)
        self.local_config['parameters']['down_sampling'] =  self.local_config['parameters'].get('down_sampling', 0.0001)
        self.local_config['parameters']['workers'] =  self.local_config['parameters'].get('workers', 4)
        self.local_config['parameters']['epochs'] =  self.local_config['parameters'].get('epochs', 10)
        self.local_config['parameters']['learning_rate'] =  self.local_config['parameters'].get('learning_rate', 0.025)
        self.local_config['parameters']['seed'] =  self.local_config['parameters'].get('seed', 42)
        self.local_config['parameters']['selected_feature'] =  self.local_config['parameters'].get('selected_feature', None)

    def _get_instace_features(self, instance: GraphInstance):
        feature = self.selected_feature if self.selected_feature in self.dataset.node_features_map.keys() else 'degrees'

        if feature in self.dataset.node_features_map.keys():
            key = self.dataset.node_features_map[feature]
            
            values =  {i:elem for i,elem in enumerate(instance.node_features[:,key])}
            return values
        
        return { node: instance.degree(node) for node in instance.nodes() }

        

        