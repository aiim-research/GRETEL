from src.core.factory_base import Factory
from src.utils.cfg_utils import inject_dataset

class EmbedderFactory(Factory):

    def get_embedder(self, embedder_snippet, dataset):
        inject_dataset(embedder_snippet, dataset)        
        return self._get_object(embedder_snippet)
        

