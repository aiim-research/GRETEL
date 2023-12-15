from src.core.embedder_base import Embedder

import numpy as np
from rdkit.Chem import RDKFingerprint


class RDKFingerprintEmbedder(Embedder):

    def init(self):
        self.model = []
        self.embedding = []

    def real_fit(self):
        for inst in self.dataset.instances:
            fingerprint = RDKFingerprint(inst.graph_features['mol'])
            fingerprint_np = np.array(fingerprint)
            self.model.append(fingerprint_np)

    def get_embeddings(self):
        return self.model

    def get_embedding(self, instance):
        fingerprint_rdk = RDKFingerprint(instance.graph_features['mol'])
        fingerprint_rdk_np = np.array(fingerprint_rdk).reshape(1, -1)
        return fingerprint_rdk_np
    
    
    
