from src.core.oracle_base import Oracle 

import numpy as np
#TODO: Revise with the new data_instance logic.

class TrianglesSquaresOracle(Oracle):

    def _real_predict(self, data_instance):
        # Classify
        if len(data_instance.get_nx().nodes) == 3 and len(data_instance.get_nx().edges) == 3:
            return 1 # triangle
        else:
            return 0 # other shape (squares)

    def _real_predict_proba(self, data_instance):
        # Classify
        if len(data_instance.get_nx().nodes) == 3 and len(data_instance.get_nx().edges) == 3:
            return np.array([[0,1]]) # triangle
        else:
            return np.array([[1,0]]) # other shape (squares)



    
