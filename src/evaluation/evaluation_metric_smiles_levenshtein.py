from functools import lru_cache

from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class SmilesLevenshteinMetric(EvaluationMetric):
    """Provides the ratio between the number of features modified to obtain the counterfactual example
     and the number of features in the original instance. Only considers structural features.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Smiles-Levenshtein'

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset  = None):
        return self.lev_dist(instance_1.smiles, instance_2.smiles)

    def lev_dist(self, a, b):
        '''
        This function will calculate the levenshtein distance between two input
        strings a and b
        
        params:
            a (String) : The first string you want to compare
            b (String) : The second string you want to compare
            
        returns:
            This function will return the distnace between string a and b.
            
        example:
            a = 'stamp'
            b = 'stomp'
            lev_dist(a,b)
            >> 1.0
        '''
        
        @lru_cache(None)  # for memorization
        def min_dist(s1, s2):

            if s1 == len(a) or s2 == len(b):
                return len(a) - s1 + len(b) - s2

            # no change required
            if a[s1] == b[s2]:
                return min_dist(s1 + 1, s2 + 1)

            return 1 + min(
                min_dist(s1, s2 + 1),      # insert character
                min_dist(s1 + 1, s2),      # delete character
                min_dist(s1 + 1, s2 + 1),  # replace character
            )

        return min_dist(0, 0)