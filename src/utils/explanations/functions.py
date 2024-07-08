from typing import List

from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.instances.graph import GraphInstance


def unpack_cf_instances(explanations: List[LocalGraphCounterfactualExplanation]) -> List[GraphInstance]:
    """
    Takes a list of LocalGraphCounterfactualExplanation objects and adds the counterfactual 
    instances of each of then to a single list
    """
    result = []
    for exp in explanations:
        result.extend(exp.counterfactual_instances)
        
    return result