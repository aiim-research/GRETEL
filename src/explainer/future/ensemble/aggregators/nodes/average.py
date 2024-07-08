from typing import List
from copy import deepcopy
import numpy as np

from src.dataset.instances.base import DataInstance
from src.explainer.future.ensemble.aggregators.nodes.base import NodeFeatureAggregator
from src.utils.utils import pad_features
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.instances.graph import GraphInstance

class AverageAggregator(NodeFeatureAggregator):

    def aggregate(self, aggregated_explanation: LocalGraphCounterfactualExplanation, base_explanations: List[LocalGraphCounterfactualExplanation]):
        result = []
        instances = aggregated_explanation.counterfactual_instances

        for i in range(len(instances)):
            cf_instance = instances[i]
            adj = cf_instance.data
            edges = np.nonzero(adj)
            nodes = np.array(list(range(adj.shape[0])))

            # if there's at least one edge that the aggreagtor produced
            # then get the features of the incident nodes
            if edges[0].size:
                feature_dim = instances[i].node_features.shape[-1]
                avg_node_features = np.zeros((max(nodes)+1, feature_dim))
                for instance in instances:
                    # what nodes are incommon with the nodes of the instance
                    # I need to know the node indices to update only their feature vectors
                    curr_nodes = np.array(list(set(nodes).intersection(set(range(instance.data.shape[0])))))
                    m = max(curr_nodes) + 1
                    # in case the explainers don't do anything in terms of feature vectors
                    # then I need to add zero d-dimensional vectors to the dummy nodes
                    instance.node_features = pad_features(instance.node_features, m)            
                    # add the instance node features
                    avg_node_features[curr_nodes,:] += instance.node_features[curr_nodes,:]

                # Calculating the average node features
                avg_node_features = avg_node_features / len(instances)

                # Updating the node features and creating an updated instance
                cf_candidate = GraphInstance(id=aggregated_explanation.id,
                            label=cf_instance.label,
                            data=adj,
                            node_features=avg_node_features,
                            dataset=aggregated_explanation.dataset)

                # TODO Check if it has sense to re-apply the manipulators
                for manipulator in cf_candidate._dataset.manipulators:
                    manipulator._process_instance(cf_candidate)

                # Append the instance to the new counterfactual instances list
                result.append(cf_candidate)
                
            else:
                cf_candidate = deepcopy(aggregated_explanation.input_instance)
                result.append(cf_candidate)

        return result


