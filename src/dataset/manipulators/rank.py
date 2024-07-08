import networkx as nx
import numpy as np

from src.dataset.manipulators.base import BaseManipulator
from src.utils.metrics.ged import graph_edit_distance_metric


class RankManipulator(BaseManipulator):
    def graph_info(self, instance):

        result = [ (graph_edit_distance_metric(instance.data, x.data, instance.directed and x.directed), x.id) for x in self.dataset.instances]

        result.sort(key=lambda x: x[0])

        dist = [ x for (x,_) in result]
        index = [y for (_,y) in result]

        return { 
            "distance_rank_index" : index, 
            "distance_rank_value": dist
            }
