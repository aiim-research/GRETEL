"""Edit cost of each counterfactual under its node correspondence (protocol v2, task 0).

``src/utils/metrics/correspondence.py`` defines the cost. The correspondence of a counterfactual is the one
its explainer recorded in ``explanation.info['correspondences']`` (one per counterfactual), and otherwise the
identity over the common node indices, which is exact for every explainer that edits the input. Like the
GED stage, the value of an explanation is the mean over its correct counterfactuals and the aggregate
keeps only correct instances.

Three stages, one value each: EditDistance (total, with the costs ``node_cost``, ``edge_cost`` and
``feature_cost``, all 1 by default), StructuralEditDistance (node and edge edits, unweighted, the
padding-free counterpart of the GED stage) and MatchedFeatureDistance (feature distance of the mapped
nodes, the counterpart of FED under the same correspondence).
"""
import numpy as np

from src.evaluation.future.stages.metric_stage import MetricStage
from src.future.explanation.base import Explanation
from src.utils.metrics.correspondence import instance_edit_cost


class EditDistance(MetricStage):
    component = 'total'

    def check_configuration(self):
        super().check_configuration()
        self.logger = self.context.logger

    def init(self):
        super().init()
        p = self.local_config.get('parameters', {})
        self.costs = {key: p.get(key, 1.0) for key in ('node_cost', 'edge_cost', 'feature_cost')}

    def process(self, explanation: Explanation) -> Explanation:
        input_inst = explanation.input_instance
        input_lbl = explanation.oracle.predict(input_inst)
        explanation.oracle._call_counter -= 1
        mappings = (explanation.info or {}).get('correspondences') or [None] * len(explanation.counterfactual_instances)
        total, correct = 0.0, 0
        for cf, mapping in zip(explanation.counterfactual_instances, mappings):
            cf_lbl = explanation.oracle.predict(cf)
            explanation.oracle._call_counter -= 1
            if cf_lbl != input_lbl:
                correct += 1
                total += instance_edit_cost(input_inst, cf, mapping, **self.costs)[self.component]
        self.write_into_explanation(explanation, float(total / correct) if correct else 0.0)
        return explanation

    @classmethod
    def aggregate(cls, measure_list, instances_correctness_list=None):
        if instances_correctness_list is None:
            return super().aggregate(measure_list, instances_correctness_list)
        kept = [m for m, flag in zip(measure_list, instances_correctness_list) if flag > 0.0]
        return (np.mean(kept), np.std(kept)) if kept else (0.0, 0.0)


class StructuralEditDistance(EditDistance):
    component = 'structure'


class MatchedFeatureDistance(EditDistance):
    component = 'feature_distance'
