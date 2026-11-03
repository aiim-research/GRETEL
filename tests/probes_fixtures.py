"""Explicitly scripted components for integration tests, never paper models."""
import ast
import json

from src.core.configurable import Configurable
from src.core.explainer_base import Explainer
from src.core.llm_base import LLM
from src.dataset.generators.treecycles_seeded import generate_tree_cycles
from src.dataset.instances.graph import GraphInstance
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.probe_graph import apply_edits


def delta(added=(), removed=(), features=()):
    return dict(nodes_added=[], nodes_removed=[], edges_added=list(added),
                edges_removed=list(removed), features_changed=list(features))


def answer(edits):
    # A synthetic narrative encoding lets us check routing and exact scoring.
    return json.dumps({'edits': edits, 'Natural_Language_Explanation': 'TEST EDITS: ' + json.dumps(edits)})


class DatasetFixture(Configurable):
    def init(self):
        self.domain = 'Undirected synthetic graphs. Class 1 means a cycle exists; class 0 means no cycles.'
        self.node_features_map = {}
        self.instances = [GraphInstance(i, label, adj, dataset=self)
                          for i, (adj, label) in enumerate(generate_tree_cycles(50, 16, .4, 0))]

    def manipulate(self, instance):
        pass


class OneEditFixture(Explainer):
    def init(self):
        pass

    def explain(self, instance):
        assert self.oracle.predict(instance) == 0
        u, v = next((u, v) for u in range(len(instance.data)) for v in range(u + 1, len(instance.data))
                    if not instance.data[u, v])
        cf = apply_edits(instance, delta(added=[[u, v]]))
        assert self.oracle.predict(cf) == 1
        return LocalGraphCounterfactualExplanation(self.context, self.dataset, self.oracle, self, instance, [cf])


class ScriptedLLM(LLM):
    init_counts = {'generator': 0, 'judge': 0}

    def init(self):
        self.init_counts[self.local_config['parameters']['role']] += 1
        self.prompts = []

    def explain_many(self, prompts):
        return [self.explain_counterfactual(*item) for item in prompts]

    def explain_counterfactual(self, system, prompt):
        self.prompts.append((system, prompt))
        if self.local_config['parameters']['role'] == 'generator':
            return answer(json.loads(prompt.split('--- MODIFICATIONS ---\n')[1]))
        if 'Criteria for YES' in system:
            return 'YES'
        if '**Goal.**' in system:
            nodes = ast.literal_eval(prompt.split('Nodes: ')[1].splitlines()[0])
            edges = ast.literal_eval(prompt.split('Edges [u,v,value]: ')[1].splitlines()[0])
            present = {tuple(edge[:2]) for edge in edges}
            original = json.loads(prompt.split('ORIGINAL EDITS:\n')[1].splitlines()[0])
            if 'EXPLANATION:\n' in prompt:
                edit = next([u, v] for u in nodes for v in nodes if u < v and (u, v) not in present
                            and [u, v] not in original['edges_added'])
                return json.dumps(delta(added=[edit]))
            return json.dumps(delta(removed=[list(next(iter(sorted(present))))]))
        # Reconstruction extracts only the supplied narrative, without a truth lookup.
        text = prompt.split('TEST EDITS: ')[1]
        extracted, _ = json.JSONDecoder().raw_decode(text)
        return json.dumps(extracted)


def llm(role):
    return {'class': 'probes_fixtures.ScriptedLLM', 'parameters': {'model': 'scripted-' + role, 'role': role}}
