"""Synthetic narrator for the judge validation V0 of the narrative paper: templated answers with known content.

It reads the typed delta from the generation prompt itself (``--- MODIFICATIONS ---``), so it runs through the
same matrix as the real narrators, and writes the two-part answer: the true delta in ``edits`` and a templated
narrative in which the variant controls what the text states. The expected result of every probe is then known
by construction (``stated_edits`` and ``expected_verdict``), and the judges are scored against it
(scripts/v0_judge_validation.py). No model and no GPU.

Variants (parameter ``variant``):

- ``exact``: every edit and the mechanism.
- ``omit``: about half of the edits (the first n - n // 2) and the mechanism.
- ``spurious``: every edit, n // 2 (at least one) edges added that the change does not contain, and the mechanism.
- ``vague``: no concrete edit, only the mechanism.
- ``no_mechanism``: every edit and no reason why the class changes.
- ``contradict``: as ``exact``, but the identity narrative (sampled with another seed) gives a mechanism that
  contradicts the first one.
- ``static``: every edit and always the mechanism of class 1, whatever the change, so the inverse narrative does
  not negate the direct one.
- ``false_mechanism``: every edit and a mechanism that is false on Tree-Cycles (node degrees instead of cycles).

The mechanism is the true one on Tree-Cycles (a cycle makes the class) and a generic one on molecules
(lipophilicity against polarity), whose truth does not matter for the consistency tests: only the relation
between two narratives does.
"""
import json
import random
import re

from src.core.llm_base import LLM

VARIANTS = ('exact', 'omit', 'spurious', 'vague', 'no_mechanism', 'contradict', 'static', 'false_mechanism')
TESTS = ('inversion', 'renaming', 'identity')


def parse_prompt(system, prompt):
    """(delta, input class, counterfactual class, Tree-Cycles?, nodes, edges of the input) from a generation prompt."""
    original = prompt.split('--- COUNTERFACTUAL ---')[0]
    after = prompt.split('--- COUNTERFACTUAL ---')[1].split('--- MODIFICATIONS ---')[0]
    delta = json.loads(prompt.split('--- MODIFICATIONS ---')[1].strip())
    classes = [int(re.search(r'Oracle predicted class: (\S+)', text).group(1)) for text in (original, after)]
    nodes = json.loads(re.search(r'Nodes: (\[.*?\])', original).group(1))
    edges = json.loads(re.search(r'Edges \[u,v,value\]: (\[.*?\])\n', original).group(1))
    return delta, classes[0], classes[1], 'Tree-Cycles' in system, nodes, {tuple(sorted(e[:2])) for e in edges}


def _edit_list(delta):
    return ([('edges_added', e) for e in delta.get('edges_added', [])]
            + [('edges_removed', e) for e in delta.get('edges_removed', [])]
            + [('features_changed', c) for c in delta.get('features_changed', [])]
            + [('nodes_removed', n) for n in delta.get('nodes_removed', [])])


def stated_edits(variant, system, prompt):
    """The elementary edits the variant's narrative states, as (field, value) pairs."""
    delta, _, _, _, nodes, edges = parse_prompt(system, prompt)
    edits = _edit_list(delta)
    if variant == 'vague':
        return []
    if variant == 'omit':
        return edits[:len(edits) - len(edits) // 2]
    if variant == 'spurious':
        changed = {tuple(sorted(e)) for e in delta.get('edges_added', []) + delta.get('edges_removed', [])}
        candidates = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]
                      if (a, b) not in edges and (a, b) not in changed]
        random.Random(prompt).shuffle(candidates)
        return edits + [('edges_added', list(pair)) for pair in candidates[:max(1, len(edits) // 2)]]
    return edits


def edit_key(field, value):
    if field.startswith('edges_'):
        return (field, *sorted(int(x) for x in value))
    if field == 'features_changed':
        return (field, int(value['node']), str(value['feature']), float(value['from']), float(value['to']))
    return (field, int(value))


def expected_verdict(variant, test):
    if variant == 'no_mechanism':
        return 'no_mechanism'
    if (variant, test) in (('contradict', 'identity'), ('static', 'inversion')):
        return 'incompatible'
    return 'compatible'


def _sentence(field, value, molecule):
    link = 'bond' if molecule else 'edge'
    if field == 'edges_added':
        return f"{'An' if link == 'edge' else 'A'} {link} is added between nodes {value[0]} and {value[1]}."
    if field == 'edges_removed':
        return f'The {link} between nodes {value[0]} and {value[1]} is removed.'
    if field == 'features_changed':
        return f"Node {value['node']}: {value['feature']} changes from {value['from']} to {value['to']}."
    return f'Node {value} is removed.'


def mechanism(kind, tree_cycles, target):
    if tree_cycles:
        if kind == 'false':
            return (f'The model predicts class {target} because these edits change the degree of the nodes involved; '
                    'cycles play no role in the decision.')
        if target == 1:
            return 'These edits close a cycle, and a graph that contains a cycle is predicted as class 1 (cyclic).'
        return 'These edits break every cycle, so the graph becomes a tree, which is predicted as class 0 (acyclic).'
    if (target == 1) != (kind == 'false'):
        return (f'Together these edits make the molecule more lipophilic and compact, which the model associates '
                f'with class {target}.')
    return (f'Together these edits make the molecule more polar and add hydrogen-bond donors, which the model '
            f'associates with class {target}.')


def narrative(variant, system, prompt, other_seed=False):
    _, _, target, tree_cycles, _, _ = parse_prompt(system, prompt)
    stated = stated_edits(variant, system, prompt)
    if variant == 'vague':
        body = 'Several bonds and atoms of the molecule are changed.' if not tree_cycles else 'Several edges of the graph are changed.'
    else:
        body = ' '.join(_sentence(field, value, not tree_cycles) for field, value in stated)
    if variant == 'no_mechanism':
        return body + ' These are the modifications of the counterfactual.'
    if variant == 'static':
        reason = mechanism('true', tree_cycles, 1)
    elif variant == 'false_mechanism' or (variant == 'contradict' and other_seed):
        reason = mechanism('false', tree_cycles, target)
    else:
        reason = mechanism('true', tree_cycles, target)
    return body + ' ' + reason


class SyntheticNarrator(LLM):

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        if p.get('variant') not in VARIANTS:
            raise ValueError('SyntheticNarrator needs a variant among ' + ', '.join(VARIANTS))
        p.setdefault('seed', 0)

    def init(self):
        super().init()
        self.variant = self.local_config['parameters']['variant']
        self.seed = self.local_config['parameters']['seed']

    def explain_counterfactual(self, system, prompt):
        return self.explain_many([(system, prompt)])[0]

    def explain_many(self, pairs, seeds=None):
        seeds = [self.seed] * len(pairs) if seeds is None else [self.seed if s is None else s for s in seeds]
        answers = []
        for (system, prompt), seed in zip(pairs, seeds):
            delta = json.loads(prompt.split('--- MODIFICATIONS ---')[1].strip())
            text = narrative(self.variant, system, prompt, other_seed=seed != self.seed)
            answers.append(json.dumps({'edits': delta, 'Natural_Language_Explanation': text}))
        return answers
