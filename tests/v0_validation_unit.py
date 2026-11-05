#!/usr/bin/env python3
"""The synthetic narrators of V0 and their scorer: every variant gives a valid two-part answer, and a judge that
extracts exactly the stated edits scores fidelity 1 while the expected precision and recall follow the variant.

    python tests/v0_validation_unit.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.LLMexplaneability.synthetic import (VARIANTS, SyntheticNarrator, edit_key, expected_verdict,  # noqa: E402
                                             narrative, stated_edits)
from src.utils.probe_common import generation_outcome, narrative_field  # noqa: E402
from src.utils import reconstruction_probe  # noqa: E402

SYSTEM = 'DOMAIN:\nTree-Cycles is a synthetic benchmark.'
PROMPT = ('--- ORIGINAL ---\nNodes: [0, 1, 2, 3, 4, 5]\nDirected: False\nEdges [u,v,value]: [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0]]\n'
          'Oracle predicted class: 0\n\n--- COUNTERFACTUAL ---\nNodes: [0, 1, 2, 3, 4, 5]\nOracle predicted class: 1\n\n'
          '--- MODIFICATIONS ---\n' + json.dumps({'nodes_added': [], 'nodes_removed': [], 'edges_added': [[0, 2], [2, 4]],
                                                  'edges_removed': [[3, 4]], 'features_changed': []}))


class ParsingJudge:
    """Extracts the edits the synthetic sentences state, as a perfect Reconstruction judge would."""
    local_config = {'parameters': {'model': 'parsing-judge'}}

    def explain_counterfactual(self, system, prompt):
        text = prompt.split('--- TEXT ---')[1]
        added = [[int(a), int(b)] for a, b in re.findall(r'is added between nodes (\d+) and (\d+)', text)]
        removed = [[int(a), int(b)] for a, b in re.findall(r'between nodes (\d+) and (\d+) is removed', text)]
        return json.dumps({'nodes_added': [], 'nodes_removed': [], 'edges_added': added, 'edges_removed': removed,
                           'features_changed': []})


def test_every_variant_gives_a_valid_answer():
    for variant in VARIANTS:
        n = SyntheticNarrator.__new__(SyntheticNarrator)
        n.variant, n.seed = variant, 0
        for answer in n.explain_many([(SYSTEM, PROMPT)] * 2, [0, 1]):
            assert generation_outcome({'status': 'success', 'judge_output': answer})['status'] == 'success', variant


def test_stated_edits_follow_the_variant():
    truth = {edit_key(f, v) for f, v in stated_edits('exact', SYSTEM, PROMPT)}
    assert len(truth) == 3
    assert len(stated_edits('omit', SYSTEM, PROMPT)) == 2 and not stated_edits('vague', SYSTEM, PROMPT)
    spurious = {edit_key(f, v) for f, v in stated_edits('spurious', SYSTEM, PROMPT)}
    assert truth < spurious and len(spurious - truth) == 1


def test_perfect_extraction_has_fidelity_one():
    for variant in VARIANTS:
        answer = json.dumps({'edits': {}, 'Natural_Language_Explanation': narrative(variant, SYSTEM, PROMPT)})
        truth = json.loads(PROMPT.split('--- MODIFICATIONS ---')[1])
        record = reconstruction_probe.run(ParsingJudge(), [(answer, '', truth, False)], False, 'dict', 'transition')[0]
        got = {edit_key(f, v) for f in record['extraction'] for v in record['extraction'][f]}
        assert got == {edit_key(f, v) for f, v in stated_edits(variant, SYSTEM, PROMPT)}, variant


def test_mechanisms():
    direct = narrative('exact', SYSTEM, PROMPT)
    assert 'close a cycle' in direct
    assert 'cycles play no role' in narrative('contradict', SYSTEM, PROMPT, other_seed=True)
    assert 'close a cycle' in narrative('contradict', SYSTEM, PROMPT, other_seed=False)
    assert 'These are the modifications' in narrative('no_mechanism', SYSTEM, PROMPT)
    assert expected_verdict('static', 'inversion') == 'incompatible' and expected_verdict('static', 'identity') == 'compatible'
    assert expected_verdict('contradict', 'identity') == 'incompatible' and expected_verdict('exact', 'renaming') == 'compatible'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok ', test.__name__)
    print(f'{len(tests)} passed')
