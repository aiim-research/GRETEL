#!/usr/bin/env python3
"""Unit checks for src/utils/reconstruction_probe.py with a fake judge.

Needs no third-party package. Run from the repo root with either of:

    python tests/reconstruction_probe_unit.py
    python -m pytest tests/reconstruction_probe_unit.py
"""
import importlib.util
import os
import sys
import types

# Loaded from their files: importing the src.utils package pulls in torch
_utils = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils')


def _load(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_utils, name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for pkg in ('src', 'src.utils'):
    sys.modules.setdefault(pkg, types.ModuleType(pkg))
_load('probe_common.py', 'src.utils.probe_common')
_load('reconstruction_metrics.py', 'src.utils.reconstruction_metrics')
probe = _load('reconstruction_probe.py', 'src.utils.reconstruction_probe')

TWO_PART = ('Edits:\n```json\n{"edits": {"edges_added": [], "edges_removed": [[1, 2]], "features_changed": []}, '
            '"Natural_Language_Explanation": "Node 1 was disconnected from node 2."}\n```\nDone.')
TRUTH = {'edges_added': [], 'edges_removed': [[1, 2]], 'features_changed': []}
EXTRACTED = '{"edges_added": [], "edges_removed": [[1, 2]], "features_changed": []}'



class EchoJudge:
    """Answers with a fixed delta; records what it was asked."""
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def explain_many(self, pairs):
        self.prompts.extend(pairs)
        return [self.answer] * len(pairs)


def test_mode_dict_skips_outputs_without_narrative():
    judge = EchoJudge(EXTRACTED)
    records = probe.run(judge, [('plain prose', 'G', TRUTH, False), (TWO_PART, 'G', TRUTH, False)],
                        use_context=False, mode='dict')
    assert records[0]['status'] == 'unparsed'
    assert records[1]['status'] == 'success'
    assert records[1]['scores']['edges_removed']['f1'] == 1.0
    assert len(judge.prompts) == 1
    assert 'FACTUAL GRAPH' not in judge.prompts[0][1]


def test_context_on_adds_graph_and_full_mode_uses_raw_output():
    judge = EchoJudge(EXTRACTED.replace('[1, 2]', '[2, 1]'))
    records = probe.run(judge, [(TWO_PART, 'Edges: {(1 -- 2)}', TRUTH, False)], use_context=True, mode='full')
    system, prompt = judge.prompts[0]
    assert system.startswith(probe.JUDGE_SYSTEM_PROMPT)
    assert 'FACTUAL GRAPH' in prompt and 'Edges: {(1 -- 2)}' in prompt
    assert TWO_PART.strip() in prompt
    assert records[0]['scores']['edges']['f1'] == 1.0


def test_unparseable_judge_answer_is_recorded():
    judge = EchoJudge('I cannot tell.')
    records = probe.run(judge, [('some text', 'G', TRUTH, False)], use_context=False, mode='full')
    assert records[0]['status'] == 'judge_unparsed'
    assert records[0]['judge_output'] == 'I cannot tell.'
    assert records[0]['scores'] is None


if __name__ == '__main__':
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print(f'{len(tests)} passed')
