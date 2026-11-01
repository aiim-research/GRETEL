#!/usr/bin/env python3
"""General contracts and a two-pass integration test with real TreeCyclesOracle.

LLM answers and the minimal CF explainer are explicitly scripted. --light
replaces only the torch-importing utils package initializer and Dataset's
annotation import; all exercised graph, oracle, factory and stage code is real.
Use --demo-output DIR to retain five sample dumps and the offline results.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--light', action='store_true')
parser.add_argument('--demo-output')
OPTIONS, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
if OPTIONS.light:
    import src
    package = types.ModuleType('src.utils')
    package.__path__ = [str(ROOT / 'src' / 'utils')]
    sys.modules['src.utils'] = package
    src.utils = package
    import src.dataset
    annotation = types.ModuleType('src.dataset.dataset_base')
    annotation.Dataset = type('Dataset', (), {})
    sys.modules[annotation.__name__] = annotation
    src.dataset.dataset_base = annotation

import numpy as np
from probes_fixtures import DatasetFixture, OneEditFixture, ScriptedLLM, answer, delta, llm
from src.core.factory_base import get_instance_kvargs
from src.dataset.instances.graph import GraphInstance
from src.evaluation.future.stages.main_pipeline import MainPipeline
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.oracle.oracle_factory import OracleFactory
from src.utils.context import Context
from src.utils.logger import GLogger
from src.utils.probe_common import call_many, inverse_delta, read_object, saved_generation_outcomes, select_text, validate_delta
from src.utils.probe_graph import apply_edits, restore, snapshot, unsupported_changes
from src.utils.probe_inputs import NARRATIVES_STAGE, run_saved, summarize
from src.utils import reconstruction_probe, reversal_probe, recourse_probe


class Replies:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.prompts = []

    def explain_counterfactual(self, system, prompt):
        self.prompts.append((system, prompt))
        return next(self.answers)


def context_at(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = {'experiment': {'scope': 'scripted-integration'}, 'store_paths': [
        {'name': name + '_store_path', 'address': str(root / name)}
        for name in ('dataset', 'oracle', 'explainer', 'output', 'log', 'working', 'embedder')]}
    path = root / 'test-context.json'
    path.write_text(json.dumps(config))
    Context._Context__global = None
    context = Context.get_context(str(path))
    context.run_number = 1
    GLogger._path = str(root / 'log')
    return context


def integrated_run(root, samples=5):
    context = context_at(root)
    dataset = DatasetFixture(context, {'class': 'probes_fixtures.DatasetFixture', 'parameters': {}})
    oracle = OracleFactory(context).get_oracle({'class': 'src.oracle.custom.oracle_tree_cycles.TreeCyclesOracle', 'parameters': {}}, dataset)
    explainer = OneEditFixture(context, {'class': 'probes_fixtures.OneEditFixture', 'parameters': {'fold_id': 0}, 'dataset': dataset, 'oracle': oracle})
    config = {'class': 'src.evaluation.future.stages.main_pipeline.MainPipeline', 'parameters': {'stages': [
        {'class': 'src.evaluation.future.stages.runtime.Runtime', 'parameters': {}},
        {'class': NARRATIVES_STAGE, 'parameters': {'generator': llm('generator'), 'generator_family': 'scripted-generator'}},
        {'class': 'src.evaluation.future.stages.probe_dump.ProbeDump', 'parameters': {}}]}}
    pipeline = MainPipeline(context, config)
    explanations = []
    for instance in [i for i in dataset.instances if i.label == 0][:samples]:
        explanation = LocalGraphCounterfactualExplanation(context, dataset, oracle, explainer, instance, [])
        explanations.append(pipeline.process(explanation))
    judge = ScriptedLLM(context, llm('judge'))
    options = {'judge': llm('judge'), 'judge_family': 'scripted-judge', 'probes': {name: {} for name in ('reconstruction', 'reversal', 'recourse')}}
    from scripts.run_paper_probes import execute, load_dumps
    dumps = load_dumps(Path(root) / 'output')
    output, errors = execute(judge, dumps, options, context, list(options['probes']), Path(root) / 'results')
    return context, explanations, dumps, output, errors


class GeneralContracts(unittest.TestCase):
    def test_invalid_edits_cannot_become_empty_predictions(self):
        for value in ({}, {'foo': []}, {'edges_added': [[True, 1]], 'edges_removed': [], 'features_changed': []},
                      delta(added=[[-1, 2]]), delta(added=[[1.2, 2]]), delta(added=[[1, 2], [2, 1]]),
                      delta(added=[[1, 2]], removed=[[2, 1]]), delta(features=[{'node': 0, 'feature': 'x'}])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_delta(value, feature_match='transition')
        self.assertFalse(any(validate_delta(delta()).values()))

    def test_quoted_node_indices_are_the_same_nodes(self):
        parsed = validate_delta(delta(removed=[['22', '23']], features=[{'node': '0', 'feature': 'x', 'from': 0, 'to': 1}]))
        self.assertEqual(parsed['edges_removed'], [[22, 23]])
        self.assertEqual(parsed['features_changed'][0]['node'], 0)
        for index in ('node 5', '5.0', ' 5', '-1', '', '\u0665', '+5'):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_delta(delta(added=[[index, 1]]))
        with self.assertRaises(ValueError):
            validate_delta(delta(added=[['1', 2], [1, '2']]))

    def test_saved_generation_validity_follows_the_current_parser(self):
        g = GraphInstance(1, 0, np.zeros((3, 3)))
        item = {'input_status': 'success', 'input': snapshot(g),
                'direct_output': answer(delta(removed=[['1', '2']])), 'inverse_output': None,
                'direct_generation': {'status': 'unparsed', 'call_status': 'success', 'error': 'old parser'},
                'inverse_generation': {'status': 'model_error', 'call_status': 'model_error', 'error': 'timeout'}}
        outcomes = saved_generation_outcomes(item)
        self.assertEqual(outcomes['direct'], {'status': 'success', 'call_status': 'success'})
        self.assertEqual(outcomes['inverse']['status'], 'model_error')

    def test_oracle_view_matches_explainer_candidates(self):
        from src.dataset.manipulators.oracle_view import OracleView
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        g = GraphInstance(0, 0, adj, edge_features=np.arange(8, dtype=float).reshape(4, 2),
                          graph_features={'smile': 'CCO', 'mol': object()})
        dataset = types.SimpleNamespace(instances=[g], edge_features_map={'SINGLE': 0, 'DOUBLE': 1},
                                        graph_features_map={})
        manipulator = object.__new__(OracleView)
        manipulator.dataset = dataset
        manipulator.process()
        self.assertEqual(dataset.edge_features_map, {})
        self.assertEqual(g.edge_features.shape, (4, 1))
        self.assertIsNone(g.graph_features)
        json.dumps(snapshot(g))
        # OFS and LocalSearch build candidates without edge or graph features.
        self.assertFalse(unsupported_changes(g, GraphInstance(0, 0, adj.copy())))

    def test_feature_names_rename_columns_only(self):
        from src.dataset.manipulators.feature_names import FeatureNames
        features = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        g = GraphInstance(0, 0, np.array([[0, 1], [1, 0]]), node_features=features.copy())
        dataset = types.SimpleNamespace(instances=[g], node_features_map={'attribute_0': 0, 'attribute_1': 1, 'attribute_2': 2})
        manipulator = object.__new__(FeatureNames)
        manipulator.dataset = dataset
        manipulator.local_config = {'parameters': {'names': ['charge', 'C', 'O']}}
        manipulator.process()
        self.assertEqual(dataset.node_features_map, {'charge': 0, 'C': 1, 'O': 2})
        np.testing.assert_array_equal(g.node_features, features)
        manipulator.local_config = {'parameters': {'names': ['charge', 'C']}}
        with self.assertRaises(ValueError):
            manipulator.process()

    def test_seed_outcome_tells_minimized_from_returned_seed(self):
        from src.evaluation.future.stages.probe_narratives import seed_outcome
        g = GraphInstance(0, 0, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        seed = apply_edits(g, delta(added=[[0, 2]], removed=[[0, 1]]))
        minimized = apply_edits(g, delta(added=[[0, 2]]))
        self.assertEqual(seed_outcome(g, minimized, seed)['minimized'], True)
        returned = seed_outcome(g, seed, seed)
        self.assertEqual(returned['minimized'], False)
        self.assertEqual((returned['size']['edges_added'], returned['size']['edges_removed']), (1, 1))
        self.assertIsNone(seed_outcome(g, minimized, None))

    @unittest.skipUnless(importlib.util.find_spec('rdkit'), 'needs RDKit')
    def test_seeded_molecules_keep_their_order(self):
        from src.dataset.generators.mol_gen import rdk_edge_features_map, rdk_node_features_map
        from src.dataset.generators.mol_seeded import SeededMolGenerator
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'molecules.csv'
            path.write_text('smiles,y\nCCO,1\nc1ccccc1,0\nnot-a-smiles,1\nCC(=O)O,1\nCCN,0\n')
            orders = []
            for _ in range(2):
                dataset = types.SimpleNamespace(instances=[], node_features_map=rdk_node_features_map(),
                                                edge_features_map=rdk_edge_features_map())
                generator = object.__new__(SeededMolGenerator)
                generator.dataset, generator._data_label_name = dataset, 'y'
                generator.local_config = {'parameters': {'seed': 0}}
                generator._read(str(path))
                orders.append([(g.id, g.graph_features['smile'], g.label) for g in dataset.instances])
        self.assertEqual(orders[0], orders[1])
        self.assertEqual([i for i, _, _ in orders[0]], [0, 1, 2, 3])

    def test_duplicate_json_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            read_object('{"edges_added": [], "edges_added": [[0,1]]}')
        with self.assertRaises(ValueError):
            read_object("{'edits': {}, 'edits': {'edges_added': []}}", allow_literal=True)

    def test_full_preserves_generator_answer_and_dict_has_no_truth(self):
        raw = 'Preface\n```json\n' + answer(delta(added=[[0, 2]])) + '\n```'
        self.assertEqual(select_text(raw, 'full', True), raw)
        self.assertTrue(select_text(raw, 'dict', True).startswith('TEST EDITS:'))
        malformed = json.dumps({'Natural_Language_Explanation': 'Valid-looking text'})
        with self.assertRaises(ValueError):
            select_text(malformed, 'dict', True)

    def test_batch_cardinality_errors_keep_every_attempt(self):
        model = types.SimpleNamespace(explain_many=lambda prompts: ['only one'])
        records = call_many(model, [('a', 'b'), ('a', 'c')])
        self.assertEqual(len(records), 2)
        self.assertEqual([r['status'] for r in records], ['model_error'] * 2)

    def test_directed_edges_and_feature_transitions_are_measured(self):
        truth = delta(added=[[0, 1]], features=[{'node': 0, 'feature': 'x', 'from': 0, 'to': 1}])
        predicted = delta(added=[[1, 0]], features=[{'node': 0, 'feature': 'x', 'from': 1, 'to': 0}])
        r = reconstruction_probe.run(Replies(json.dumps(predicted)), [(answer(truth), '', truth, True)],
                                     False, 'dict', 'transition', True)[0]
        self.assertEqual(r['status'], 'success')
        self.assertEqual(r['scores']['edges']['f1'], 0)
        self.assertEqual(r['scores']['features_changed']['f1'], 0)

    def test_reversal_independent_of_truth_and_semantic_parse(self):
        forward = delta(added=[[7, 9]])
        backward = inverse_delta(forward)
        judge = Replies(json.dumps(forward), json.dumps(backward), 'YES because...')
        r = reversal_probe.run(judge, [(answer(forward), answer(backward), '', '', False)],
                               mode='dict', feature_match='transition', require_structured=True)[0]
        self.assertEqual(r['structural_scores']['edges']['f1'], 1)
        self.assertEqual(r['status'], 'partial_error')
        self.assertEqual(r['semantic']['status'], 'judge_unparsed')
        self.assertEqual(len(judge.prompts), 3)

    def test_snapshot_and_removal_preserve_all_attributes(self):
        a = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
        g = GraphInstance(3, 1, a, node_features=np.array([[.1], [.2], [.3]]),
                          edge_features=np.array([[4], [5], [6], [7]]), edge_weights=np.array([2, 2, 3, 3]),
                          graph_features=np.array([11, 12]))
        clone = restore(json.loads(json.dumps(snapshot(g))))
        for field in ('data', 'node_features', 'edge_features', 'edge_weights', 'graph_features'):
            np.testing.assert_array_equal(getattr(clone, field), getattr(g, field))
        changed = apply_edits(clone, delta(removed=[[0, 1]]))
        np.testing.assert_array_equal(changed.edge_weights, [3, 3])
        np.testing.assert_array_equal(changed.edge_features, [[6], [7]])
        np.testing.assert_array_equal(changed.graph_features, [11, 12])
        with self.assertRaises(NotImplementedError):
            apply_edits(g, delta(added=[[0, 2]]))
        added = apply_edits(g, delta(added=[[0, 2]]), edge_defaults={'weight': 4, 'features': [8]})
        self.assertEqual(len(added.edge_features), 6)
        np.testing.assert_array_equal(g.data, a)

    def test_feature_from_and_node_additions_cannot_be_guessed(self):
        g = GraphInstance(1, 0, np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            apply_edits(g, delta(features=[{'node': 0, 'feature': 'x', 'from': 1, 'to': 2}]), {'x': 0})
        value = delta()
        value['nodes_added'] = [2]
        with self.assertRaises(NotImplementedError):
            apply_edits(g, value)
        candidate = apply_edits(g, delta(features=[{'node': 0, 'feature': 'x', 'from': 0, 'to': 2}]), {'x': 0})
        self.assertEqual(candidate.node_features[0, 0], 2)

    def test_float32_feature_values_and_unrepresentable_changes(self):
        g = GraphInstance(1, 0, np.zeros((2, 2)), node_features=np.array([[.1], [0]]))
        candidate = apply_edits(g, delta(features=[{'node': 0, 'feature': 0, 'from': .1, 'to': .2}]))
        self.assertEqual(candidate.node_features[0, 0], np.float32(.2))
        with self.assertRaises(ValueError):
            apply_edits(g, delta(features=[{'node': 0, 'feature': 0, 'from': .1, 'to': .1000000001}]))

    def test_self_loops_are_visible_in_the_true_delta(self):
        from src.utils.typed_delta import typed_delta_from_instances
        g = GraphInstance(1, 0, np.zeros((2, 2)))
        candidate = apply_edits(g, delta(added=[[0, 0]]))
        self.assertEqual(typed_delta_from_instances(g, candidate)['edges_added'], [[0, 0]])

    def test_snapshot_preserves_empty_attribute_dimensions(self):
        g = GraphInstance(1, 0, np.zeros((2, 2)), node_features=np.empty((2, 0)), edge_features=np.empty((0, 0)))
        restored = restore(json.loads(json.dumps(snapshot(g))))
        self.assertEqual(restored.node_features.shape, (2, 0))
        self.assertEqual(restored.edge_features.shape, (0, 0))
        candidate = apply_edits(restored, delta(added=[[0, 1]]))
        self.assertEqual(candidate.edge_features.shape, (2, 0))

    def test_directed_recourse_changes_only_one_orientation(self):
        g = GraphInstance(1, 0, np.array([[0, 1], [0, 0]]), directed=True)
        candidate = apply_edits(g, delta(added=[[1, 0]]))
        removed = apply_edits(candidate, delta(removed=[[0, 1]]))
        np.testing.assert_array_equal(removed.data, [[0, 0], [1, 0]])

    def test_recourse_rejects_proposal_restored_by_preprocessing(self):
        g = GraphInstance(1, 0, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        truth = delta(added=[[0, 2]])
        item = dict(instance=g, output=answer(truth), truth=truth, input_label=0, target_label=1,
                    graph_text='graph', modifications_text='edits', domain='domain', feature_map={})
        def preprocess(candidate):
            candidate.data[0, 1] = candidate.data[1, 0] = 1
            candidate.edge_features = g.edge_features.copy()
            candidate.edge_weights = g.edge_weights.copy()
        r = recourse_probe.run(Replies(json.dumps(delta(removed=[[0, 1]]))), [item], lambda graph: 0,
                               mode='dict', control=False, require_structured=True, preprocess=preprocess)[0]
        self.assertEqual(r['with_explanation']['status'], 'invalid_proposal')

    def test_padding_nodes_are_hidden_and_edits_on_them_detected(self):
        from src.utils.probe_graph import graph_text, shown_nodes, touches_hidden
        adj = np.zeros((5, 5))
        adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = 1
        features = np.zeros((5, 2))
        features[:3] = [[6, 1], [8, 0], [6, 1]]
        g = GraphInstance(1, 0, adj, node_features=features)
        self.assertEqual(shown_nodes(g), 3)
        text = graph_text(g, 0, {'AtomicNum': 0, 'IsAromatic': 1}, 3)
        self.assertIn('Nodes: [0, 1, 2]\n', text)
        self.assertIn('Node features: [[6.0, 1.0], [8.0, 0.0], [6.0, 1.0]]\n', text)
        self.assertFalse(touches_hidden(g, apply_edits(g, delta(removed=[[0, 1]])), 3))
        padded_edge = apply_edits(g, delta(added=[[2, 3]]))
        self.assertTrue(touches_hidden(g, padded_edge, 3))
        with self.assertRaises(ValueError):
            graph_text(padded_edge, 1, {}, 3)

    def test_recourse_rejects_edits_on_hidden_padding(self):
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = 1
        g = GraphInstance(1, 0, adj)
        truth = delta(added=[[0, 2]])
        item = dict(instance=g, output=answer(truth), truth=truth, input_label=0, target_label=1,
                    graph_text='graph', modifications_text='edits', domain='domain', feature_map={},
                    shown_nodes=3)
        r = recourse_probe.run(Replies(json.dumps(delta(added=[[2, 3]]))), [item], lambda graph: 0,
                               mode='dict', control=False, require_structured=True)[0]
        self.assertEqual(r['with_explanation']['status'], 'invalid_proposal')
        self.assertIn('does not show', r['with_explanation']['error'])

    def test_backend_final_channel_does_not_mix_in_analysis(self):
        from src.LLMexplaneability.huggingface import HuggingFaceLLM
        text = '<|channel|>analysis<|message|>{"wrong":1}<|end|><|start|>assistant<|channel|>final<|message|>{"right":2}<|return|>'
        self.assertEqual(HuggingFaceLLM._clean(text), '{"right":2}')
        self.assertEqual(HuggingFaceLLM._clean('<|channel|>analysis<|message|>{"wrong":1}'), '')
        self.assertEqual(HuggingFaceLLM._clean('<think>unfinished'), '')
        self.assertEqual(HuggingFaceLLM._clean('<think>reasoning</think>YES'), 'YES')

    def test_backend_keeps_flattened_harmony_final_without_assistant_analysis(self):
        from src.LLMexplaneability.huggingface import HuggingFaceLLM
        for text, expected in (
            ('analysisWe checked both directions. So YES.assistantfinalYES', 'YES'),
            ('analysisThe classes do not swap.assistantfinalNO', 'NO'),
            ('assistantanalysisReasoning.assistantfinalYES', 'YES'),
            ('assistantfinalYES', 'YES'),
            ('analysisReasoning.assistantfinal{"edges_added": []}', '{"edges_added": []}'),
            ('{"narrative": "The label is assistantfinal."}', '{"narrative": "The label is assistantfinal."}'),
        ):
            with self.subTest(text=text):
                self.assertEqual(HuggingFaceLLM._clean(text), expected)

    def test_backend_keeps_muse_user_response_and_discards_unfinished_self_message(self):
        from src.LLMexplaneability.huggingface import HuggingFaceLLM
        for final in ('YES', 'NO', json.dumps(delta(added=[[0, 1]]))):
            raw = ' to=self<|message|>draft {"wrong":1}<|eom|><|start|>assistant to=user<|message|>' + final + '<|eom|>'
            self.assertEqual(HuggingFaceLLM._clean(raw), final)
            self.assertEqual(HuggingFaceLLM._clean(' to=user<|message|>' + final + '<|eom|>'), final)
        self.assertEqual(HuggingFaceLLM._clean(' to=self<|message|>YES'), '')
        self.assertEqual(HuggingFaceLLM._clean('reasoning from prompt</think>NO'), 'NO')

    def test_cached_backend_uses_snapshot_without_requiring_other_weight_exports(self):
        from src.LLMexplaneability.huggingface import HuggingFaceLLM
        with tempfile.TemporaryDirectory() as root:
            snapshot = Path(root) / 'snapshots' / ('a' * 40)
            snapshot.mkdir(parents=True)
            (snapshot / 'config.json').write_text('{}')
            calls = []
            hub = types.SimpleNamespace(try_to_load_from_cache=lambda *args, **kwargs: str(snapshot / 'config.json'))
            vllm = types.SimpleNamespace(LLM=lambda **kwargs: calls.append(kwargs), SamplingParams=lambda **kwargs: kwargs)
            judge = HuggingFaceLLM.__new__(HuggingFaceLLM)
            judge.model_id, judge.seed, judge.temperature, judge.top_p, judge.max_new_tokens = 'openai/gpt-oss-20b', 0, 0, 1, 2048
            p = {'local_files_only': True, 'revision': 'a' * 40, 'dtype': 'auto', 'quantization': None,
                 'tensor_parallel_size': 1, 'gpu_memory_utilization': .75, 'max_model_len': 8192,
                 'max_num_seqs': 1, 'limit_mm_per_prompt': {'image': 0}}
            with patch.dict(sys.modules, {'huggingface_hub': hub, 'vllm': vllm}):
                judge._init_vllm(p)
                self.assertEqual(calls[0]['model'], str(snapshot))
                self.assertEqual(p['resolved_revision'], 'a' * 40)
                self.assertEqual(calls[0]['max_num_seqs'], 1)
                self.assertTrue(judge.sampling['skip_special_tokens'])
                self.assertNotIn('revision', calls[0])
                judge.model_id = 'meta-models/Muse-Glimmer-30B'
                judge._init_vllm(p)
                self.assertFalse(judge.sampling['skip_special_tokens'])
                self.assertFalse(judge.sampling['spaces_between_special_tokens'])
                judge.system_prefix = 'Reasoning strength: low'
                self.assertEqual(judge._messages('task', 'text')[0]['content'], 'Reasoning strength: low\ntask')
            with patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(try_to_load_from_cache=lambda *a, **k: None)}):
                with self.assertRaises(FileNotFoundError):
                    judge._model_path(p)

    def test_backend_retains_raw_answers_and_stop_diagnostics_across_probes(self):
        from src.LLMexplaneability.huggingface import HuggingFaceLLM, _LLMAnswer
        edit = delta(added=[[0, 1]])
        final = json.dumps(edit)
        raw = 'to=self<|message|>draft<|eom|><|start|>assistant to=user<|message|>' + final + '<|eom|>'
        unfinished = 'analysisRepeated draft {"wrong": 1}'
        judge = HuggingFaceLLM.__new__(HuggingFaceLLM)
        judge.engine, judge.sampling, judge.chat_kwargs = 'vllm', None, {}
        completion = lambda text, reason: types.SimpleNamespace(outputs=[types.SimpleNamespace(
            text=text, finish_reason=reason, stop_reason=None, token_ids=[1, 2, 3])])
        judge.llm = types.SimpleNamespace(chat=lambda *a, **k: [completion(raw, 'stop'), completion(unfinished, 'length')])
        records = reconstruction_probe.run(judge, [(answer(edit), '', edit, False)] * 2,
            use_context=False, mode='dict', feature_match='transition', require_structured=True)
        self.assertEqual(records[0]['status'], 'success')
        self.assertEqual(records[0]['judge_output'], final)
        self.assertEqual(records[0]['judge_raw_output'], raw)
        self.assertEqual(records[0]['finish_reason'], 'stop')
        self.assertEqual(records[1]['status'], 'judge_unparsed')
        self.assertEqual(records[1]['error'], 'Missing model output')
        self.assertEqual(records[1]['judge_output'], '')
        self.assertEqual(records[1]['judge_raw_output'], unfinished)
        self.assertEqual(records[1]['finish_reason'], 'length')
        self.assertEqual(records[1]['output_token_count'], 3)
        json.loads(json.dumps(records))  # Diagnostics remain ordinary saved JSON.
        unknown = reconstruction_probe.run(Replies(json.dumps({**edit, 'size': {}})),
            [(answer(edit), '', edit, False)], use_context=False, mode='dict')[0]
        self.assertEqual(unknown['error'], 'Unknown delta fields')
        self.assertEqual(HuggingFaceLLM._clean('Analysis of edits'), 'Analysis of edits')
        copied = copy.deepcopy(judge.explain_counterfactual('system', 'prompt'))
        self.assertEqual(copied, final)
        self.assertEqual(copied.llm_metadata['judge_raw_output'], raw)
        inverse = inverse_delta(edit)
        semantic_raw = 'analysisChecked both directions.assistantfinalYES'
        r = reversal_probe.run(Replies(_LLMAnswer(final, raw, 'stop'), json.dumps(inverse),
            _LLMAnswer('YES', semantic_raw, 'stop')),
            [(answer(edit), answer(inverse), '', '', False)], mode='dict', require_structured=True)[0]
        self.assertEqual(r['status'], 'success')
        self.assertEqual(r['forward']['judge_raw_output'], raw)
        self.assertEqual(r['semantic']['judge_raw_output'], semantic_raw)
        g = GraphInstance(1, 0, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        truth = delta(added=[[0, 2]])
        item = dict(instance=g, output=answer(truth), truth=truth, input_label=0, target_label=1,
                    graph_text='graph', modifications_text='edits', domain='domain', feature_map={})
        proposal = json.dumps(delta(removed=[[0, 1]]))
        r = recourse_probe.run(Replies(_LLMAnswer(proposal, 'raw proposal', 'stop')),
            [item], lambda graph: 0, mode='dict', control=False, require_structured=True)[0]
        self.assertEqual(r['status'], 'success')
        self.assertEqual(r['with_explanation']['judge_raw_output'], 'raw proposal')

    def test_reversal_flattened_harmony_final_still_requires_exact_verdict(self):
        from src.LLMexplaneability.huggingface import HuggingFaceLLM
        forward = delta(added=[[8, 9]])
        backward = inverse_delta(forward)
        for final, status, consistent in (
            ('YES', 'success', True), ('NO', 'success', False),
            ('YES because the edits are inverse', 'partial_error', None),
            ('', 'partial_error', None),
        ):
            with self.subTest(final=final):
                replies = iter([json.dumps(forward), json.dumps(backward),
                                'analysisThe pair was examined.assistantfinal' + final])
                calls = []
                def chat(messages, sampling, **kwargs):
                    calls.extend(messages)
                    return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=next(replies))])
                            for _ in messages]
                # Exercise the real backend answer path without loading model weights.
                judge = HuggingFaceLLM.__new__(HuggingFaceLLM)
                judge.engine, judge.sampling, judge.chat_kwargs = 'vllm', None, {}
                judge.llm = types.SimpleNamespace(chat=chat)
                record = reversal_probe.run(judge, [(answer(forward), answer(backward), '', '', False)],
                                            mode='dict', feature_match='transition', require_structured=True)[0]
                self.assertEqual(record['structural_scores']['edges']['f1'], 1)
                self.assertEqual(record['status'], status)
                self.assertEqual(record['semantic'].get('consistent'), consistent)
                self.assertEqual(record['semantic']['judge_output'], final)
                self.assertEqual(len(calls), 3)
                summary = summarize('reversal', [record])
                self.assertEqual(summary['n_semantic_valid'], int(status == 'success'))
                self.assertEqual(summary['semantic_yes_count'], int(consistent is True))
                if status != 'success':
                    self.assertEqual(record['semantic']['status'], 'judge_unparsed')

    def test_recourse_control_differs_only_by_explanation(self):
        a = recourse_probe.build_prompt('graph', 'edits', 'domain', 'narrative', 1)
        b = recourse_probe.build_prompt('graph', 'edits', 'domain', None, 1)
        self.assertEqual(a[1].replace('EXPLANATION:\nnarrative\n', ''), b[1])
        self.assertNotIn('Explanation', b[0])
        self.assertNotIn('empty set', a[0])

    def test_recourse_empty_repeat_bad_index_and_oracle_errors(self):
        g = GraphInstance(1, 0, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        truth = delta(added=[[0, 2]])
        item = dict(instance=g, output=answer(truth), truth=truth, input_label=0, target_label=1,
                    graph_text='graph', modifications_text='edits', domain='domain', feature_map={})
        for edit, status in ((delta(), 'invalid_proposal'), (truth, 'reused_original'), (delta(added=[[0, 20]]), 'invalid_proposal')):
            r = recourse_probe.run(Replies(json.dumps(edit)), [item], lambda graph: 0, mode='dict', control=False, require_structured=True)[0]
            self.assertEqual(r['with_explanation']['status'], status)
        r = recourse_probe.run(Replies(json.dumps(delta(removed=[[0, 1]]))), [item], lambda graph: 0,
                               mode='dict', control=False, require_structured=True)[0]
        self.assertEqual(r['status'], 'success')
        self.assertFalse(r['with_explanation']['successful'])
        self.assertIn('candidate', r['with_explanation'])
        bad = dict(item, input_label=1)
        r = recourse_probe.run(Replies(), [bad], lambda graph: 0, control=False)[0]
        self.assertEqual(r['status'], 'oracle_error')

    def test_recourse_rejects_partial_reuse_in_both_arms(self):
        g = GraphInstance(1, 0, np.zeros((4, 4)))
        truth = delta(added=[[0, 1], [0, 2]])
        item = dict(instance=g, output=answer(truth), truth=truth, input_label=0, target_label=1,
                    graph_text='graph', modifications_text='edits', domain='domain', feature_map={})
        for proposal in (delta(added=[[1, 0]]), delta(added=[[0, 1], [0, 3]])):
            calls = []
            def predictor(graph):
                calls.append(graph)
                return 0
            r = recourse_probe.run(Replies(json.dumps(proposal), json.dumps(proposal)), [item], predictor,
                                   mode='dict', require_structured=True)[0]
            for arm in ('with_explanation', 'without_explanation'):
                self.assertEqual(r[arm]['status'], 'reused_original')
                self.assertEqual(r[arm]['reused_edits'], {'edges_added': [[0, 1]]})
                self.assertEqual(r[arm]['reuse_phase'], 'proposal')
            self.assertEqual(len(calls), 1)  # Factual replay only, no invalid-candidate predictions.
            summary = summarize('recourse', [r])
            self.assertEqual(summary['with_explanation_n_reused_original'], 1)
            self.assertEqual(summary['without_explanation_n_reused_original'], 1)
            self.assertEqual(summary['with_explanation_n_valid'], 0)
            self.assertEqual(summary['with_explanation_success_rate_all_attempts'], 0)

    def test_recourse_reuse_respects_orientation_and_feature_aliases(self):
        g = GraphInstance(1, 0, np.array([[0, 1], [0, 0]]), directed=True)
        truth = delta(removed=[[0, 1]], features=[{'node': 0, 'feature': 'x', 'from': 0.0, 'to': 1.0}])
        item = dict(instance=g, output=answer(truth), truth=truth, input_label=0, target_label=1,
                    graph_text='graph', modifications_text='edits', domain='domain', feature_map={'x': 0})
        proposal = delta(features=[{'node': 0, 'feature': 0, 'from': 0, 'to': 1}])
        r = recourse_probe.run(Replies(json.dumps(proposal)), [item], lambda graph: 0,
                               mode='dict', control=False, require_structured=True)[0]
        self.assertEqual(r['with_explanation']['status'], 'reused_original')
        self.assertIn('features_changed', r['with_explanation']['reused_edits'])
        # Another transition on that column, and the opposite directed edge, are new edits.
        proposal = delta(added=[[1, 0]], features=[{'node': 0, 'feature': 0, 'from': 0, 'to': 2}])
        r = recourse_probe.run(Replies(json.dumps(proposal)), [item], lambda graph: int(graph.data[1, 0] != 0),
                               mode='dict', control=False, require_structured=True)[0]
        self.assertTrue(r['with_explanation']['successful'])

    def test_recourse_rejects_reuse_created_by_preprocessing(self):
        g = GraphInstance(1, 0, np.zeros((4, 4)))
        truth = delta(added=[[0, 1]])
        item = dict(instance=g, output=answer(truth), truth=truth, input_label=0, target_label=1,
                    graph_text='graph', modifications_text='edits', domain='domain', feature_map={})
        def preprocess(candidate):
            changed = apply_edits(candidate, truth)
            candidate.data = changed.data
            candidate.edge_features = changed.edge_features
            candidate.edge_weights = changed.edge_weights
        r = recourse_probe.run(Replies(json.dumps(delta(added=[[0, 2]]))), [item], lambda graph: 0,
                               mode='dict', control=False, require_structured=True, preprocess=preprocess)[0]
        arm = r['with_explanation']
        self.assertEqual(arm['status'], 'reused_original')
        self.assertEqual(arm['reuse_phase'], 'realized')
        self.assertEqual(arm['reused_edits'], {'edges_added': [[0, 1]]})
        self.assertEqual(arm['realized_edits']['edges_added'], [[0, 1], [0, 2]])
        self.assertIn('candidate', arm)

    def test_summary_exposes_errors_and_both_denominators(self):
        rows = [{'status': 'success', 'with_explanation': {'status': 'success', 'successful': True}},
                {'status': 'input_error'}, {'status': 'partial_error', 'with_explanation': {'status': 'judge_unparsed'}}]
        s = summarize('recourse', rows)
        self.assertEqual(s['n'], 3)
        self.assertEqual(s['with_explanation_success_rate_all_attempts'], 1 / 3)
        self.assertEqual(s['with_explanation_success_rate_valid'], 1)

    def test_model_error_remains_distinct_from_wrong_extraction(self):
        truth = delta(added=[[0, 2]])
        records = reconstruction_probe.run(Replies(None, '{}', json.dumps(delta())),
                                           [(answer(truth), '', truth, False)] * 3,
                                           False, 'dict', 'transition', True)
        self.assertEqual([r['status'] for r in records], ['model_error', 'judge_unparsed', 'success'])
        self.assertEqual(records[2]['scores']['edges']['f1'], 0)

    def test_saved_generation_states_keep_skipped_and_batch_failures(self):
        g = GraphInstance(1, 0, np.zeros((2, 2)))
        truth = delta(added=[[0, 1]])
        item = {'status': 'partial_error', 'input_status': 'success', 'input': snapshot(g),
                'truth': truth, 'direct_output': None, 'inverse_output': None}
        failure = call_many(types.SimpleNamespace(explain_many=lambda prompts: []), [('s', 'direct'), ('s', 'inverse')])
        item.update(direct_generation=failure[0], inverse_generation=failure[1])
        saved = {'schema_version': 1, 'node_matching': 'position', 'counterfactuals': [item]}
        judge = Replies()
        r = run_saved('reversal', judge, saved)[0]
        self.assertEqual(r['status'], 'model_error')
        self.assertEqual(r['generation_failed_directions'], ['direct', 'inverse'])
        self.assertFalse(judge.prompts)
        skipped = {'status': 'not_counterfactual', 'error': 'No flip'}
        saved['counterfactuals'] = [skipped]
        r = run_saved('reconstruction', judge, saved)[0]
        self.assertEqual(r['status'], 'input_error')
        self.assertEqual(r['input_status'], 'not_counterfactual')
        for outcome in r['generation_outcomes'].values():
            self.assertEqual(outcome['status'], 'not_attempted')

    def test_invalid_runner_conditions_fail_before_inference(self):
        from scripts.run_paper_probes import validate_configuration
        for options in ({'conditions': [{'context': 'on'}]}, {'control': 'false'},
                        {'conditions': [{}, {}]}, {'conditions': []}):
            with self.assertRaises(ValueError):
                validate_configuration({'probes': {'recourse': options}}, ['recourse'])

    def test_corrupted_saved_file_is_retained_as_an_error(self):
        from scripts.run_paper_probes import load_dumps
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'probe_inputs'
            folder.mkdir()
            (folder / 'cf_bad.json').write_text('{broken')
            dumps = load_dumps(root)
            self.assertEqual(len(dumps), 1)
            self.assertIn('input_error', dumps[0][1])


class TwoPassIntegration(unittest.TestCase):
    def test_generation_states_survive_dump_and_do_not_block_unrelated_probes(self):
        with tempfile.TemporaryDirectory(prefix='gretel-generation-states-') as root:
            context, explanations, _, _, _ = integrated_run(root, samples=1)
            exp = explanations[0]
            original = copy.deepcopy(exp.stages_info[NARRATIVES_STAGE])
            direct = original['counterfactuals'][0]['direct_output']
            inverse = original['counterfactuals'][0]['inverse_output']
            stage = get_instance_kvargs(NARRATIVES_STAGE, {'context': context,
                'local_config': {'parameters': {'generator': llm('generator')}}})
            dump = get_instance_kvargs('src.evaluation.future.stages.probe_dump.ProbeDump',
                                       {'context': context, 'local_config': {'parameters': {}}})
            from scripts.run_paper_probes import execute, load_dumps
            for direction, replies in (('inverse', Replies(direct)), ('direct', Replies(None, inverse))):
                replies.local_config = llm('generator')
                stage.generator = replies
                stage.process(exp)
                dump.process(exp)
                dumps = load_dumps(Path(root) / 'output')
                saved = dumps[0][1]['data']
                item = saved['counterfactuals'][0]
                self.assertEqual(item['input_status'], 'success')
                self.assertEqual(item['status'], 'partial_error')
                self.assertEqual(item[direction + '_generation']['status'], 'model_error')
                self.assertIn('error', item[direction + '_generation'])
                options = {'judge': llm('judge'), 'probes': {n: {} for n in ('reconstruction', 'reversal', 'recourse')}}
                output, errors = execute(ScriptedLLM(context, llm('judge')), dumps, options, context,
                                         list(options['probes']), Path(root) / 'results')
                self.assertTrue(errors)
                self.assertEqual(json.loads((output / 'manifest.json').read_text())['protocol_version'], 6)
                for name in options['probes']:
                    path = next((output / name / 'ctx-off_mode-dict').glob('*.json'))
                    result = json.loads(path.read_text())['counterfactuals'][0]
                    expected = 'success' if direction == 'inverse' and name != 'reversal' else 'model_error'
                    self.assertEqual(result['status'], expected)
                    self.assertEqual(result['generation_outcomes'][direction]['status'], 'model_error')
                    if expected == 'model_error':
                        self.assertEqual(result['error_origin'], 'generator')
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary[1]['generation_status_counts'][direction], {'model_error': 1})

    def test_malformed_generation_is_distinct_from_backend_error_and_legacy_is_readable(self):
        with tempfile.TemporaryDirectory(prefix='gretel-malformed-generation-') as root:
            context, explanations, _, _, _ = integrated_run(root, samples=1)
            exp = explanations[0]
            inverse = exp.stages_info[NARRATIVES_STAGE]['counterfactuals'][0]['inverse_output']
            stage = get_instance_kvargs(NARRATIVES_STAGE, {'context': context,
                'local_config': {'parameters': {'generator': llm('generator')}}})
            model = Replies('{}', inverse)
            model.local_config = llm('generator')
            stage.generator = model
            stage.process(exp)
            saved = exp.stages_info[NARRATIVES_STAGE]
            generation = saved['counterfactuals'][0]['direct_generation']
            self.assertEqual(generation['call_status'], 'success')
            self.assertEqual(generation['status'], 'unparsed')
            judge = Replies()
            record = run_saved('reconstruction', judge, saved)[0]
            self.assertEqual(record['status'], 'unparsed')
            self.assertEqual(record['error_origin'], 'generator')
            self.assertFalse(judge.prompts)
            # Old schema-1 records have no input_status; backend errors still propagate.
            old = copy.deepcopy(saved)
            item = old['counterfactuals'][0]
            item.pop('input_status')
            item['status'] = 'success'
            item['direct_generation'] = {'status': 'model_error', 'error': 'backend failure'}
            item['direct_output'] = None
            record = run_saved('reversal', Replies(), old)[0]
            self.assertEqual(record['status'], 'model_error')
            self.assertEqual(record['generation_failed_directions'], ['direct'])
            # Older valid text without direction metadata can still be evaluated.
            item['direct_output'] = answer(delta(added=[[0, 1]]))
            item.pop('direct_generation')
            extracted = {k: v for k, v in item['truth'].items() if k != 'size'}
            record = run_saved('reconstruction', Replies(json.dumps(extracted)), old)[0]
            self.assertEqual(record['status'], 'success')
            self.assertEqual(record['generation_outcomes']['direct']['status'], 'success')

    def test_five_treecycles_graphs_all_probes_and_offline_stage_parity(self):
        with tempfile.TemporaryDirectory(prefix='gretel-paper-probes-') as root:
            generator_inits = ScriptedLLM.init_counts['generator']
            context, explanations, dumps, output, errors = integrated_run(root)
            self.assertEqual(ScriptedLLM.init_counts['generator'] - generator_inits, 1)
            self.assertFalse(errors)
            self.assertEqual(len(dumps), 5)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual([r['n'] for r in summary], [5, 5, 5])
            self.assertEqual(summary[0]['edges_f1_valid_mean'], 1)
            self.assertEqual(summary[1]['edges_f1_valid_mean'], 1)
            self.assertEqual(summary[1]['semantic_yes_count'], 5)
            self.assertEqual(summary[2]['with_explanation_n_successful'], 5)
            self.assertEqual(summary[2]['without_explanation_n_successful'], 0)
            for name in ('reconstruction', 'reversal', 'recourse'):
                stage = get_instance_kvargs('src.evaluation.future.stages.' + name + '.' + name.capitalize(),
                    {'context': context, 'local_config': {'parameters': {'judge': llm('judge'),
                     'narratives_stage': NARRATIVES_STAGE, 'feature_match': 'transition', 'mode': 'dict'}}})
                before = explanations[0].oracle.get_calls_count()
                stage.process(explanations[0])
                self.assertEqual(before, explanations[0].oracle.get_calls_count())
                actual = explanations[0].stages_info[context.get_fullname(stage)]['counterfactuals']
                expected = run_saved(name, ScriptedLLM(context, llm('judge')), explanations[0].stages_info[NARRATIVES_STAGE],
                                     predictor=explanations[0].oracle.predict, dataset=explanations[0].dataset,
                                     preprocess=explanations[0].dataset.manipulate)
                self.assertEqual(actual, expected)
            # Reusing sample IDs in another fold must preserve the first files.
            explanations[0].explainer.fold_id = 1
            pipeline_dump = get_instance_kvargs('src.evaluation.future.stages.probe_dump.ProbeDump',
                                                {'context': context, 'local_config': {'parameters': {}}})
            pipeline_dump.process(explanations[0])
            from scripts.run_paper_probes import load_dumps
            self.assertEqual(len(load_dumps(Path(root) / 'output')), 6)
            self.assertEqual(len(load_dumps(Path(root) / 'output', 2)), 2)
            saved = copy.deepcopy(explanations[0].stages_info[NARRATIVES_STAGE])
            without_truth = copy.deepcopy(saved)
            without_truth['counterfactuals'][0].pop('truth')
            independent = run_saved('reversal', ScriptedLLM(context, llm('judge')), without_truth)
            self.assertEqual(independent[0]['status'], 'success')
            saved['counterfactuals'].append({'status': 'not_counterfactual'})
            records = run_saved('reconstruction', ScriptedLLM(context, llm('judge')), saved)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]['status'], 'input_error')
            saved['generator'] = llm('judge')
            with self.assertRaises(ValueError):
                run_saved('reversal', ScriptedLLM(context, llm('judge')), saved)


class ModelMatrixIntegration(unittest.TestCase):
    def prepare_fixture(self, root, samples=5):
        from scripts import paper_probe_matrix as matrix
        context, _, dumps, _, _ = integrated_run(Path(root) / 'fixture', samples=samples)
        config = json.loads((ROOT / matrix.DEFAULT_CONFIG).read_text())
        # The shared integration fixture uses 16 nodes; production requires 10.
        config['num_nodes'] = 16
        path = Path(root) / 'matrix-config.json'
        path.write_text(json.dumps(config))
        with patch.object(matrix.subprocess, 'check_output', return_value='0' * 40 + '\n'):
            run = matrix.prepare(path, Path(root) / 'fixture/output', Path(root) / 'matrix')
        for path in (run / 'configs').glob('*.json'):
            settings = json.loads(path.read_text())
            settings.pop('compose_strs')
            settings['store_paths'] = context.conf['store_paths']
            path.write_text(json.dumps(settings))
        return matrix, config, run, context, dumps

    def test_all_models_replay_same_pairs_and_each_judge_loads_once(self):
        with tempfile.TemporaryDirectory(prefix='gretel-matrix-') as root:
            matrix, config, run, context, originals = self.prepare_fixture(root)
            generator_models = {entry['parameters']['model'] for entry in config['generators']}
            loaded = []
            factory = get_instance_kvargs
            def scripted_model(class_name, kwargs):
                if class_name != 'src.LLMexplaneability.huggingface.HuggingFaceLLM':
                    return factory(class_name, kwargs)
                model = kwargs['local_config']['parameters']['model']
                role = 'generator' if model in generator_models else 'judge'
                loaded.append((role, model))
                return ScriptedLLM(kwargs['context'], {'class': 'probes_fixtures.ScriptedLLM',
                    'parameters': {'role': role, 'model': model}})
            with patch('src.core.factory_base.get_instance_kvargs', side_effect=scripted_model):
                for entry in config['generators']:
                    Context._Context__global = None
                    matrix.run_generator(run, entry['id'])
                for entry in config['judges']:
                    Context._Context__global = None
                    self.assertEqual(matrix.run_judge(run, entry['id']), 0)
            self.assertEqual(len(loaded), 9)
            self.assertEqual(sum(role == 'judge' for role, _ in loaded), 3)
            rows, cells = matrix.collect(run)
            self.assertEqual(len(cells), 18)
            self.assertTrue(all(cell['status'] == 'complete' for cell in cells))
            self.assertEqual(len(rows), 54)
            self.assertTrue(all(row['n'] == 5 and row['n_success'] == 5 for row in rows))
            for entry in config['generators']:
                from scripts.run_paper_probes import load_dumps
                generated = load_dumps(run / 'generated' / entry['id'])
                sources = {payload['id']: payload['data'] for _, payload, _ in originals}
                self.assertEqual(len(generated), 5)
                for _, payload, _ in generated:
                    old, new = sources[payload['id']], payload['data']
                    self.assertEqual(new['generator']['parameters']['model'], entry['parameters']['model'])
                    for before, after in zip(old['counterfactuals'], new['counterfactuals']):
                        for field in ('input', 'counterfactual', 'truth', 'input_label', 'target_label'):
                            self.assertEqual(after[field], before[field])
                        for direction in ('direct', 'inverse'):
                            self.assertEqual(after[direction + '_generation']['judge_request'],
                                             before[direction + '_generation']['judge_request'])

    def test_regeneration_retains_bad_answers_and_preserves_original_dump(self):
        from src.utils.probe_generation import regenerate_saved
        with tempfile.TemporaryDirectory(prefix='gretel-regenerate-') as root:
            context, _, dumps, _, _ = integrated_run(root, samples=1)
            source = dumps[0][1]['data']
            before = copy.deepcopy(source)
            model = Replies(source['counterfactuals'][0]['direct_output'], '{}')
            model.local_config = llm('generator')
            result = regenerate_saved(model, source, 'scripted-generator')
            self.assertEqual(source, before)
            record = result['counterfactuals'][0]
            self.assertEqual(record['status'], 'partial_error')
            self.assertEqual(record['direct_generation']['status'], 'success')
            self.assertEqual(record['inverse_generation']['status'], 'unparsed')
            self.assertEqual(record['inverse_output'], '{}')
            self.assertEqual(run_saved('reconstruction', ScriptedLLM(context, llm('judge')), result)[0]['status'], 'success')
            self.assertEqual(run_saved('reversal', ScriptedLLM(context, llm('judge')), result)[0]['error_origin'], 'generator')

    def test_saved_recourse_prompt_has_only_edits_without_changing_truth(self):
        with tempfile.TemporaryDirectory(prefix='gretel-recourse-prompt-') as root:
            context, _, dumps, _, _ = integrated_run(root, samples=1)
            saved = copy.deepcopy(dumps[0][1]['data'])
            original = copy.deepcopy(saved)
            captured = []
            def probe(judge, items, *args):
                captured.extend(items)
                return [{'status': 'success'} for _ in items]
            with patch.object(recourse_probe, 'run', side_effect=probe):
                run_saved('recourse', ScriptedLLM(context, llm('judge')), saved,
                          predictor=lambda graph: original['counterfactuals'][0]['target_label'])
            self.assertIn('size', original['counterfactuals'][0]['truth'])
            self.assertNotIn('size', json.loads(captured[0]['modifications_text'].split('\n', 1)[1]))
            self.assertEqual(saved, original)

    def test_judge_rerun_preserves_narratives_and_queues_only_selected_judges(self):
        from scripts.run_paper_probes import load_dumps
        with tempfile.TemporaryDirectory(prefix='gretel-rerun-') as root:
            matrix, config, run, context, originals = self.prepare_fixture(root)
            factory = get_instance_kvargs
            def scripted_model(class_name, kwargs):
                if class_name != 'src.LLMexplaneability.huggingface.HuggingFaceLLM':
                    return factory(class_name, kwargs)
                return ScriptedLLM(kwargs['context'], {'class': 'probes_fixtures.ScriptedLLM',
                    'parameters': {'role': 'generator', 'model': kwargs['local_config']['parameters']['model']}})
            with patch('src.core.factory_base.get_instance_kvargs', side_effect=scripted_model):
                for entry in config['generators']:
                    Context._Context__global = None
                    matrix.run_generator(run, entry['id'])
            matrix.write_json(run / 'probes/old/old/old/summary.json', [{'old': True}])
            config_path = Path(root) / 'matrix-config.json'
            new = Path(root) / 'rerun'
            with patch.object(matrix.subprocess, 'check_output', return_value='0' * 40 + '\n'):
                all_judges = matrix.prepare_judge_rerun(config_path, run, Path(root) / 'all-judges')
                matrix.prepare_judge_rerun(config_path, run, new,
                    judges=['muse-glimmer-30b'], probes=['reconstruction'])
            with patch.object(matrix.subprocess, 'run', side_effect=[types.SimpleNamespace(stdout=str(i) + '\n')
                    for i in range(541500, 541503)]) as sbatch:
                matrix.submit(all_judges, include_generators=False)
            commands = [call.args[0] for call in sbatch.call_args_list]
            self.assertEqual([cmd[-3] for cmd in commands], ['matrix-judge'] * 3)
            self.assertIn('--dependency=afterany:541500', commands[1])
            self.assertIn('--dependency=afterany:541501', commands[2])
            self.assertFalse((new / 'probes').exists())
            for entry in config['generators']:
                old_files = load_dumps(run / 'generated' / entry['id'])
                new_files = load_dumps(new / 'generated' / entry['id'])
                self.assertEqual([p.read_bytes() for p, _, _ in old_files], [p.read_bytes() for p, _, _ in new_files])
            with patch.object(matrix.subprocess, 'run', return_value=types.SimpleNamespace(stdout='541400\n')) as sbatch:
                matrix.submit(new, include_generators=False)
            self.assertEqual(sbatch.call_count, 1)
            self.assertEqual(sbatch.call_args.args[0][-3:], ['matrix-judge', str(new), 'muse-glimmer-30b'])
            settings = json.loads((new / 'configs/judge-muse-glimmer-30b.json').read_text())
            settings.pop('compose_strs')
            settings['store_paths'] = context.conf['store_paths']
            matrix.write_json(new / 'configs/judge-muse-glimmer-30b.json', settings)
            judge = ScriptedLLM(context, llm('judge'))
            Context._Context__global = None
            with patch('src.core.factory_base.get_instance_kvargs', return_value=judge) as load:
                self.assertEqual(matrix.run_judge(new, 'muse-glimmer-30b'), 0)
            self.assertEqual(load.call_count, 1)
            rows, cells = matrix.collect(new)
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row['n'] == 5 and row['n_success'] == 5 for row in rows))
            self.assertTrue(all(cell['status'] == 'complete' for cell in cells))
            # Reruns are themselves reusable; corrupted narratives fail before preparation.
            with patch.object(matrix.subprocess, 'check_output', return_value='0' * 40 + '\n'):
                matrix.prepare_judge_rerun(config_path, new, Path(root) / 'rerun-again')
            file = load_dumps(run / 'generated' / config['generators'][0]['id'])[0][0]
            payload = json.loads(file.read_text())
            payload['data']['counterfactuals'][0]['input_label'] = 100
            matrix.write_json(file, payload)
            with self.assertRaisesRegex(ValueError, 'changed graph pair'):
                matrix.prepare_judge_rerun(config_path, run, Path(root) / 'bad-rerun')
            self.assertFalse((Path(root) / 'bad-rerun').exists())
            with self.assertRaisesRegex(ValueError, 'Unknown judge'):
                matrix.prepare_judge_rerun(config_path, new, Path(root) / 'unknown', judges=['unknown'])

    def test_source_validation_and_cached_judge_failure_are_visible(self):
        with tempfile.TemporaryDirectory(prefix='gretel-matrix-errors-') as root:
            matrix, config, run, context, dumps = self.prepare_fixture(root)
            with self.assertRaises(ValueError):
                matrix.validate_sources(dumps, {**config, 'num_nodes': 10})
            with self.assertRaises(ValueError):
                matrix.validate_sources(dumps + [dumps[0]], config)
            Context._Context__global = None
            with patch('src.core.factory_base.get_instance_kvargs', side_effect=FileNotFoundError('Missing cached checkpoint')):
                self.assertEqual(matrix.run_judge(run, config['judges'][0]['id']), 2)
            _, cells = matrix.collect(run)
            self.assertEqual(sum(cell['status'] == 'error' for cell in cells), 6)
            frozen = Path(json.loads((run / 'matrix.json').read_text())['sources'][0]['frozen'])
            frozen.write_text(frozen.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                matrix.frozen_dumps(run)

    def test_slurm_chain_records_ids_and_continues_after_failed_jobs(self):
        from scripts import paper_probe_matrix as matrix
        with tempfile.TemporaryDirectory(prefix='gretel-slurm-matrix-') as root:
            run = Path(root)
            config = json.loads((ROOT / matrix.DEFAULT_CONFIG).read_text())
            matrix.write_json(run / 'matrix.json', {'configuration': config})
            ids = [str(541300 + i) for i in range(9)]
            with patch.object(matrix.subprocess, 'run', side_effect=[types.SimpleNamespace(stdout=i + '\n') for i in ids]) as sbatch:
                matrix.submit(run)
                calls = [args.args[0] for args in sbatch.call_args_list]
            self.assertEqual(len(calls), 9)
            self.assertFalse(any(arg.startswith('--dependency') for arg in calls[0]))
            for previous, command in zip(ids, calls[1:]):
                self.assertIn('--dependency=afterany:' + previous, command)
            self.assertEqual([command[-3] for command in calls], ['matrix-generator'] * 6 + ['matrix-judge'] * 3)
            ledger = [json.loads(line) for line in (run / 'jobs.jsonl').read_text().splitlines()]
            self.assertEqual([row['job_id'] for row in ledger], ids)
            with self.assertRaises(ValueError):
                matrix.submit(run)


if __name__ == '__main__':
    result = unittest.main(exit=False)
    if result.result.wasSuccessful() and OPTIONS.demo_output:
        _, _, _, output, errors = integrated_run(OPTIONS.demo_output)
        Path(OPTIONS.demo_output, 'README.md').write_text(
            '# Feedback tecnico su cinque grafi TreeCycles\n\n'
            'Dataset: generazione TreeCyclesSeeded (50 grafi, 16 nodi, seed 0); cinque grafi senza cicli. '
            'Oracolo reale: TreeCyclesOracle di GRETEL. Counterfactual minimo: aggiunta di un solo arco.\n\n'
            'Generatore e giudice sono SCRIPTED: questi punteggi verificano il software, '
            'non sono risultati sperimentali o evidenza per il paper. La narrazione sintetica '
            'codifica gli edit; Recourse con narrazione aggiunge un arco diverso, il controllo rimuove un arco.\n\n'
            f'Risultati salvati: `{output}`. Dump completi sotto `output/`; grafi delle proposte nei risultati Recourse.\n'
            f'Modalità di test locale: {"light (sole importazioni delle annotazioni Dataset e initializer utils isolate)" if OPTIONS.light else "ambiente completo"}.\n')
        print('Scripted integration artifacts:', output)
    sys.exit(0 if result.result.wasSuccessful() else 1)
