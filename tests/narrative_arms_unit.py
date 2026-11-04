#!/usr/bin/env python3
"""The paper path on top of the probes: arms, renaming, identity, Consistency
and the Recourse budget, through real stages, factories and TreeCyclesOracle.

The explainer and the LLMs are scripted. The explainer returns a seed of three
added edges and a minimized counterfactual of one, as GenerateMinimize stores
them. Run from the repository root:

    python tests/narrative_arms_unit.py
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import numpy as np  # noqa: E402
from probes_fixtures import DatasetFixture, ScriptedLLM, delta, llm  # noqa: E402
from src.core.explainer_base import Explainer  # noqa: E402
from src.core.factory_base import get_instance_kvargs  # noqa: E402
from src.evaluation.future.stages.main_pipeline import MainPipeline  # noqa: E402
from src.evaluation.future.stages.probe_narratives import permute, random_subset, renaming, instance_rng  # noqa: E402
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation  # noqa: E402
from src.oracle.oracle_factory import OracleFactory  # noqa: E402
from src.utils import consistency_probe, recourse_probe  # noqa: E402
from src.utils.probe_graph import apply_edits, restore  # noqa: E402
from src.utils.probe_generation import regenerate_saved  # noqa: E402
from src.utils.probe_inputs import NARRATIVES_STAGE, run_saved, summarize  # noqa: E402
from src.utils.typed_delta import typed_delta_from_instances  # noqa: E402
from src.utils.context import Context  # noqa: E402
from paper_probes_unit import context_at  # noqa: E402

DIRECTIONS = ['direct', 'inverse', 'renamed', 'identity']


class SeedMinimizeFixture(Explainer):
    """A seed with three added edges and its minimized version with the first one."""

    def init(self):
        pass

    def explain(self, instance):
        n = len(instance.data)
        free = [[u, v] for u in range(n) for v in range(u + 1, n) if not instance.data[u, v]][:3]
        seed = apply_edits(instance, delta(added=free))
        minimized = apply_edits(instance, delta(added=free[:1]))
        explanation = LocalGraphCounterfactualExplanation(self.context, self.dataset, self.oracle, self, instance, [minimized])
        explanation._info['generator_explanation'] = LocalGraphCounterfactualExplanation(
            self.context, self.dataset, self.oracle, self, instance, [seed])
        return explanation


class SeededGenerator(ScriptedLLM):
    """The scripted generator, keeping the seed it was asked to sample each prompt with."""

    def explain_many(self, prompts, seeds=None):
        self.seeds = list(seeds) if seeds is not None else [None] * len(prompts)
        return [self.explain_counterfactual(*item) for item in prompts]


class AlsoConsistency:
    """Answers the Consistency prompts; everything else as the scripted LLM it is mixed into."""

    def explain_counterfactual(self, system, prompt):
        if '--- RELATION ---' in prompt:
            return json.dumps({'mechanism_a': 'a cycle appears', 'mechanism_b': 'a cycle appears', 'verdict': 'compatible'})
        return super().explain_counterfactual(system, prompt)


class PaperGenerator(AlsoConsistency, SeededGenerator):
    pass


class PaperJudge(AlsoConsistency, ScriptedLLM):
    pass


class ConsistencyJudge:
    local_config = {'parameters': {'model': 'scripted-consistency'}}

    def __init__(self, verdict='compatible'):
        self.verdict, self.prompts = verdict, []

    def explain_counterfactual(self, system, prompt):
        self.prompts.append(prompt)
        mechanism = None if self.verdict == 'no_mechanism' else 'a cycle appears'
        return json.dumps({'mechanism_a': 'a cycle appears', 'mechanism_b': mechanism, 'verdict': self.verdict})


def frozen_pairs(root, samples=3, dump=False):
    context = context_at(root)
    dataset = DatasetFixture(context, {'class': 'probes_fixtures.DatasetFixture', 'parameters': {}})
    oracle = OracleFactory(context).get_oracle({'class': 'src.oracle.custom.oracle_tree_cycles.TreeCyclesOracle', 'parameters': {}}, dataset)
    explainer = SeedMinimizeFixture(context, {'class': 'narrative_arms_unit.SeedMinimizeFixture', 'parameters': {'fold_id': 0},
                                              'dataset': dataset, 'oracle': oracle})
    stage = {'class': NARRATIVES_STAGE, 'parameters': {'freeze_only': True, 'arms': ['minimized', 'seed', 'random_subset'],
                                                       'directions': DIRECTIONS}}
    pipeline = MainPipeline(context, {'class': 'src.evaluation.future.stages.main_pipeline.MainPipeline', 'parameters': {'stages': [
        {'class': 'src.evaluation.future.stages.runtime.Runtime', 'parameters': {}}, stage]
        + ([{'class': 'src.evaluation.future.stages.probe_dump.ProbeDump', 'parameters': {}}] if dump else [])}})
    saved = []
    for instance in [i for i in dataset.instances if i.label == 0][:samples]:
        explanation = LocalGraphCounterfactualExplanation(context, dataset, oracle, explainer, instance, [])
        saved.append(pipeline.process(explanation).stages_info[NARRATIVES_STAGE])
    return context, dataset, oracle, saved


class GraphOperations(unittest.TestCase):
    def setUp(self):
        self.g = restore({'id': 0, 'label': 0, 'directed': False, 'data': [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                          'node_features': [[1.0], [2.0], [3.0], [4.0]], 'edge_features': [[1.0]] * 6,
                          'edge_weights': [1.0] * 6, 'graph_features': None})

    def test_permutation_renames_nodes_and_keeps_the_structure(self):
        permutation = np.array([2, 0, 3, 1])
        p = permute(self.g, permutation)
        for u, v in zip(*np.nonzero(self.g.data)):
            self.assertEqual(p.data[permutation[u], permutation[v]], 1)
        self.assertEqual(np.count_nonzero(p.data), np.count_nonzero(self.g.data))
        np.testing.assert_array_equal(p.node_features[permutation], self.g.node_features)
        self.assertEqual(p.edge_weights.shape, (6,))

    def test_renaming_keeps_hidden_padding_in_place(self):
        permutation = renaming(6, 4, instance_rng(0, 'x', 'renaming'))
        self.assertEqual(sorted(permutation[:4]), [0, 1, 2, 3])
        self.assertEqual(list(permutation[4:]), [4, 5])
        self.assertEqual(permutation.tolist(), renaming(6, 4, instance_rng(0, 'x', 'renaming')).tolist())

    def test_random_subset_draws_edits_of_the_seed(self):
        seed = apply_edits(self.g, delta(added=[[0, 2], [0, 3], [1, 3]]))
        graph, chosen = random_subset(self.g, seed, 2, instance_rng(0, 0, 'random_subset'))
        truth = typed_delta_from_instances(self.g, graph)
        self.assertEqual(len(chosen), 2)
        self.assertTrue(set(map(tuple, truth['edges_added'])) <= {(0, 2), (0, 3), (1, 3)})
        self.assertEqual(truth['size']['edges_added'], 2)
        with self.assertRaises(ValueError):
            random_subset(self.g, seed, 3, instance_rng(0, 0, 'random_subset'))


class DeletedNodes(unittest.TestCase):
    """Protocol v2: a counterfactual materialized without the nodes its edits isolated comes back in the
    input's frame, with those nodes as nodes_removed and out of the prompt text."""

    def test_lift_truth_and_text(self):
        from src.evaluation.future.stages.probe_narratives import frame_truth
        from src.utils import probe_graph
        original = restore({'id': 0, 'label': 0, 'directed': False, 'data': [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                            'node_features': [[1.0], [2.0], [3.0], [4.0]], 'edge_features': [[1.0]] * 6,
                            'edge_weights': [1.0] * 6, 'graph_features': None})
        materialized = restore({'id': 0, 'label': 1, 'directed': False, 'data': [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                                'node_features': [[1.0], [2.0], [9.0]], 'edge_features': [[1.0]] * 6,
                                'edge_weights': [1.0] * 6, 'graph_features': None})
        frame = probe_graph.lift(original, materialized, [0, 1, 2, -1])
        self.assertEqual(frame.data.shape, (4, 4))
        removed = probe_graph.deleted_nodes(original, frame)
        self.assertEqual(removed, [3])
        truth = frame_truth(original, frame, removed, None, None, 0.0)
        self.assertEqual(truth['nodes_removed'], [3])
        self.assertEqual(truth['edges_removed'], [[2, 3]])
        self.assertEqual(truth['edges_added'], [[0, 2]])
        self.assertEqual([c['node'] for c in truth['features_changed']], [2])
        text = probe_graph.graph_text(frame, 1, {}, nodes=[0, 1, 2])
        self.assertIn('Nodes: [0, 1, 2]', text)
        self.assertIsNone(probe_graph.lift(original, original, [0, 1, -1, -1]))


class ArmsAndProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.context, cls.dataset, cls.oracle, cls.saved = frozen_pairs(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_freeze_only_saves_three_arms_with_prompts_and_no_answers(self):
        for saved in self.saved:
            self.assertIsNone(saved['generator'])
            records = {r['arm']: r for r in saved['counterfactuals']}
            self.assertEqual(set(records), {'minimized', 'seed', 'random_subset'})
            self.assertTrue(all(r['status'] == 'success' for r in records.values()))
            self.assertEqual(records['minimized']['recourse_budget'], 1)
            self.assertEqual(records['seed']['truth']['size']['edges_added'], 3)
            self.assertEqual(records['random_subset']['directions'], ['direct'])
            self.assertEqual(records['random_subset']['truth']['size']['edges_added'], 1)
            for record in records.values():
                for direction in record['directions']:
                    self.assertIsNone(record[direction + '_output'])
                    self.assertIn('prompt', record[direction + '_generation']['judge_request'])
            minimized = records['minimized']
            self.assertEqual(minimized['identity_generation']['judge_request']['prompt'],
                             minimized['direct_generation']['judge_request']['prompt'])
            self.assertEqual(minimized['identity_generation']['judge_request']['seed_offset'], 1)
            self.assertNotIn('seed_offset', minimized['direct_generation']['judge_request'])
            permutation = minimized['renaming']['permutation']
            (u, v), = minimized['truth']['edges_added']
            self.assertEqual(minimized['renaming']['truth']['edges_added'], [sorted([permutation[u], permutation[v]])])

    def generated(self):
        generator = SeededGenerator(self.context, llm('generator'))
        generator.local_config['parameters']['seed'] = 7
        return generator, [regenerate_saved(generator, saved, 'scripted-generator') for saved in self.saved]

    def test_generators_answer_every_direction_and_identity_uses_another_seed(self):
        generator, generated = self.generated()
        offsets = [r[d + '_generation']['judge_request'].get('seed_offset', 0)
                   for r in generated[-1]['counterfactuals'] for d in r['directions']]
        self.assertEqual(generator.seeds, [7 + offset for offset in offsets])
        for data in generated:
            for record in data['counterfactuals']:
                self.assertEqual(record['status'], 'success')
                self.assertTrue(all(record[d + '_output'] for d in record['directions']))

    def test_consistency_tests_skip_the_random_subset_and_pass_the_mapping(self):
        _, generated = self.generated()
        for test in ('inversion', 'renaming', 'identity'):
            judge = ConsistencyJudge()
            records = [r for data in generated for r in run_saved('consistency', judge, data, test=test,
                                                                    arms=['seed', 'minimized'])]
            summary = summarize('consistency', records)
            self.assertEqual(summary['n'], 2 * len(generated))
            self.assertEqual(summary['consistent_rate_valid'], 1.0)
            self.assertEqual(sum(r['status'] == 'not_applicable' for r in records), len(generated))
            self.assertEqual('NODE MAPPING' in judge.prompts[0], test == 'renaming')
        judge = ConsistencyJudge('no_mechanism')
        record = run_saved('consistency', judge, generated[0], test='identity')[0]
        self.assertEqual((record['verdict'], record['consistent']), ('no_mechanism', False))

    def test_consistency_parsing_is_strict(self):
        for output in ('compatible', json.dumps({'verdict': 'compatible'}),
                       json.dumps({'mechanism_a': 'x', 'mechanism_b': None, 'verdict': 'compatible'}),
                       json.dumps({'mechanism_a': 'x', 'mechanism_b': 'y', 'verdict': 'maybe'})):
            with self.subTest(output=output), self.assertRaises(ValueError):
                consistency_probe.parse(output)

    def test_self_evaluation_is_explicit(self):
        _, generated = self.generated()
        judge = ConsistencyJudge()
        judge.local_config = {'parameters': {'model': 'scripted-generator'}}
        with self.assertRaises(ValueError):
            run_saved('consistency', judge, generated[0], test='inversion')
        self.assertEqual(run_saved('consistency', judge, generated[0], test='inversion',
                                   self_evaluation=True)[0]['status'], 'success')
        with self.assertRaises(ValueError):
            run_saved('reconstruction', judge, generated[0], self_evaluation=True)

    def test_recourse_budget_is_stated_and_enforced(self):
        g = self.saved[0]['counterfactuals'][0]
        original = restore(g['input'])
        system, prompt = recourse_probe.build_prompt('graph', 'edits', 'domain', 'narrative', 1, budget=2)
        self.assertIn('BUDGET: at most 2 elementary edits', prompt)
        self.assertIn('at most K = 2 elementary edits', system)
        for text in (system, recourse_probe.build_prompt('graph', 'edits', 'domain', None, 1)[0]):
            self.assertNotIn('{{', text)
            self.assertNotIn("G'", text)
        n = len(original.data)
        free = [[u, v] for u in range(n) for v in range(u + 1, n) if not original.data[u, v]]
        known = {tuple(e) for e in g['truth']['edges_added']}
        free = [e for e in free if tuple(e) not in known]

        class Proposer:
            local_config = {'parameters': {'model': 'scripted-proposer'}}

            def __init__(self, edits):
                self.edits = edits

            def explain_counterfactual(self, system, prompt):
                return json.dumps(delta(added=self.edits))

        item = {'instance': original, 'truth': g['truth'], 'input_label': 0, 'target_label': 1,
                'output': json.dumps({'edits': delta(), 'Natural_Language_Explanation': 'a cycle'}),
                'graph_text': g['graph_text'], 'modifications_text': 'edits', 'domain': 'd', 'feature_map': {},
                'budget': 1}
        predict = lambda graph: int(self.oracle.predict(graph))
        over = recourse_probe.run(Proposer(free[:2]), [item], predict, mode='dict', control=False)[0]
        self.assertEqual(over['with_explanation']['status'], 'over_budget')
        self.assertFalse(over['with_explanation']['successful'])
        within = recourse_probe.run(Proposer(free[:1]), [item], predict, mode='dict', control=False)[0]
        self.assertEqual(within['with_explanation']['status'], 'success')


class PaperMatrix(unittest.TestCase):
    """prepare, generators, primary judge, budget sweep on a source subset, self-evaluation, summary."""

    def test_matrix_over_arms(self):
        from scripts import paper_probe_matrix as matrix
        with tempfile.TemporaryDirectory(prefix='gretel-narr-') as root:
            context, *_ = frozen_pairs(Path(root) / 'fixture', samples=3, dump=True)
            model = lambda name, family: {'id': name, 'family': family, 'parameters': {'model': 'scripted-' + name}}
            # MODE-full would need a scripted judge that reads the raw answer; the fixture reads narratives only.
            probes = {'reconstruction': {'conditions': [{'context': c, 'mode': 'dict'} for c in ('off', 'on')]},
                      'consistency': {'arms': ['seed', 'minimized'],
                                      'conditions': [{'test': t} for t in ('inversion', 'renaming', 'identity')]},
                      'recourse': {'arms': ['seed', 'minimized'], 'conditions': [{'budget': 'record'}]}}
            config = {'scope': 'narr-test', 'arms': ['minimized', 'seed', 'random_subset'], 'model_defaults': {},
                      'slurm': {'node': 'n', 'time': '1:00:00', 'mem': '1G', 'min_free_mb': 1, 'max_wait': 1},
                      'generators': [model('gen-a', 'qwen'), model('gen-b', 'llama')],
                      'judges': [model('judge', 'gpt-oss'),
                                 {**model('sweep', 'gpt-oss'), 'source_filter': ['no-such-scope'],
                                  'probes': {'recourse': {'arms': ['minimized'], 'conditions': [{'budget': k} for k in (1, 2)]}}}],
                      'probes': probes,
                      'self_evaluation': {'probes': {'consistency': {'arms': ['seed', 'minimized'], 'conditions': [{'test': 'inversion'}]}}}}
            path = Path(root) / 'matrix-config.json'
            path.write_text(json.dumps(config))
            with patch.object(matrix.subprocess, 'check_output', return_value='0' * 40 + '\n'):
                run = matrix.prepare(path, [Path(root) / 'fixture/output'], Path(root) / 'matrix')
            for config_path in (run / 'configs').glob('*.json'):
                settings = json.loads(config_path.read_text())
                settings.pop('compose_strs')
                settings['store_paths'] = context.conf['store_paths']
                config_path.write_text(json.dumps(settings))
            self.assertEqual(sorted(p.name for p in (run / 'configs').glob('self-*')), ['self-gen-a.json', 'self-gen-b.json'])
            factory = get_instance_kvargs

            def scripted(class_name, kwargs):
                if class_name != 'src.LLMexplaneability.huggingface.HuggingFaceLLM':
                    return factory(class_name, kwargs)
                name = kwargs['local_config']['parameters']['model']
                role = 'generator' if name.startswith('scripted-gen') else 'judge'
                cls = PaperGenerator if role == 'generator' else PaperJudge
                return cls(kwargs['context'], {'class': 'probes_fixtures.ScriptedLLM', 'parameters': {'role': role, 'model': name}})
            with patch('src.core.factory_base.get_instance_kvargs', side_effect=scripted):
                for tag in ('gen-a', 'gen-b'):
                    Context._Context__global = None
                    matrix.run_generator(run, tag)
                for tag in ('judge', 'sweep'):
                    Context._Context__global = None
                    self.assertEqual(matrix.run_judge(run, tag), 0)
                for tag in ('gen-a', 'gen-b'):
                    Context._Context__global = None
                    self.assertEqual(matrix.run_self(run, tag), 0)
            rows, cells = matrix.collect(run)
            self.assertTrue(all(cell['status'] == 'complete' for cell in cells), cells)
            primary = [r for r in rows if r['judge'] == 'judge']
            self.assertEqual(len(primary), 2 * (2 + 3 + 1))
            for row in primary:
                expected = 9 if row['probe'] == 'reconstruction' else 6
                self.assertEqual((row['n'], row['n_success']), (expected, expected), row)
            consistency = [r for r in primary if r['probe'] == 'consistency']
            self.assertTrue(all(r['consistent_rate_valid'] == 1.0 for r in consistency))
            self.assertTrue(all(r['n'] == 0 for r in rows if r['judge'] == 'sweep'))
            own = [r for r in rows if r['judge'] == 'self']
            self.assertEqual([(r['probe'], r['n_success']) for r in own], [('consistency', 6)] * 2)
            with patch.object(matrix.subprocess, 'run') as sbatch:
                sbatch.return_value.stdout = '123\n'
                matrix.submit(run, only=['judge:sweep', 'self:gen-a'], after='99')
                commands = [call.args[0] for call in sbatch.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertIn('--dependency=afterany:99', commands[0])
            self.assertEqual(commands[1][-3:], ['matrix-self', str(run), 'gen-a'])
            with self.assertRaises(ValueError):
                matrix.submit(run, only=['judge:sweep'])


if __name__ == '__main__':
    unittest.main()
