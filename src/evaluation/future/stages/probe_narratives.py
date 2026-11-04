"""Generate the paper's two-part answers using the upstream narrative task.

Two record layouts. Without ``arms`` there is one record per counterfactual of
the explanation (the pilots). With ``arms`` (the paper runs) every instance
gives one record per arm, paired by the instance id of the dump:

- ``minimized``: the final counterfactual of the explainer (LBS).
- ``seed``: the generator's counterfactual before minimization (DCE).
- ``random_subset``: the input plus a random subset of the seed's edits, of the
  size of the minimized edit set. It is not a counterfactual in general, so
  only Reconstruction scores it; its only direction is ``direct``.

``directions`` adds, to ``direct`` and ``inverse``, ``renamed`` (the pair with
its shown nodes permuted, the permutation saved with the record) and
``identity`` (the direct prompt again, sampled with the generator's seed plus
``identity_seed_offset``). ``freeze_only`` builds and saves every prompt
without calling a model: the generators run later on the frozen pairs
(scripts/paper_probe_matrix.py), each in its own job.
"""
import copy
import hashlib
import json

import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.evaluation.future.stages.stage import Stage
from src.LLMexplaneability.prompt_generator import SYSTEM_PROMPT
from src.utils import probe_graph
from src.utils.probe_common import OUTPUT_SCHEMA, call_many, generation_outcome, inverse_delta
from src.utils.typed_delta import typed_delta_from_instances
from src.utils.context import clean_cfg


def seed_outcome(original, cf, seed, feature_columns=None, feature_names=None, atol=0.0):
    """How the final counterfactual relates to the explainer's seed, when one was kept.

    `minimized` is False when the minimizer returned the seed unchanged; `size`
    is the edit count of the seed, to compare with the final delta.
    """
    if seed is None:
        return None
    same = (seed.data.shape == cf.data.shape and np.array_equal(seed.data, cf.data)
            and np.array_equal(seed.node_features, cf.node_features))
    outcome = {'minimized': not same}
    try:
        outcome['size'] = typed_delta_from_instances(original, seed, feature_columns=feature_columns,
                                                     feature_names=feature_names, atol=atol)['size']
    except Exception as exc:
        outcome['size_error'] = f'{type(exc).__name__}: {exc}'
    return outcome


ARMS = ('minimized', 'seed', 'random_subset')
DIRECTIONS = ('direct', 'inverse', 'renamed', 'identity')
EDIT_FIELDS = ('nodes_added', 'nodes_removed', 'edges_added', 'edges_removed', 'features_changed')


def delta_size(delta):
    return sum(len(delta[key]) for key in EDIT_FIELDS)


def instance_rng(sample_seed, instance_id, purpose):
    """A generator that depends only on the seed, the instance and the use."""
    digest = hashlib.sha256(f'{sample_seed}:{instance_id}:{purpose}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def _with_edges(template, data, attributes):
    """A copy of template with adjacency data; edge attributes looked up per edge."""
    graph = copy.deepcopy(template)
    coords = list(zip(*np.nonzero(data)))
    graph.data = data
    graph.edge_weights = np.asarray([attributes[c][0] for c in coords], dtype=np.float32)
    width = template.edge_features.shape[1]
    graph.edge_features = np.asarray([attributes[c][1] for c in coords], dtype=np.float32).reshape((len(coords), width))
    graph._nx_repr = None
    return graph


def random_subset(original, seed_cf, size, rng, feature_columns=None):
    """The input plus `size` elementary edits drawn from the seed's, and that subset.

    Added edges take the seed's edge attributes; node-count changes are not
    sampled (the seed and the input have the same nodes on padded datasets)."""
    truth = typed_delta_from_instances(original, seed_cf, feature_columns=feature_columns)
    if truth['nodes_added'] or truth['nodes_removed']:
        raise ValueError('The seed changes the node count')
    edits = ([('edges_added', e) for e in truth['edges_added']] + [('edges_removed', e) for e in truth['edges_removed']]
             + [('features_changed', c) for c in truth['features_changed']])
    if size >= len(edits):
        raise ValueError(f'The minimized edit set ({size}) is not smaller than the seed\'s ({len(edits)})')
    chosen = sorted(rng.choice(len(edits), size=size, replace=False).tolist())
    data = original.data.copy()
    features = original.node_features.copy()
    for index in chosen:
        field, edit = edits[index]
        if field == 'features_changed':
            features[edit['node'], edit['feature']] = seed_cf.node_features[edit['node'], edit['feature']]
            continue
        u, v = edit
        data[u, v] = seed_cf.data[u, v] if field == 'edges_added' else 0
        if not original.directed:
            data[v, u] = data[u, v]
    attributes = {**probe_graph.edge_attributes(seed_cf), **probe_graph.edge_attributes(original)}
    graph = _with_edges(original, data, attributes)
    graph.node_features = features
    return graph, [edits[i] for i in chosen]


def permute(instance, permutation):
    """The same graph with node i renamed permutation[i]."""
    n = len(instance.data)
    inverse = np.empty(n, dtype=int)
    inverse[permutation] = np.arange(n)
    data = instance.data[np.ix_(inverse, inverse)]
    old = probe_graph.edge_attributes(instance)
    attributes = {(a, b): old[(int(inverse[a]), int(inverse[b]))] for a, b in zip(*np.nonzero(data))}
    graph = _with_edges(instance, data, attributes)
    graph.node_features = instance.node_features[inverse]
    return graph


def renaming(n, shown, rng):
    """A permutation of the shown nodes; hidden padding nodes keep their index."""
    permutation = np.arange(n)
    permutation[:shown] = rng.permutation(shown)
    return permutation


class ProbeNarratives(Stage):

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        p.setdefault('freeze_only', False)
        if 'generator' not in p and not p['freeze_only']:
            raise ValueError('ProbeNarratives requires a generator LLM snippet')
        p.setdefault('node_features', None)
        p.setdefault('atol', 0.0)
        p.setdefault('node_matching', 'position')
        p.setdefault('generator_family', None)
        # Padded datasets: show the LLMs only the nodes before the trailing padding.
        p.setdefault('hide_padding', False)
        p.setdefault('arms', None)
        p.setdefault('directions', ['direct', 'inverse'])
        p.setdefault('identity_seed_offset', 1)
        p.setdefault('sample_seed', 0)
        if p['node_matching'] != 'position':
            raise ValueError('A semantic node mapper has not been configured; use explicit positional matching')
        if p['arms'] is not None and (not p['arms'] or set(p['arms']) - set(ARMS) or len(set(p['arms'])) != len(p['arms'])):
            raise ValueError('arms must be distinct values among ' + ', '.join(ARMS))
        if p['arms'] is not None and 'random_subset' in p['arms'] and 'minimized' not in p['arms']:
            raise ValueError('The random-subset arm takes its size from the minimized arm')
        if set(p['directions']) - set(DIRECTIONS) or p['directions'][:2] != ['direct', 'inverse']:
            raise ValueError('directions start with direct and inverse, then renamed and/or identity')

    def init(self):
        super().init()
        # Upstream MainPipeline constructs stages twice. Delay costly loading
        # until this actual stage is used, so discarded stages reserve no GPU.
        self.generator = None

    def process(self, explanation):
        p = self.local_config['parameters']
        if self.generator is None and not p['freeze_only']:
            self.generator = get_instance_kvargs(p['generator']['class'], {'context': self.context, 'local_config': p['generator']})
        feature_map = explanation.dataset.node_features_map or {}
        if p['node_features'] is not None:
            if any(name not in feature_map for name in p['node_features']):
                raise ValueError('Unknown selected node features')
            columns = [feature_map[name] for name in p['node_features']]
        else:
            columns = None
        names = {v: k for k, v in feature_map.items()}
        original = explanation.input_instance
        input_label = self._predict(explanation, original)
        shown = probe_graph.shown_nodes(original) if p['hide_padding'] else None
        factual_text = probe_graph.graph_text(original, input_label, feature_map, shown)
        generated = (explanation.info or {}).get('generator_explanation')
        seed = generated.counterfactual_instances[0] if generated is not None and generated.counterfactual_instances else None
        system = SYSTEM_PROMPT.split('OUTPUT FORMAT:')[0] + '\n' + OUTPUT_SCHEMA + '\n' + explanation.dataset.domain
        records, prompts, owners = [], [], []
        # Do not silently lose failed counterfactual generation from the denominator.
        for arm, cf, extra in self._arms(explanation, original, seed, columns, names):
            directions = ['direct'] if arm == 'random_subset' else list(p['directions'])
            record = {'status': 'input_error', 'directions': directions}
            if arm is not None:
                record['arm'] = arm
            for direction in directions:
                record[direction + '_output'] = None
                record[direction + '_generation'] = {'status': 'not_attempted', 'call_status': 'not_attempted'}
            record.update(extra)
            records.append(record)
            if cf is None:
                record.setdefault('error', 'No counterfactual produced')
                continue
            target = self._predict(explanation, cf)
            record.update(input=probe_graph.snapshot(original), counterfactual=probe_graph.snapshot(cf),
                          input_label=input_label, target_label=target, graph_text=factual_text,
                          seed=seed_outcome(original, cf, seed, columns, names, p['atol']))
            if shown is not None:
                record['shown_nodes'] = shown
                if probe_graph.touches_hidden(original, cf, shown):
                    record.update(status='padding_edit', error='The counterfactual edits padding nodes, which the prompts do not show')
                    continue
            record['inverse_graph_text'] = probe_graph.graph_text(cf, target, feature_map, shown)
            if probe_graph.unsupported_changes(original, cf):
                record.update(status='unsupported_delta', error='Direction, graph features or existing edge values/attributes changed; TypedDelta does not describe these edits')
                continue
            if original.node_features.shape[1] != cf.node_features.shape[1]:
                record.update(status='unsupported_delta', error='Node-feature column sets differ between the two graphs')
                continue
            if arm == 'random_subset':
                record['flips_oracle'] = target != input_label
            elif target == input_label:
                record.update(status='not_counterfactual', error='The final graph does not flip the oracle')
                continue
            truth = typed_delta_from_instances(original, cf, feature_columns=columns, feature_names=names, atol=p['atol'])
            record.update(status='success', truth=truth)
            # Give the generator the states and edits; never synthesize its answer afterwards.
            prompt_truth = {key: truth[key] for key in EDIT_FIELDS}
            requests = {'direct': (factual_text, record['inverse_graph_text'], prompt_truth),
                        'inverse': (record['inverse_graph_text'], factual_text, inverse_delta(prompt_truth))}
            if 'renamed' in directions:
                n = len(original.data)
                permutation = renaming(n, n if shown is None else shown,
                                       instance_rng(p['sample_seed'], original.id, 'renaming'))
                renamed, renamed_cf = permute(original, permutation), permute(cf, permutation)
                renamed_truth = typed_delta_from_instances(renamed, renamed_cf, feature_columns=columns,
                                                           feature_names=names, atol=p['atol'])
                record['renaming'] = {'permutation': permutation.tolist(), 'truth': renamed_truth,
                                      'graph_text': probe_graph.graph_text(renamed, input_label, feature_map, shown),
                                      'inverse_graph_text': probe_graph.graph_text(renamed_cf, target, feature_map, shown)}
                requests['renamed'] = (record['renaming']['graph_text'], record['renaming']['inverse_graph_text'],
                                       {key: renamed_truth[key] for key in EDIT_FIELDS})
            for direction in directions:
                before, after, delta = requests['direct' if direction == 'identity' else direction]
                prompt = f'--- ORIGINAL ---\n{before}\n--- COUNTERFACTUAL ---\n{after}\n--- MODIFICATIONS ---\n{json.dumps(delta)}'
                offset = p['identity_seed_offset'] if direction == 'identity' else 0
                request = {'system': system, 'prompt': prompt}
                if offset:
                    request['seed_offset'] = offset
                record[direction + '_generation']['judge_request'] = request
                prompts.append((system, prompt))
                owners.append((len(records) - 1, direction, offset))
        for record in records:
            record['input_status'] = record['status']
        if not p['freeze_only']:
            base = self.generator.local_config['parameters'].get('seed', 0)
            seeds = [base + offset for _, _, offset in owners] if any(o for _, _, o in owners) else None
            answers = call_many(self.generator, prompts, seeds)
            for (index, direction, offset), answer in zip(owners, answers):
                if offset:
                    answer['judge_request']['seed_offset'] = offset
                records[index][direction + '_output'] = answer['judge_output']
                records[index][direction + '_generation'] = generation_outcome(answer, original.directed)
            for record in records:
                if record['input_status'] == 'success':
                    statuses = [record[d + '_generation']['status'] for d in record['directions']]
                    record['status'] = 'success' if all(s == 'success' for s in statuses) else 'partial_error'
        self.write_into_explanation(explanation, {
            'schema_version': 1, 'node_matching': p['node_matching'], 'feature_map': feature_map,
            'feature_columns': columns, 'atol': p['atol'], 'domain': explanation.dataset.domain,
            'generator': None if self.generator is None else copy.deepcopy(clean_cfg(self.generator.local_config)),
            'generator_family': p['generator_family'],
            'dataset': copy.deepcopy(clean_cfg(explanation.dataset.local_config)),
            'oracle': copy.deepcopy(clean_cfg(explanation.oracle.local_config)), 'counterfactuals': records,
        })
        return explanation

    def _arms(self, explanation, original, seed, columns, names):
        """(arm, counterfactual, extra record fields) for every record of the instance."""
        p = self.local_config['parameters']
        finals = explanation.counterfactual_instances or [None]
        if p['arms'] is None:
            return [(None, cf, {}) for cf in finals]
        minimized = finals[0]
        extra = {}
        if minimized is not None:
            truth = typed_delta_from_instances(original, minimized, feature_columns=columns, feature_names=names,
                                               atol=p['atol'])
            # Recourse budget K of every arm of the instance: the size of the minimized edit set.
            extra['recourse_budget'] = delta_size(truth)
        result = []
        for arm in p['arms']:
            if arm == 'minimized':
                result.append((arm, minimized, dict(extra)))
            elif arm == 'seed':
                result.append((arm, seed, dict(extra, **({} if seed is not None else {'error': 'No seed kept by the explainer'}))))
            else:
                if minimized is None or seed is None:
                    result.append((arm, None, dict(extra, error='The random subset needs the seed and the minimized counterfactual')))
                    continue
                try:
                    graph, chosen = random_subset(original, seed, extra['recourse_budget'],
                                                  instance_rng(p['sample_seed'], original.id, 'random_subset'), columns)
                except ValueError as exc:
                    result.append((arm, None, dict(extra, error=str(exc))))
                    continue
                result.append((arm, graph, dict(extra, subset_size=len(chosen))))
        return result

    @staticmethod
    def _predict(explanation, instance):
        try:
            return int(explanation.oracle.predict(instance))
        finally:
            explanation.oracle._call_counter -= 1
