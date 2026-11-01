"""The same saved-record interface for in-pipeline and offline probes."""
import json

from src.utils import reconstruction_probe, reversal_probe, recourse_probe
from src.utils.probe_graph import restore
from src.utils.probe_common import counts, check_judge, saved_generation_outcomes, validate_delta

NARRATIVES_STAGE = 'src.evaluation.future.stages.probe_narratives.ProbeNarratives'
PROBE_PROTOCOL_VERSION = 6  # Quoted node indices ("22") parse as integers.


def run_saved(name, judge, saved, context='off', mode='dict', feature_match='transition',
              control=True, predictor=None, dataset=None, success='target', edge_defaults=None, preprocess=None):
    if name not in ('reconstruction', 'reversal', 'recourse') or context not in ('on', 'off'):
        raise ValueError('Invalid probe or context')
    if mode not in ('full', 'dict') or feature_match not in ('identity', 'transition'):
        raise ValueError('Invalid mode or feature matching')
    if saved.get('schema_version') != 1 or saved.get('node_matching') != 'position':
        raise ValueError('Unknown first-pass schema or node-matching contract')
    check_judge(getattr(judge, 'local_config', {}), saved)
    if name == 'recourse' and (predictor is None or context != 'off'):
        raise ValueError('Recourse needs an oracle; CTX is not its no-explanation control')
    source = saved['counterfactuals']
    if not isinstance(source, list) or not source:
        raise ValueError('Missing first-pass records')
    records, items, pending = [], [], []
    for index, item in enumerate(source):
        records.append(None)
        if not isinstance(item, dict):
            records[index] = {'status': 'input_error', 'error': 'A first-pass record must be an object'}
            continue
        input_status = item.get('input_status', item.get('status'))
        if input_status != 'success':
            records[index] = {'status': 'input_error', 'input_status': input_status, 'error': item.get('error'),
                              'generation_outcomes': saved_generation_outcomes(item)}
            continue
        try:
            directed = item['input']['directed']
            if type(directed) is not bool:
                raise ValueError('Invalid saved graph direction')
            outcomes = saved_generation_outcomes(item)
            required = ('direct', 'inverse') if name == 'reversal' else ('direct',)
            failed = [d for d in required if outcomes[d]['status'] != 'success']
            if failed:
                records[index] = {
                    'status': 'model_error' if any(outcomes[d]['status'] == 'model_error' for d in failed) else 'unparsed',
                    'error_origin': 'generator', 'generation_failed_directions': failed,
                    'error': '; '.join(f"{d}: {outcomes[d].get('error', outcomes[d]['status'])}" for d in failed),
                    'generation_outcomes': outcomes, 'input_status': input_status,
                }
                continue
            truth = None
            if name != 'reversal':
                truth = validate_delta({k: v for k, v in item['truth'].items() if k != 'size'},
                                       directed, feature_match='transition')
            if name == 'reconstruction':
                inputs = (item['direct_output'], item['graph_text'], truth, directed)
            elif name == 'reversal':
                inputs = (item['direct_output'], item['inverse_output'], item['graph_text'], item['inverse_graph_text'], directed)
            else:
                original = restore(item['input'], dataset)
                cf = restore(item['counterfactual'], dataset)
                try:
                    target = int(predictor(cf))
                except Exception as exc:
                    records[index] = {'status': 'oracle_error', 'error': f'{type(exc).__name__}: {exc}'}
                    continue
                if target != item['target_label']:
                    records[index] = {'status': 'oracle_error', 'error': 'Replay oracle disagrees with the saved target prediction'}
                    continue
                inputs = {'instance': original, 'truth': truth, 'input_label': item['input_label'],
                          'target_label': item['target_label'], 'output': item['direct_output'],
                          'graph_text': item['graph_text'], 'modifications_text': 'ORIGINAL EDITS:\n' + json.dumps(truth),
                          'domain': saved['domain'], 'feature_map': saved['feature_map'],
                          'feature_columns': saved.get('feature_columns'), 'atol': saved.get('atol', 0.0),
                          'shown_nodes': item.get('shown_nodes')}
        except Exception as exc:
            records[index] = {'status': 'input_error', 'error': f'{type(exc).__name__}: {exc}'}
            continue
        items.append(inputs)
        pending.append(index)
    if name == 'reconstruction':
        evaluated = reconstruction_probe.run(judge, items, context == 'on', mode, feature_match, True)
    elif name == 'reversal':
        evaluated = reversal_probe.run(judge, items, context == 'on', mode, feature_match, True)
    else:
        evaluated = recourse_probe.run(judge, items, predictor, mode, control, True, success, edge_defaults, preprocess)
    if len(evaluated) != len(pending):
        raise RuntimeError('Probe returned a different number of results than inputs')
    for index, result in zip(pending, evaluated):
        result.update(input_status='success', generation_outcomes=saved_generation_outcomes(source[index]),
                      seed=source[index].get('seed'))
        records[index] = result
    # Keep the generation states even when replay or input validation failed.
    for item, record in zip(source, records):
        if isinstance(item, dict):
            record.setdefault('input_status', item.get('input_status', item.get('status')))
            record.setdefault('generation_outcomes', saved_generation_outcomes(item))
            record.setdefault('seed', item.get('seed'))
    return records


def summarize(name, records):
    result = counts(records)
    result['generation_status_counts'] = {
        direction: counts([r['generation_outcomes'][direction] for r in records if 'generation_outcomes' in r])['status_counts']
        for direction in ('direct', 'inverse')
    }
    n = len(records)
    field = 'scores' if name == 'reconstruction' else 'structural_scores'
    if name in ('reconstruction', 'reversal'):
        scored = [r[field] for r in records if r.get(field) is not None]
        result['n_scored'] = len(scored)
        for edit_type in ('edges', 'edges_added', 'edges_removed', 'features_changed', 'nodes_added', 'nodes_removed'):
            for metric in ('precision', 'recall', 'f1', 'jaccard'):
                values = [score[edit_type][metric] for score in scored]
                result[f'{edit_type}_{metric}_valid_mean'] = sum(values) / len(values) if values else None
        if name == 'reversal':
            semantic = [r['semantic'] for r in records if r.get('semantic') is not None]
            result['n_semantic_valid'] = sum(r['status'] == 'success' for r in semantic)
            result['semantic_yes_count'] = sum(r.get('consistent', False) for r in semantic)
    else:
        for arm in ('with_explanation', 'without_explanation'):
            attempts = [r[arm] for r in records if r.get(arm) is not None]
            if not attempts and arm == 'without_explanation':
                continue
            result[arm + '_n_valid'] = sum(r['status'] == 'success' for r in attempts)
            result[arm + '_n_successful'] = sum(r.get('successful', False) for r in attempts)
            # Attempt rates include errors; raw status counts keep them distinguishable.
            result[arm + '_success_rate_all_attempts'] = sum(r.get('successful', False) for r in attempts) / n if n else None
            valid = result[arm + '_n_valid']
            result[arm + '_success_rate_valid'] = result[arm + '_n_successful'] / valid if valid else None
            result[arm + '_status_counts'] = counts(attempts)['status_counts']
            result[arm + '_n_reused_original'] = sum(r['status'] == 'reused_original' for r in attempts)
        if 'without_explanation_success_rate_all_attempts' in result and n:
            result['success_rate_difference_all_attempts'] = (result['with_explanation_success_rate_all_attempts']
                                                              - result['without_explanation_success_rate_all_attempts'])
    return result
