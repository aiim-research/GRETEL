"""Recourse built on Alejandra's FlipRateEvaluator, with an external model.

The original prompt's alternative-edit task is reused. A JSON output suffix
aligns it with the other probes; graph application validates edits and preserves
attributes instead of using the old lossy edit_graph implementation.
"""
from src.utils.probe_common import DELTA_FIELDS, call_many, read_object, select_text, validate_delta
from src.utils.probe_graph import apply_edits, delta_nodes, snapshot, unsupported_changes
from src.utils.typed_delta import typed_delta_from_instances


class ReusedOriginalEdits(ValueError):
    """The proposal shares an elementary edit with the original counterfactual."""


def build_prompt(graph_text, modifications_text, domain, narrative, target_label):
    from src.LLMexplaneability.flip_rate_evaluation import FlipRateEvaluator
    original = FlipRateEvaluator(graph_text.rstrip() + '\n', modifications_text.rstrip() + '\n', domain.rstrip() + '\n',
                                 '' if narrative is None else 'EXPLANATION:\n' + narrative + '\n')
    system, prompt = original.flip_rate_prompt()
    # Keep the task/requirements; replace the old labeled full-vector format.
    system = system.split('**Requested output format')[0].split('6. Output')[0]
    if narrative is None:
        system = system.replace('*for the same underlying reasons stated in the explanation*', '')
        system = system.replace('1. Your edits must operationalize the causal mechanisms described in the Explanation not just arbitrary perturbations.',
                                '1. Propose edits using the factual graph and domain information.')
    system += ('\nFor this experiment return ONLY JSON with edges_added, edges_removed, '
               'features_changed (empty lists when absent). Feature changes are '
               '{"node": id, "feature": name_or_column, "from": old_value, "to": new_value}. '
               'Edges are [u,v]; their direction is that of the factual graph. '
               'Keep existing nodes. No individual edit may be reused from ORIGINAL EDITS, '
               'even as part of a larger or smaller proposal. For features this means the '
               'same node, feature column and from/to transition. '
               'Propose exactly one edit set, with no extra text.')
    prompt += f'\nTARGET ORACLE CLASS: {target_label}\n'
    return system, prompt


def _reused_edits(proposal, original, directed, feature_map=None):
    """Exact elementary edits shared with the original CF, including aliases.

    Features are identified by column and numeric transition; 0 and 0.0 match.
    A different transition on the same feature is a different edit.
    """
    feature_map = feature_map or {}

    def key(field, edit):
        if field == 'features_changed':
            column = feature_map.get(edit['feature'], edit['feature'])
            return (edit['node'], str(column), edit['from'], edit['to'])
        return tuple(edit) if field.startswith('edges_') else edit

    def normalized(delta):
        return validate_delta({k: delta.get(k, []) for k in DELTA_FIELDS},
                              directed, feature_match='transition')

    a, b = normalized(proposal), normalized(original)
    reused = {}
    for field in DELTA_FIELDS:
        original_keys = {key(field, edit) for edit in b[field]}
        overlap = [edit for edit in a[field] if key(field, edit) in original_keys]
        if overlap:
            reused[field] = overlap
    return reused


def run(judge, items, predictor, mode='full', control=True, require_structured=False,
        success='target', edge_defaults=None, preprocess=None):
    """Each item supplies the factual GraphInstance, true delta, labels, texts,
    domain and feature map. predictor receives an actual GraphInstance.
    """
    if success not in ('target', 'flip') or type(control) is not bool:
        raise ValueError('Invalid Recourse success criterion or control flag')
    records, prompts, owners = [], [], []
    for i, item in enumerate(items):
        record = {'status': None, 'with_explanation': None, 'without_explanation': None}
        records.append(record)
        try:
            text = select_text(item['output'], mode, require_structured, item['instance'].directed)
        except ValueError as exc:
            record.update(status='unparsed', error=str(exc))
            continue
        try:
            actual = int(predictor(item['instance']))
            if actual != item['input_label']:
                raise ValueError('Replay oracle disagrees with the saved factual prediction')
        except Exception as exc:
            record.update(status='oracle_error', error=f'{type(exc).__name__}: {exc}')
            continue
        for arm, narrative in [('with_explanation', text)] + ([('without_explanation', None)] if control else []):
            prompts.append(build_prompt(item['graph_text'], item['modifications_text'], item['domain'], narrative, item['target_label']))
            owners.append((i, arm))
    for (index, arm), answer in zip(owners, call_many(judge, prompts)):
        item = items[index]
        if answer['status'] == 'success':
            try:
                edits = validate_delta(read_object(answer['judge_output']), item['instance'].directed,
                                       feature_match='transition')
                answer['edits'] = edits
            except ValueError as exc:
                answer.update(status='judge_unparsed', error=str(exc))
            else:
                try:
                    reused = _reused_edits(edits, item['truth'], item['instance'].directed, item.get('feature_map'))
                    if reused:
                        answer.update(reused_edits=reused, reuse_phase='proposal')
                        raise ReusedOriginalEdits('Proposal reuses original edits')
                    shown = item.get('shown_nodes')
                    if shown is not None and any(node >= shown for node in delta_nodes(edits)):
                        raise ValueError('Proposal edits a node the prompt does not show')
                    candidate = apply_edits(item['instance'], edits, item.get('feature_map'), edge_defaults)
                    if preprocess is not None:
                        try:
                            preprocess(candidate)
                        except Exception as exc:
                            raise ValueError(f'Dataset preprocessing failed: {type(exc).__name__}: {exc}') from exc
                    if unsupported_changes(item['instance'], candidate):
                        raise NotImplementedError('Preprocessing changed direction, graph features or existing edge attributes outside the delta contract')
                    feature_map = item.get('feature_map') or {}
                    realized = typed_delta_from_instances(item['instance'], candidate,
                                                         feature_names={v: k for k, v in feature_map.items()},
                                                         feature_columns=item.get('feature_columns'), atol=item.get('atol', 0.0))
                    answer['realized_edits'] = realized
                    answer['candidate'] = snapshot(candidate)
                    if not any(realized[key] for key in ('nodes_added', 'nodes_removed', 'edges_added', 'edges_removed', 'features_changed')):
                        raise ValueError('Proposal has no measured effect after preprocessing')
                    reused = _reused_edits(realized, item['truth'], item['instance'].directed, feature_map)
                    if reused:
                        answer.update(reused_edits=reused, reuse_phase='realized')
                        raise ReusedOriginalEdits('Proposal reuses original edits after preprocessing')
                except NotImplementedError as exc:
                    answer.update(status='unsupported_edits', error=str(exc))
                except ReusedOriginalEdits as exc:
                    answer.update(status='reused_original', successful=False, error=str(exc))
                except (ValueError, IndexError, TypeError) as exc:
                    answer.update(status='invalid_proposal', error=str(exc))
                else:
                    try:
                        predicted = int(predictor(candidate))
                        answer.update(predicted_label=predicted, flipped=predicted != item['input_label'],
                                      reached_target=predicted == item['target_label'])
                        answer['successful'] = answer['reached_target'] if success == 'target' else answer['flipped']
                    except Exception as exc:
                        answer.update(status='oracle_error', error=f'{type(exc).__name__}: {exc}')
        records[index][arm] = answer
    for record in records:
        if record['status'] is not None:
            continue
        arms = [record['with_explanation']] + ([record['without_explanation']] if control else [])
        record['status'] = 'success' if all(r['status'] == 'success' for r in arms) else 'partial_error'
        if control:
            a, b = arms
            record['success_difference'] = int(a.get('successful', False)) - int(b.get('successful', False))
    return records
