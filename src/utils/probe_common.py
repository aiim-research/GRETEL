"""Shared data and model-call contracts for the paper's independent probes.

No dataset rules belong here. Node identifiers use GRETEL's matrix indices;
that positional assumption is recorded by the first pass, not a graph matcher.
"""
import ast
import json
import math
import re

DELTA_FIELDS = ('nodes_added', 'nodes_removed', 'edges_added', 'edges_removed', 'features_changed')
CORE_FIELDS = ('edges_added', 'edges_removed', 'features_changed')
OUTPUT_SCHEMA = ('Return one JSON object with two fields: "edits" and '
                 '"Natural_Language_Explanation". "edits" contains nodes_added, '
                 'nodes_removed, edges_added, edges_removed and features_changed. '
                 'Use lists for each field; edges are [u,v] pairs and feature changes '
                 'are {"node": id, "feature": name, "from": value, "to": value}. '
                 'Use empty lists for unchanged types. Declare the edits yourself; '
                 'put your explanation in Natural_Language_Explanation.')


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'Duplicate JSON field: {key}')
        value[key] = item
    return value


def read_object(text, allow_literal=False):
    """JSON for judges; optionally thesis-style infos/Python literal for generators.

    Accept one fenced object or an object surrounded by prose. Never fill in
    missing content, discard malformed edits, or choose between multiple blocks.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Missing model output')
    blocks = re.findall(r'```(?:json|python)?\s*(.*?)```', text, re.DOTALL)
    if len(blocks) > 1:
        raise ValueError('Ambiguous output: multiple code blocks')
    candidate = blocks[0] if blocks else text
    start, end = candidate.find('{'), candidate.rfind('}')
    if start < 0 or end < start:
        raise ValueError('No dictionary found')
    candidate = candidate[start:end + 1]
    try:
        obj = json.loads(candidate, object_pairs_hook=_unique_pairs)
    except json.JSONDecodeError as exc:
        if not allow_literal:
            raise ValueError('Invalid JSON') from exc
        try:
            tree = ast.parse(candidate, mode='eval')
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    keys = [ast.literal_eval(key) for key in node.keys]
                    if len(set(keys)) != len(keys):
                        raise ValueError('Duplicate literal dictionary field')
            obj = ast.literal_eval(tree)
        except (SyntaxError, ValueError, TypeError) as literal_exc:
            raise ValueError('Invalid dictionary literal') from literal_exc
    if not isinstance(obj, dict):
        raise ValueError('Expected a dictionary')
    return obj


def node_index(value):
    # JSON-quoted indices ("22") name the same node; any other string is rejected.
    if type(value) is str and re.fullmatch('[0-9]+', value):
        value = int(value)
    if type(value) is not int or value < 0:
        raise ValueError('Node indices must be nonnegative integers')
    return value


def scalar(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Feature values must be finite numbers')
    return value


def validate_delta(delta, directed=False, strict=True, feature_match='identity'):
    if feature_match not in ('identity', 'transition'):
        raise ValueError('feature_match must be identity or transition')
    if not isinstance(delta, dict) or not any(key in delta for key in DELTA_FIELDS):
        raise ValueError('Expected a typed delta, not an empty/unrelated dictionary')
    if strict and not all(key in delta for key in CORE_FIELDS):
        raise ValueError('Missing required delta fields')
    if set(delta) - set(DELTA_FIELDS):
        raise ValueError('Unknown delta fields')
    result = {key: [] for key in DELTA_FIELDS}
    for key in DELTA_FIELDS:
        items = delta.get(key, [])
        if not isinstance(items, list):
            raise ValueError(f'{key} must be a list')
        seen = set()
        for item in items:
            if key.startswith('nodes_'):
                normalized = node_index(item)
                identity = normalized
            elif key.startswith('edges_'):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError('Each edge must contain two indices')
                u, v = map(node_index, item)
                normalized = [u, v] if directed else [min(u, v), max(u, v)]
                identity = tuple(normalized)
            else:
                if not isinstance(item, dict) or not {'node', 'feature'} <= set(item):
                    raise ValueError('Feature changes need node and feature')
                if set(item) - {'node', 'feature', 'from', 'to'}:
                    raise ValueError('Unknown feature-change fields')
                feature = item['feature']
                if isinstance(feature, bool) or not isinstance(feature, (str, int)) or str(feature) == '':
                    raise ValueError('Feature must be a name or column index')
                normalized = {'node': node_index(item['node']), 'feature': feature}
                if ('from' in item) != ('to' in item):
                    raise ValueError('Feature changes need both from and to')
                if 'from' in item:
                    normalized.update({'from': scalar(item['from']), 'to': scalar(item['to'])})
                    if normalized['from'] == normalized['to']:
                        raise ValueError('A feature transition must change its value')
                elif feature_match == 'transition':
                    raise ValueError('Transition matching needs feature from/to values')
                identity = (normalized['node'], str(feature))
            if identity in seen:
                raise ValueError(f'Duplicate edit in {key}')
            seen.add(identity)
            result[key].append(normalized)
    if set(result['nodes_added']) & set(result['nodes_removed']):
        raise ValueError('Conflicting node edits')
    if {tuple(e) for e in result['edges_added']} & {tuple(e) for e in result['edges_removed']}:
        raise ValueError('Conflicting edge edits')
    return result


def inverse_delta(delta):
    result = {key: list(delta.get(key, [])) for key in DELTA_FIELDS}
    for prefix in ('nodes', 'edges'):
        result[prefix + '_added'], result[prefix + '_removed'] = result[prefix + '_removed'], result[prefix + '_added']
    result['features_changed'] = [dict(change, **{'from': change['to'], 'to': change['from']})
                                  if 'from' in change and 'to' in change else dict(change)
                                  for change in result['features_changed']]
    return result


def narrative_field(raw):
    try:
        obj = read_object(raw, allow_literal=True)
    except ValueError:
        return None
    for key in ('Natural_Language_Explanation', 'narrative'):
        if isinstance(obj.get(key), str) and obj[key].strip():
            return obj[key]
    return None


def select_text(raw, mode='full', require_structured=False, directed=False):
    if mode not in ('dict', 'full'):
        raise ValueError('mode must be dict or full')
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('Missing generator response')
    narrative = narrative_field(raw)
    if require_structured:
        obj = read_object(raw, allow_literal=True)
        validate_delta(obj.get('edits'), directed, feature_match='transition')
        if narrative is None:
            raise ValueError('Missing narrative in structured generator response')
    if mode == 'dict' and narrative is None:
        raise ValueError('Missing narrative field')
    return raw if mode == 'full' else narrative


def call_many(model, prompts):
    """Exactly one record per prompt, including call and cardinality errors."""
    if not prompts:
        return []
    if hasattr(model, 'explain_many'):
        try:
            answers = model.explain_many(prompts)
            if not isinstance(answers, (list, tuple)) or len(answers) != len(prompts):
                raise ValueError('Backend returned a different number of answers than prompts')
            return [dict(_answer_record(answer), judge_request={'system': system, 'prompt': prompt})
                    for answer, (system, prompt) in zip(answers, prompts)]
        except Exception as exc:
            return [{'status': 'model_error', 'judge_output': None,
                     'error': f'{type(exc).__name__}: {exc}', 'judge_request': {'system': system, 'prompt': prompt}}
                    for system, prompt in prompts]
    records = []
    for system, prompt in prompts:
        try:
            records.append(_answer_record(model.explain_counterfactual(system=system, prompt=prompt)))
        except Exception as exc:
            records.append({'status': 'model_error', 'judge_output': None,
                            'error': f'{type(exc).__name__}: {exc}'})
        records[-1]['judge_request'] = {'system': system, 'prompt': prompt}
    return records


def generation_outcome(answer, directed=False):
    """Retain the backend outcome separately from structured-answer validity."""
    result = dict(answer)
    result.setdefault('call_status', result['status'])
    if result['status'] == 'success':
        try:
            select_text(result.get('judge_output'), 'full', True, directed)
        except ValueError as exc:
            result.update(status='unparsed', error=str(exc))
    return result


def saved_generation_outcomes(item):
    """Read new direction states, deriving them for older schema-1 dumps."""
    outcomes = {}
    graph = item.get('input')
    directed = graph.get('directed', False) if isinstance(graph, dict) else False
    input_status = item.get('input_status', item.get('status'))
    for direction in ('direct', 'inverse'):
        saved = item.get(direction + '_generation')
        if saved is None:
            answer = {'status': 'success' if input_status == 'success' else 'not_attempted'}
        elif not isinstance(saved, dict) or 'status' not in saved:
            answer = {'status': 'model_error', 'error': 'Invalid saved generation outcome'}
        else:
            answer = dict(saved)
            # Validity is re-derived from the saved text with the current parser.
            if answer.get('call_status') == 'success':
                answer['status'] = 'success'
                answer.pop('error', None)
        answer['judge_output'] = item.get(direction + '_output')
        result = generation_outcome(answer, directed)
        outcomes[direction] = {key: result[key] for key in ('status', 'call_status', 'error') if key in result}
    return outcomes


def _answer_record(answer):
    if not isinstance(answer, str):
        return {'status': 'model_error', 'judge_output': None, 'error': 'Backend did not return text'}
    return {'status': 'success', 'judge_output': str(answer),
            **getattr(answer, 'llm_metadata', {})}


def counts(records):
    statuses = {status: sum(r['status'] == status for r in records)
                for status in sorted({r['status'] for r in records})}
    return {'n': len(records), 'n_success': statuses.get('success', 0), 'status_counts': statuses}


def check_judge(judge_config, saved, judge_family=None):
    generator_id = saved.get('generator', {}).get('parameters', {}).get('model')
    judge_id = judge_config.get('parameters', {}).get('model')
    if generator_id and generator_id == judge_id:
        raise ValueError('The judge must differ from the generator')
    generator_family = saved.get('generator_family')
    if generator_family and judge_family and generator_family == judge_family:
        raise ValueError('The configured generator and judge families must differ')
