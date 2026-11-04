"""Replay the saved first-pass prompts on unchanged graph pairs."""
import copy

from src.utils.context import clean_cfg
from src.utils.probe_common import call_many, generation_outcome


def regenerate_saved(generator, saved, family):
    result = copy.deepcopy(saved)
    if result.get('schema_version') != 1 or result.get('node_matching') != 'position':
        raise ValueError('Unsupported first-pass schema')
    records = result.get('counterfactuals')
    if not isinstance(records, list) or not records:
        raise ValueError('Missing graph pairs')
    prompts, owners = [], []
    base = generator.local_config['parameters'].get('seed', 0)
    for index, record in enumerate(records):
        status = record.get('input_status', record.get('status'))
        if status != 'success':
            # Arm records keep their input errors (padding edits, no seed, ...) in the denominator.
            if 'arm' in record:
                continue
            raise ValueError('Matrix generation requires validated graph pairs')
        for direction in record.get('directions', ('direct', 'inverse')):
            request = record.get(direction + '_generation', {}).get('judge_request', {})
            if any(not isinstance(request.get(key), str) or not request[key] for key in ('system', 'prompt')):
                raise ValueError('Missing saved generation prompt: ' + direction)
            prompts.append((request['system'], request['prompt']))
            owners.append((index, direction, request.get('seed_offset', 0)))
    seeds = [base + offset for _, _, offset in owners] if any(o for _, _, o in owners) else None
    for (index, direction, offset), answer in zip(owners, call_many(generator, prompts, seeds)):
        record = records[index]
        if offset:
            answer['judge_request']['seed_offset'] = offset
        record[direction + '_output'] = answer['judge_output']
        record[direction + '_generation'] = generation_outcome(answer, record['input']['directed'])
    for record in records:
        if record.get('input_status', record.get('status')) != 'success':
            continue
        record['input_status'] = 'success'
        record['status'] = ('success' if all(record[d + '_generation']['status'] == 'success'
                                           for d in record.get('directions', ('direct', 'inverse'))) else 'partial_error')
    result['generator'] = copy.deepcopy(clean_cfg(generator.local_config))
    result['generator_family'] = family
    return result
