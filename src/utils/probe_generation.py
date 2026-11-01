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
    for index, record in enumerate(records):
        if record.get('input_status', record.get('status')) != 'success':
            raise ValueError('Matrix generation requires validated graph pairs')
        for direction in ('direct', 'inverse'):
            request = record.get(direction + '_generation', {}).get('judge_request', {})
            if any(not isinstance(request.get(key), str) or not request[key] for key in ('system', 'prompt')):
                raise ValueError('Missing saved generation prompt: ' + direction)
            prompts.append((request['system'], request['prompt']))
            owners.append((index, direction))
    for (index, direction), answer in zip(owners, call_many(generator, prompts)):
        record = records[index]
        record[direction + '_output'] = answer['judge_output']
        record[direction + '_generation'] = generation_outcome(answer, record['input']['directed'])
    for record in records:
        record['input_status'] = 'success'
        record['status'] = ('success' if all(record[d + '_generation']['status'] == 'success'
                                           for d in ('direct', 'inverse')) else 'partial_error')
    result['generator'] = copy.deepcopy(clean_cfg(generator.local_config))
    result['generator_family'] = family
    return result
