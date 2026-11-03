#!/usr/bin/env python3
"""Second pass: independent paper probes over ProbeDump's complete records.

Run from the repository root. --probe selects one probe; omit it to execute
the configured probes. Only Recourse restores a dataset and oracle, through
the existing factories. A manifest identifies the effective judge and sources.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_dumps(root, limit=None):
    dumps = []
    for path in sorted(Path(root).glob('**/probe_inputs/**/cf_*.json')):
        if 'probes' in path.relative_to(root).parts:
            continue
        content = path.read_bytes()
        try:
            payload = json.loads(content)
            if not isinstance(payload, dict) or not isinstance(payload.get('data'), dict):
                raise ValueError('Missing saved generation data')
        except (ValueError, TypeError) as exc:
            payload = {'id': path.stem, 'fold_id': None, 'run_id': None,
                       'input_error': f'{type(exc).__name__}: {exc}'}
        dumps.append((path, payload, hashlib.sha256(content).hexdigest()))
        if limit is not None and len(dumps) >= limit:
            break
    return dumps


def validate_configuration(configuration, selected):
    for name in selected:
        if name not in ('reconstruction', 'reversal', 'recourse') or name not in configuration['probes']:
            raise ValueError('Unknown or unconfigured probe: ' + name)
        options = configuration['probes'][name]
        if not isinstance(options, dict):
            raise ValueError('Probe configuration must be an object')
        if options.get('feature_match', 'transition') not in ('identity', 'transition'):
            raise ValueError('Invalid feature matching')
        if options.get('success', 'target') not in ('target', 'flip'):
            raise ValueError('Invalid Recourse success criterion')
        if any(type(options.get(key, True)) is not bool for key in ('control', 'preprocess')):
            raise ValueError('control and preprocess must be booleans')
        conditions = options.get('conditions', [{'context': 'off', 'mode': 'dict'}])
        seen = set()
        if not isinstance(conditions, list) or not conditions:
            raise ValueError('A probe needs at least one condition')
        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError('A condition must be an object')
            ctx, mode = condition.get('context', 'off'), condition.get('mode', 'dict')
            if ctx not in ('on', 'off') or mode not in ('full', 'dict'):
                raise ValueError('Invalid context or mode')
            if (ctx, mode) in seen or name == 'recourse' and ctx != 'off':
                raise ValueError('Duplicate condition or CTX-on used as a Recourse control')
            seen.add((ctx, mode))


def execute(judge, dumps, configuration, context, selected, output_root):
    from src.utils import probe_inputs
    from src.utils.context import clean_cfg
    from src.utils.probe_common import check_judge

    validate_configuration(configuration, selected)
    effective = copy.deepcopy(configuration)
    effective['judge'] = clean_cfg(judge.local_config)
    sources = [{'path': str(path.resolve()), 'sha256': digest} for path, _, digest in dumps]
    identity = {'protocol_version': probe_inputs.PROBE_PROTOCOL_VERSION,
                'configuration': effective, 'probes': selected, 'sources': sources}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    output = Path(output_root) / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(identity, indent=2) + '\n')
    oracles = {}
    summaries = []
    had_errors = False
    for name in selected:
        options = effective['probes'][name]
        conditions = options.get('conditions', [{'context': 'off', 'mode': 'dict'}])
        for condition in conditions:
            ctx, mode = condition.get('context', 'off'), condition.get('mode', 'dict')
            tag = f'ctx-{ctx}_mode-{mode}'
            all_records = []
            for path, payload, digest in dumps:
                saved = payload.get('data')
                dataset, predictor, preprocess = None, None, None
                records = None
                if saved is None:
                    records = [{'status': 'input_error', 'error': payload.get('input_error', 'Missing first-pass data')}]
                else:
                    check_judge(effective['judge'], saved, effective.get('judge_family'))
                    if name == 'recourse':
                        try:
                            from src.dataset.dataset_factory import DatasetFactory
                            from src.oracle.oracle_factory import OracleFactory
                            key = json.dumps({'dataset': saved['dataset'], 'oracle': saved['oracle']}, sort_keys=True)
                            if key not in oracles:
                                dataset = DatasetFactory(context).get_dataset(copy.deepcopy(saved['dataset']))
                                oracle = OracleFactory(context).get_oracle(copy.deepcopy(saved['oracle']), dataset)
                                oracles[key] = (dataset, oracle)
                            dataset, oracle = oracles[key]
                            predictor = oracle.predict
                            if options.get('preprocess', True):
                                preprocess = dataset.manipulate
                        except Exception as exc:
                            records = [{'status': 'oracle_error', 'error': f'Replay initialization: {type(exc).__name__}: {exc}'}
                                       for _ in saved.get('counterfactuals') or [None]]
                    if records is None:
                        try:
                            records = probe_inputs.run_saved(name, judge, saved, ctx, mode,
                                                             options.get('feature_match', 'transition'),
                                                             options.get('control', True), predictor, dataset,
                                                             options.get('success', 'target'), options.get('edge_defaults'), preprocess)
                        except (ValueError, KeyError, TypeError) as exc:
                            records = [{'status': 'input_error', 'error': f'{type(exc).__name__}: {exc}'}
                                       for _ in saved.get('counterfactuals') or [None]]
                had_errors |= any(record['status'] != 'success' for record in records)
                all_records.extend(records)
                # Source digest + path keep sample IDs/folds/configurations distinct.
                source_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]
                destination = output / name / tag / f'{source_id}.json'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({'id': payload['id'], 'fold_id': payload['fold_id'],
                                                  'run_id': payload['run_id'], 'source': str(path),
                                                  'source_sha256': digest, 'configuration': options,
                                                  'condition': condition, 'counterfactuals': records}, indent=2) + '\n')
            row = {'probe': name, 'condition': tag, **probe_inputs.summarize(name, all_records)}
            summaries.append(row)
            print(f"{name} {tag}: {row['n_success']}/{row['n']} completed; statuses={row['status_counts']}", flush=True)
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    return output, had_errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--probe', choices=('reconstruction', 'reversal', 'recourse'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', help='Default: <results>/probes')
    parser.add_argument('--fail-on-error', action='store_true', help='Exit 2 after saving results if any attempt has an error')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    from src.utils.context import Context
    from src.core.factory_base import get_instance_kvargs
    context = Context.get_context(args.config)
    configuration = context.conf
    selected = [args.probe] if args.probe else list(configuration['probes'])
    if not selected or any(name not in ('reconstruction', 'reversal', 'recourse') for name in selected):
        parser.error('Configure at least one supported probe')
    for name in selected:
        if name not in configuration['probes']:
            parser.error(f'{name} is not configured')
    try:
        validate_configuration(configuration, selected)
    except ValueError as exc:
        parser.error(str(exc))
    dumps = load_dumps(args.results, args.limit)
    if not dumps:
        parser.error('No ProbeDump files found; run the paper generation pipeline first')
    judge = get_instance_kvargs(configuration['judge']['class'],
                               {'context': context, 'local_config': copy.deepcopy(configuration['judge'])})
    output, errors = execute(judge, dumps, configuration, context, selected,
                             args.output or str(Path(args.results) / 'probes'))
    print(f'Results: {output}', flush=True)
    return 2 if errors and args.fail_on_error else 0


if __name__ == '__main__':
    sys.exit(main())
