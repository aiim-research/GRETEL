#!/usr/bin/env python3
"""A cached-model pilot: frozen graph pairs, six generators, three judges.

Run from the GRETEL root. submit prepares an immutable source copy and queues
six generator jobs followed by three judge jobs. Each judge loads once and
evaluates all generator outputs. summary also works while jobs are running.
rerun-judges creates a fresh matrix using the saved generator answers.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = 'lab/config/probes/paper_treecycles_10nodes_matrix.json'
DEFAULT_SOURCE = 'lab/output/results/paper-probes-mini-10nodes'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def validate_sources(dumps, config):
    from src.utils.probe_common import DELTA_FIELDS, validate_delta
    from src.utils.probe_graph import restore, unsupported_changes
    from src.utils.typed_delta import typed_delta_from_instances
    print('GRETEL dependencies loaded; checking graph pairs', flush=True)
    if len(dumps) != config['sample_limit']:
        raise ValueError(f'Expected exactly {config["sample_limit"]} source dumps, found {len(dumps)}; select one run/fold')
    seen = set()
    for path, payload, digest in dumps:
        saved = payload.get('data', {})
        key = (payload.get('id'), payload.get('fold_id'), payload.get('run_id'))
        if key in seen:
            raise ValueError('Repeated source instance; select one generation run')
        seen.add(key)
        if saved.get('schema_version') != 1 or saved.get('node_matching') != 'position':
            raise ValueError('Unsupported source schema: ' + str(path))
        records = saved.get('counterfactuals')
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError('This mini matrix requires one counterfactual per instance')
        for record in records:
            if record.get('input_status', record.get('status')) != 'success':
                raise ValueError('Invalid source graph pair: ' + str(path))
            original, cf = restore(record['input']), restore(record['counterfactual'])
            if any(len(graph.data) != config.get('num_nodes', 10) for graph in (original, cf)):
                raise ValueError('Unexpected source graph size: ' + str(path))
            if record['input_label'] == record['target_label'] or unsupported_changes(original, cf):
                raise ValueError('Source is not a supported counterfactual')
            truth = typed_delta_from_instances(original, cf, saved.get('feature_columns'),
                {column: name for name, column in saved.get('feature_map', {}).items()}, saved.get('atol', 0))
            declared = {field: record['truth'].get(field, []) for field in DELTA_FIELDS}
            actual = {field: truth[field] for field in DELTA_FIELDS}
            if validate_delta(declared, original.directed, feature_match='transition') != actual:
                raise ValueError('Saved truth disagrees with graph pair')
            for direction in ('direct', 'inverse'):
                request = record.get(direction + '_generation', {}).get('judge_request', {})
                if any(not isinstance(request.get(k), str) or not request[k] for k in ('system', 'prompt')):
                    raise ValueError('Saved generation prompts are required')


def model_config(config, entry, role):
    parameters = {**config['model_defaults'], **entry['parameters']}
    result = {'experiment': {'scope': config['scope']},
              'compose_strs': 'lab/config/snippets/default_store_paths.json',
              role: {'class': 'src.LLMexplaneability.huggingface.HuggingFaceLLM', 'parameters': parameters},
              role + '_family': entry['family']}
    if role == 'judge':
        result['probes'] = copy.deepcopy(config['probes'])
    return result


def prepare(config_path, source, destination=None):
    print('Preparing matrix; configuration:', config_path, flush=True)
    from scripts.run_paper_probes import load_dumps, validate_configuration
    config = json.loads(Path(config_path).read_text())
    validate_configuration(config, list(config['probes']))
    for role in ('generators', 'judges'):
        ids = [entry['id'] for entry in config[role]]
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[a-z0-9][a-z0-9.-]*', tag) for tag in ids):
            raise ValueError('Model IDs must be unique safe directory names')
    for generator in config['generators']:
        for judge in config['judges']:
            if generator['family'] == judge['family'] or generator['parameters']['model'] == judge['parameters']['model']:
                raise ValueError('Generator and judge must differ, including family')
    print('Reading source dumps:', source, flush=True)
    dumps = load_dumps(source)
    print(f'Found {len(dumps)} dumps; validating inputs and loading GRETEL dependencies', flush=True)
    validate_sources(dumps, config)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run = Path(destination or ROOT / 'lab/output/results' / config['scope'] / name).resolve()
    print('Saving frozen inputs:', run, flush=True)
    run.mkdir(parents=True, exist_ok=False)
    sources = []
    for path, payload, digest in dumps:
        frozen = run / 'source/probe_inputs' / ('cf_' + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20] + '.json')
        # Preserve the original source bytes, including both exact generation prompts.
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(path.read_bytes())
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed while preparing the matrix')
        sources.append({'path': str(path.resolve()), 'sha256': digest, 'frozen': str(frozen), 'id': payload['id']})
    for role, entries in (('generator', config['generators']), ('judge', config['judges'])):
        for entry in entries:
            write_json(run / 'configs' / (role + '-' + entry['id'] + '.json'), model_config(config, entry, role))
    manifest = {'configuration': config, 'sources': sources,
                'source_fingerprint': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write_json(run / 'matrix.json', manifest)
    print('Matrix:', run, flush=True)
    return run


def prepare_judge_rerun(config_path, previous, destination=None, judges=None, probes=None):
    """Freeze existing narratives unchanged; use current judge configurations."""
    print('Preparing judge rerun from:', previous, flush=True)
    from scripts.run_paper_probes import load_dumps
    previous = Path(previous).resolve()
    old = json.loads((previous / 'matrix.json').read_text())
    sources = frozen_dumps(previous)
    validate_sources(sources, old['configuration'])
    available = json.loads(Path(config_path).read_text())
    config = copy.deepcopy(old['configuration'])
    selected = set(judges or [entry['id'] for entry in config['judges']])
    current = {entry['id']: entry for entry in available['judges']}
    if not selected or selected - set(current):
        raise ValueError('Unknown judge selection')
    # Expand the current defaults now: old generator settings stay in the manifest.
    config['judges'] = [copy.deepcopy(entry) for entry in available['judges'] if entry['id'] in selected]
    for entry in config['judges']:
        entry['parameters'] = {**available['model_defaults'], **entry['parameters']}
    if probes:
        if set(probes) - set(config['probes']):
            raise ValueError('Unknown probe selection')
        config['probes'] = {name: value for name, value in config['probes'].items() if name in probes}
    copied = []
    source_map = {payload['id']: (path, payload, digest) for path, payload, digest in sources}
    if len(source_map) != len(sources):
        raise ValueError('Repeated instance ID; select one source run/fold')
    generation_sources = old.get('generation_sources', old['sources'])
    provenance = {source['id']: {'sha256': source['sha256'], 'path': source['frozen']}
                  for source in generation_sources}
    for entry in config['generators']:
        directory = previous / 'generated' / entry['id']
        generation = json.loads((directory / 'generation.json').read_text())
        if generation['source_fingerprint'] != old['source_fingerprint']:
            raise ValueError('Different graph pairs used by this generator')
        if generation['generator']['parameters']['model'] != entry['parameters']['model']:
            raise ValueError('Saved generator model disagrees with matrix')
        dumps = load_dumps(directory)
        validate_sources(dumps, config)
        for path, payload, digest in dumps:
            _, source, source_digest = source_map[payload['id']]
            if (payload.get('matrix_source') != provenance[payload['id']]
                    or provenance[payload['id']]['sha256'] != source_digest):
                raise ValueError('Saved generation provenance disagrees with frozen source')
            if payload['data']['generator'] != generation['generator']:
                raise ValueError('Saved generator configuration disagrees with generation manifest')
            for key in ('dataset', 'oracle', 'domain', 'feature_map', 'feature_columns', 'atol'):
                if payload['data'].get(key) != source['data'].get(key):
                    raise ValueError('Saved generation changed replay configuration')
            before, after = source['data']['counterfactuals'][0], payload['data']['counterfactuals'][0]
            for key in ('input', 'counterfactual', 'truth', 'input_label', 'target_label', 'graph_text', 'inverse_graph_text'):
                if before.get(key) != after.get(key):
                    raise ValueError('Saved generation changed graph pair')
            for direction in ('direct', 'inverse'):
                key = direction + '_generation'
                if before[key]['judge_request'] != after[key]['judge_request']:
                    raise ValueError('Saved generation changed original prompt')
        copied.append((entry['id'], generation, dumps))
    # All inputs are checked before creating a run or submitting any job.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as stream:
            temporary = stream.name
            json.dump(config, stream)
        run = prepare(temporary, previous / 'source', destination)
    finally:
        if temporary:
            os.unlink(temporary)
    manifest = json.loads((run / 'matrix.json').read_text())
    manifest['reuse_generations_from'] = str(previous)
    manifest['generation_sources'] = generation_sources
    write_json(run / 'matrix.json', manifest)
    for tag, generation, dumps in copied:
        files = []
        for path, payload, digest in dumps:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('Generator output changed while preparing rerun')
            target = run / 'generated' / tag / 'probe_inputs' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            files.append({'path': str(target.relative_to(run)), 'sha256': digest})
        write_json(run / 'generated' / tag / 'generation.json', {**generation,
            'source_fingerprint': manifest['source_fingerprint'],
            'original_source_fingerprint': old['source_fingerprint'],
            'generation_reused_from': str(previous / 'generated' / tag), 'reused_outputs': files})
    print('Reused generator outputs:', len(copied), '; judge jobs:', len(config['judges']), flush=True)
    return run


def submit(run, dry_run=False, include_generators=True):
    config = json.loads((run / 'matrix.json').read_text())['configuration']
    settings = config['slurm']
    ledger = run / 'jobs.jsonl'
    if ledger.exists() and not dry_run:
        raise ValueError('Jobs already submitted; use a new matrix run')
    (ROOT / 'lab/output/logs').mkdir(parents=True, exist_ok=True)
    previous = None
    roles = [('judge', config['judges'])]
    if include_generators:
        roles.insert(0, ('generator', config['generators']))
    for role, entries in roles:
        for entry in entries:
            cmd = ['sbatch', '--parsable', '--chdir=' + str(ROOT), '--nodelist=' + settings['node'],
                   '--time=' + settings['time'], '--mem=' + settings['mem'],
                   '--job-name=pm-' + role[:3] + '-' + entry['id'],
                   f'--export=ALL,MIN_FREE_MB={settings["min_free_mb"]},MAX_WAIT={settings["max_wait"]},HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1']
            if previous:
                # A failed model does not block the other models. One model occupies
                # the assigned GPU at a time, including across generator/judge roles.
                cmd.append('--dependency=afterany:' + previous)
            cmd.extend(['scripts/slurm_paper_probes.sh', 'matrix-' + role, str(run), entry['id']])
            if dry_run:
                import shlex
                print(shlex.join(cmd))
                previous = '<previous-job>'
            else:
                print('Submitting', role, entry['id'], flush=True)
                result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
                previous = result.stdout.strip().split(';')[0]
                if not previous.isdigit():
                    raise ValueError('Unexpected sbatch output: ' + result.stdout)
                with ledger.open('a') as stream:
                    stream.write(json.dumps({'role': role, 'model': entry['parameters']['model'], 'id': entry['id'], 'job_id': previous}) + '\n')
                print(role, entry['id'], 'job', previous, flush=True)


def frozen_dumps(run):
    from scripts.run_paper_probes import load_dumps
    manifest = json.loads((run / 'matrix.json').read_text())
    for source in manifest['sources']:
        if hashlib.sha256(Path(source['frozen']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Frozen source was modified')
    dumps = load_dumps(run / 'source')
    if len(dumps) != len(manifest['sources']):
        raise ValueError('Incomplete frozen source')
    return dumps


def run_generator(run, tag):
    from src.utils.context import Context, clean_cfg
    from src.core.factory_base import get_instance_kvargs
    from src.utils.probe_generation import regenerate_saved
    from src.utils.probe_common import counts
    context = Context.get_context(str(run / 'configs' / ('generator-' + tag + '.json')))
    config = context.conf
    dumps = frozen_dumps(run)
    generator = get_instance_kvargs(config['generator']['class'],
        {'context': context, 'local_config': copy.deepcopy(config['generator'])})
    directory = run / 'generated' / tag
    outcomes = []
    for path, payload, digest in dumps:
        data = regenerate_saved(generator, payload['data'], config['generator_family'])
        result = {**payload, 'data': data, 'matrix_source': {'sha256': digest, 'path': str(path)}}
        write_json(directory / 'probe_inputs' / path.name, result)
        outcomes.extend(data['counterfactuals'])
        print(tag, 'instance', payload['id'], [row['status'] for row in data['counterfactuals']], flush=True)
    write_json(directory / 'generation.json', {'generator': clean_cfg(generator.local_config),
        'source_fingerprint': json.loads((run / 'matrix.json').read_text())['source_fingerprint'],
        'summary': counts(outcomes), 'generation_status_counts': {
            direction: counts([row[direction + '_generation'] for row in outcomes])['status_counts']
            for direction in ('direct', 'inverse')}})


def run_judge(run, tag):
    from src.utils.context import Context
    from src.core.factory_base import get_instance_kvargs
    from scripts.run_paper_probes import execute, load_dumps
    context = Context.get_context(str(run / 'configs' / ('judge-' + tag + '.json')))
    config = context.conf
    manifest = json.loads((run / 'matrix.json').read_text())
    frozen_dumps(run)
    try:
        judge = get_instance_kvargs(config['judge']['class'],
            {'context': context, 'local_config': copy.deepcopy(config['judge'])})
    except Exception as exc:
        for generator in manifest['configuration']['generators']:
            write_json(run / 'probes' / generator['id'] / tag / 'error.json', {
                'generator': generator['id'], 'judge': tag,
                'error': f'Judge initialization: {type(exc).__name__}: {exc}'})
        collect(run)
        print(tag, 'ERROR:', type(exc).__name__, str(exc), flush=True)
        return 2
    failed = False
    for generator in manifest['configuration']['generators']:
        g = generator['id']
        output = run / 'probes' / g / tag
        try:
            generation = json.loads((run / 'generated' / g / 'generation.json').read_text())
            if generation['source_fingerprint'] != manifest['source_fingerprint']:
                raise ValueError('Different graph pairs used by this generator')
            for source in generation.get('reused_outputs', []):
                if hashlib.sha256((run / source['path']).read_bytes()).hexdigest() != source['sha256']:
                    raise ValueError('Reused generator output was modified')
            dumps = load_dumps(run / 'generated' / g)
            if len(dumps) != len(manifest['sources']):
                raise ValueError('Incomplete generator outputs')
            destination, errors = execute(judge, dumps, config, context, list(config['probes']), output)
            print(g, tag, 'Results:', destination, 'attempt_errors:', errors, flush=True)
        except Exception as exc:
            failed = True
            write_json(output / 'error.json', {'generator': g, 'judge': tag, 'error': f'{type(exc).__name__}: {exc}'})
            print(g, tag, 'ERROR:', type(exc).__name__, str(exc), flush=True)
    collect(run)
    return 2 if failed else 0


def collect(run):
    config = json.loads((run / 'matrix.json').read_text())['configuration']
    rows, cells = [], []
    for generator in config['generators']:
        for judge in config['judges']:
            g, j = generator['id'], judge['id']
            cell = run / 'probes' / g / j
            summaries = sorted(cell.glob('*/summary.json'))
            status = 'complete' if summaries else 'error' if (cell / 'error.json').exists() else 'pending'
            cells.append({'generator': g, 'judge': j, 'status': status})
            for path in summaries:
                for row in json.loads(path.read_text()):
                    rows.append({'generator': g, 'judge': j, 'result': str(path), **row})
    write_json(run / 'matrix-summary.json', {'comparisons': cells, 'results': rows})
    columns = ['generator', 'judge', 'probe', 'condition', 'n', 'n_success', 'status_counts',
               'edges_f1_valid_mean', 'n_semantic_valid', 'semantic_yes_count',
               'with_explanation_success_rate_all_attempts', 'without_explanation_success_rate_all_attempts',
               'with_explanation_n_reused_original', 'without_explanation_n_reused_original',
               'success_rate_difference_all_attempts', 'result']
    with (run / 'matrix-summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, dict) else value for key, value in row.items()})
    print('Comparisons:', {status: sum(cell['status'] == status for cell in cells) for status in ('complete', 'error', 'pending')}, flush=True)
    print('Summary:', run / 'matrix-summary.csv', flush=True)
    return rows, cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'submit'):
        command = commands.add_parser(name)
        command.add_argument('--config', default=DEFAULT_CONFIG)
        command.add_argument('--source', default=DEFAULT_SOURCE)
        command.add_argument('--output')
        if name == 'submit':
            command.add_argument('--dry-run', action='store_true')
    rerun = commands.add_parser('rerun-judges')
    rerun.add_argument('--from-matrix', required=True)
    rerun.add_argument('--config', default=DEFAULT_CONFIG)
    rerun.add_argument('--output')
    rerun.add_argument('--judge', action='append')
    rerun.add_argument('--probe', action='append')
    rerun.add_argument('--dry-run', action='store_true')
    for name in ('run-generator', 'run-judge', 'summary'):
        command = commands.add_parser(name)
        command.add_argument('--run-root', required=True)
        if name != 'summary':
            command.add_argument('--' + name.split('-')[1], required=True)
    args = parser.parse_args(argv)
    # Before NumPy/PyTorch imports: respect the cluster's CPU thread policy.
    # Prepare/submit/summary run on the access node with one thread; model jobs
    # use their allocated CPUs. This also covers already queued batch scripts
    # which invoke the current Python file without the newer shell exports.
    os.environ['OMP_NUM_THREADS'] = (os.environ.get('SLURM_CPUS_PER_TASK', '1')
        if args.command in ('run-generator', 'run-judge') else '1')
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'
    if args.command in ('prepare', 'submit'):
        run = prepare(args.config, args.source, args.output)
        if args.command == 'submit':
            submit(run, args.dry_run)
        return 0
    if args.command == 'rerun-judges':
        run = prepare_judge_rerun(args.config, args.from_matrix, args.output, args.judge, args.probe)
        submit(run, args.dry_run, include_generators=False)
        return 0
    run = Path(args.run_root).resolve()
    if args.command == 'run-generator':
        run_generator(run, args.generator)
    elif args.command == 'run-judge':
        return run_judge(run, args.judge)
    else:
        collect(run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
