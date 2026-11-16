#!/usr/bin/env python3
"""Model matrix over frozen graph pairs: generators, judges and self-evaluation.

Run from the GRETEL root. prepare makes an immutable copy of the pairs (the
dumps of one or more first passes, --source per pass) and writes one config
per model. submit queues one Slurm job per model, chained so that one model
uses the GPU at a time (or, with "parallel" in the slurm settings, side by side, one GPU each; then submit
the generators, the judges and the self jobs as separate batches, since the judges read every generator's
outputs): the generators, then the judges, each loading once and
evaluating every generator's outputs, then the self-evaluation jobs (each
generator judging its own Consistency inversion test, when the configuration
has a self_evaluation block). --only queues a subset of those jobs into an
existing run (submit --run-root), which is how a long matrix is split into
batches. summary also works while jobs are running. rerun-judges creates a
fresh matrix using the saved generator answers.

A judge entry may carry its own "probes" (e.g. the Recourse budget sweep) and
"source_filter", substrings of which one must appear in a source's path for the
judge to read it.

Shared GPUs (fleet). A model entry may carry "gpu_memory_gb", the GiB its engine
takes (weights, activations and KV cache): its task waits for that much free
memory instead of one threshold for every model, and the engine is sized to it.
An entry may also carry "tier", a memory class named in the slurm settings'
"tiers" (smallest first); fleet --tier queues one class. A fleet task is one
model over every run given, loaded once (run-chain): a narrator, the judges
that share a model, or a narrator's self-evaluation. Tasks are idempotent: a
narrator with its narratives written is skipped, a judge evaluates the cells
(narrator, judge) without results whose narrator is done, and a claim file
keeps two running jobs off the same work, so a fleet can be launched again and
only does what is missing. fit loads the models of a configuration on the real
GPU and answers the longest prompt of the pairs, before a run is prepared.
extend adds new models (new ids) to a prepared run without touching what is
frozen.
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
import socket
import subprocess
import sys
import tempfile
import traceback

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


def _alive(held):
    """Whether the job that wrote a claim still runs: a Slurm job by squeue, a local one by its process."""
    if held.get('job'):
        try:
            answer = subprocess.run(['squeue', '-h', '-o', '%i', '-j', str(held['job'])], text=True,
                                    capture_output=True, timeout=120)
            return answer.returncode == 0 and bool(answer.stdout.strip())
        except FileNotFoundError:
            # No squeue on this node: a claim of another node cannot be checked here, so its holder keeps it.
            # fleet, on the login node, removes the claims of dead jobs when it is launched again.
            if held.get('host') != socket.gethostname():
                return True
        except (OSError, subprocess.SubprocessError):
            return True   # the scheduler did not answer: leave the work to its holder, a later fleet picks it up
    if held.get('host') != socket.gethostname() or not held.get('pid'):
        return False
    try:
        os.kill(held['pid'], 0)
    except ProcessLookupError:
        return False
    except OSError:
        pass
    return True


def claim(path):
    """Take a piece of work (a narrator of a run, a judge's cell) for this job. False if a running job holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mine = {'job': os.environ.get('SLURM_JOB_ID'), 'host': socket.gethostname(), 'pid': os.getpid(),
            'time': datetime.now(timezone.utc).isoformat()}
    for _ in range(2):
        try:
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w') as stream:
                json.dump(mine, stream)
            return True
        except FileExistsError:
            try:
                held = json.loads(path.read_text())
            except (OSError, ValueError):
                held = {}
            if all(held.get(key) == mine[key] for key in ('job', 'host', 'pid')):
                return True
            if _alive(held):
                return False
            path.unlink(missing_ok=True)   # left by a job that died
    return False


def release(path):
    Path(path).unlink(missing_ok=True)


def clear_stale_claims(runs):
    """Remove the claims that jobs which no longer run left behind (killed, out of time)."""
    for run in runs:
        for path in list(run.glob('generated/*/claim.json')) + list(run.glob('probes/*/*/claim.json')):
            try:
                held = json.loads(path.read_text())
            except (OSError, ValueError):
                held = {}
            if not _alive(held):
                path.unlink(missing_ok=True)
                print('Removed the claim of a job that no longer runs:', path.relative_to(run.parent), flush=True)


class Engine:
    """A model loaded at most once per process, by its first use, and shared by the runs of a chain."""

    def __init__(self):
        self.instance = self.configuration = self.error = None

    def get(self, context, local_config):
        if self.error is not None:
            raise self.error
        if self.instance is None:
            from src.core.factory_base import get_instance_kvargs
            try:
                self.instance = get_instance_kvargs(local_config['class'],
                    {'context': context, 'local_config': copy.deepcopy(local_config)})
            except Exception as exc:
                self.error = exc
                raise
            self.configuration = copy.deepcopy(local_config)
        elif local_config != self.configuration:
            raise ValueError('A chain shares one model; this configuration is not the loaded one')
        return self.instance


def run_config(path):
    """(context, configuration) of one model of a run. The context is global to the process, so the later runs
    of a chain get their configuration processed as the context does its own."""
    from src.utils.context import Context
    context = Context.get_context(str(path))
    if Path(context.config_file).resolve() == Path(path).resolve():
        return context, context.conf
    from jsonc_parser.parser import JsoncParser
    from src.utils.composer import compose, propagate
    conf = propagate(compose(JsoncParser.parse_file(str(path))))
    parameters = conf['experiment'].setdefault('parameters', {})
    for key in ('lock_release_tout', 'lock_timeout'):
        parameters.setdefault(key, context.conf['experiment']['parameters'][key])
    return context, conf


def validate_sources(dumps, config):
    from src.utils.probe_common import DELTA_FIELDS, validate_delta
    from src.utils.probe_graph import restore, unsupported_changes
    from src.evaluation.future.stages.probe_narratives import frame_truth
    print('GRETEL dependencies loaded; checking graph pairs', flush=True)
    if 'sample_limit' in config and len(dumps) != config['sample_limit']:
        raise ValueError(f'Expected exactly {config["sample_limit"]} source dumps, found {len(dumps)}; select one run/fold')
    seen = set()
    for path, payload, digest in dumps:
        saved = payload.get('data', {})
        key = (payload.get('id'), payload.get('fold_id'), payload.get('run_id'), str(path.parent.parent.parent))
        if key in seen:
            raise ValueError('Repeated source instance; select one generation run')
        seen.add(key)
        if saved.get('schema_version') != 1 or saved.get('node_matching') != 'position':
            raise ValueError('Unsupported source schema: ' + str(path))
        records = saved.get('counterfactuals')
        if not isinstance(records, list) or not records or 'arms' not in config and len(records) != 1:
            raise ValueError('This mini matrix requires one counterfactual per instance')
        if 'arms' in config and sorted(r.get('arm') for r in records) != sorted(config['arms']):
            raise ValueError('Source records are not one per configured arm: ' + str(path))
        for record in records:
            if record.get('input_status', record.get('status')) != 'success':
                # Arm records keep their input errors (padding edits, no seed) in the denominator.
                if 'arm' in record:
                    continue
                raise ValueError('Invalid source graph pair: ' + str(path))
            original, cf = restore(record['input']), restore(record['counterfactual'])
            if 'num_nodes' in config and any(len(graph.data) != config['num_nodes'] for graph in (original, cf)):
                raise ValueError('Unexpected source graph size: ' + str(path))
            flips = record['input_label'] != record['target_label']
            if not flips and record.get('arm') != 'random_subset' or unsupported_changes(original, cf):
                raise ValueError('Source is not a supported counterfactual')
            # The nodes the counterfactual's edits deleted (protocol v2) are nodes_removed, as the stage wrote them.
            truth = frame_truth(original, cf, record.get('deleted_nodes', []), saved.get('feature_columns'),
                                {column: name for name, column in saved.get('feature_map', {}).items()}, saved.get('atol', 0))
            declared = {field: record['truth'].get(field, []) for field in DELTA_FIELDS}
            actual = {field: truth[field] for field in DELTA_FIELDS}
            if validate_delta(declared, original.directed, feature_match='transition') != actual:
                raise ValueError('Saved truth disagrees with graph pair')
            for direction in record.get('directions', ('direct', 'inverse')):
                request = record.get(direction + '_generation', {}).get('judge_request', {})
                if any(not isinstance(request.get(k), str) or not request[k] for k in ('system', 'prompt')):
                    raise ValueError('Saved generation prompts are required')


def validate_resources(config):
    """The memory classes and budgets of the entries, before anything is frozen or queued."""
    tiers = config['slurm'].get('tiers', [])
    for entry in config['generators'] + config['judges']:
        if 'tier' in entry and entry['tier'] not in tiers:
            raise ValueError(f'Unknown tier of {entry["id"]}: {entry["tier"]} (slurm.tiers: {tiers})')
        memory = entry.get('gpu_memory_gb')
        if memory is not None and (type(memory) not in (int, float) or memory <= 0):
            raise ValueError(f'gpu_memory_gb of {entry["id"]} must be a positive number of GiB')


def model_config(config, entry, role):
    parameters = {**config['model_defaults'], **entry['parameters']}
    result = {'experiment': {'scope': config['scope']},
              'compose_strs': 'lab/config/snippets/default_store_paths.json',
              role: {'class': entry.get('class', 'src.LLMexplaneability.huggingface.HuggingFaceLLM'), 'parameters': parameters},
              role + '_family': entry['family']}
    if role == 'judge':
        result['probes'] = copy.deepcopy(entry.get('probes', config['probes']))
    return result


def self_config(config, entry):
    """The generator as judge of its own outputs, only on the configured self-evaluation probes."""
    result = model_config(config, entry, 'judge')
    result['probes'] = copy.deepcopy(config['self_evaluation']['probes'])
    result['self_evaluation'] = True
    return result


def prepare(config_path, source, destination=None):
    print('Preparing matrix; configuration:', config_path, flush=True)
    from scripts.run_paper_probes import load_dumps, validate_configuration
    config = json.loads(Path(config_path).read_text())
    validate_configuration(config, list(config['probes']))
    for judge in config['judges']:
        if 'probes' in judge:
            validate_configuration(judge, list(judge['probes']))
    if 'self_evaluation' in config:
        validate_configuration({**config['self_evaluation'], 'self_evaluation': True}, list(config['self_evaluation']['probes']))
    check_models(config)
    sources_in = [source] if isinstance(source, (str, Path)) else list(source)
    print('Reading source dumps:', ', '.join(map(str, sources_in)), flush=True)
    # sample_per_source: the first N pairs of each source (by dump path), for a pilot on a few instances.
    limit = config.get('sample_per_source')
    dumps = [dump for root in sources_in for dump in load_dumps(root)[:limit]]
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
    if 'self_evaluation' in config:
        for entry in config['generators']:
            write_json(run / 'configs' / ('self-' + entry['id'] + '.json'), self_config(config, entry))
    manifest = {'configuration': config, 'sources': sources,
                'source_fingerprint': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                # The pairs themselves, independent of paths: equal in two runs (two accounts, two halves of the
                # narrators) prepared from the same first passes, so their results can be merged.
                'content_fingerprint': hashlib.sha256(json.dumps(sorted(s['sha256'] for s in sources)).encode()).hexdigest(),
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write_json(run / 'matrix.json', manifest)
    print('Matrix:', run, flush=True)
    return run


def check_models(config):
    """Unique safe ids, judges of another family than every narrator, known memory classes."""
    for role in ('generators', 'judges'):
        ids = [entry['id'] for entry in config[role]]
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[a-z0-9][a-z0-9.-]*', tag) for tag in ids):
            raise ValueError('Model IDs must be unique safe directory names')
    for generator in config['generators']:
        for judge in config['judges']:
            if generator['family'] == judge['family'] or generator['parameters']['model'] == judge['parameters']['model']:
                raise ValueError('Generator and judge must differ, including family')
    validate_resources(config)


def extend(run, config_path):
    """Add to a prepared run the models of a configuration that it does not have (new ids), e.g. a narrator at
    another precision or a judge that became available. Nothing frozen changes: the pairs, the probes, the
    defaults and the models already in the run stay as they are, and an entry with a known id must be the same
    model (only its memory class and budget, and the slurm settings, are updated). Reads and writes JSON only."""
    from scripts.run_paper_probes import validate_configuration
    manifest = json.loads((run / 'matrix.json').read_text())
    config, new = manifest['configuration'], json.loads(Path(config_path).read_text())
    for key in ('scope', 'arms', 'model_defaults', 'probes', 'self_evaluation'):
        if new.get(key) != config.get(key):
            raise ValueError(f'"{key}" is not the run\'s: that needs a new run')
    resources = ('tier', 'gpu_memory_gb')
    model = lambda entry: {key: value for key, value in entry.items() if key not in resources}
    added = []
    for role in ('generators', 'judges'):
        known = {entry['id']: entry for entry in config[role]}
        for entry in new[role]:
            if entry['id'] not in known:
                config[role].append(copy.deepcopy(entry))
                added.append((role[:-1], entry))
            elif model(entry) != model(known[entry['id']]):
                raise ValueError(f'{entry["id"]} is in the run with another configuration; a changed model needs a new id')
            else:
                known[entry['id']].update({key: entry[key] for key in resources if key in entry})
    config['slurm'] = new['slurm']
    check_models(config)
    for role, entry in added:
        if role == 'judge' and 'probes' in entry:
            validate_configuration(entry, list(entry['probes']))
        write_json(run / 'configs' / (role + '-' + entry['id'] + '.json'), model_config(config, entry, role))
        if role == 'generator' and 'self_evaluation' in config:
            write_json(run / 'configs' / ('self-' + entry['id'] + '.json'), self_config(config, entry))
    manifest.setdefault('extensions', []).append({
        'time': datetime.now(timezone.utc).isoformat(), 'added': [f'{role}:{entry["id"]}' for role, entry in added],
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    write_json(run / 'matrix.json', manifest)
    print(run.name, 'added:', [f'{role}:{entry["id"]}' for role, entry in added] or 'nothing (budgets and slurm settings updated)', flush=True)
    return added


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
            for before, after in zip(source['data']['counterfactuals'], payload['data']['counterfactuals']):
                for key in ('arm', 'input', 'counterfactual', 'truth', 'input_label', 'target_label', 'graph_text',
                            'inverse_graph_text', 'renaming'):
                    if before.get(key) != after.get(key):
                        raise ValueError('Saved generation changed graph pair')
                if before.get('input_status', before.get('status')) != 'success':
                    continue
                for direction in before.get('directions', ('direct', 'inverse')):
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


def jobs(config, include_generators=True):
    """(role, entry) of every job of the matrix, in submission order."""
    result = [('judge', entry) for entry in config['judges']]
    if include_generators:
        result = [('generator', entry) for entry in config['generators']] + result
    if 'self_evaluation' in config:
        result += [('self', entry) for entry in config['generators']]
    return result


def submit(run, dry_run=False, include_generators=True, only=None, after=None):
    """Queue the matrix's jobs, chained one after the other. ``only`` is a list of
    "role:id" (role generator, judge or self) to queue just those, into a run
    whose other jobs were submitted before or will be; ``after`` makes the first
    of them wait for a job (e.g. the last one of the previous batch)."""
    config = json.loads((run / 'matrix.json').read_text())['configuration']
    settings = config['slurm']
    ledger = run / 'jobs.jsonl'
    submitted = set()
    if ledger.exists():
        submitted = {(row['role'], row['id']) for row in map(json.loads, ledger.read_text().splitlines())}
    selected = jobs(config, include_generators)
    if only is not None:
        wanted = {tuple(item.split(':', 1)) for item in only}
        unknown = wanted - {(role, entry['id']) for role, entry in selected}
        if unknown:
            raise ValueError('Unknown jobs: ' + ', '.join(':'.join(w) for w in sorted(unknown)))
        selected = [(role, entry) for role, entry in selected if (role, entry['id']) in wanted]
    again = [f'{role}:{entry["id"]}' for role, entry in selected if (role, entry['id']) in submitted]
    if again and not dry_run:
        raise ValueError('Jobs already submitted to this run: ' + ', '.join(again))
    (ROOT / 'lab/output/logs').mkdir(parents=True, exist_ok=True)
    previous = after
    for role, entry in selected:
        memory = entry.get('gpu_memory_gb')
        cmd = ['sbatch', '--parsable', '--chdir=' + str(ROOT),
               '--time=' + settings['time'], '--mem=' + settings['mem'],
               '--job-name=pm-' + role[:3] + '-' + entry['id'],
               f'--export=ALL,MIN_FREE_MB={free_mb(memory, settings)},MAX_WAIT={settings["max_wait"]},HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1'
               + (f',GRETEL_GPU_MEMORY_GB={memory}' if memory is not None else '')
               + (f',GRETEL_ENV={settings["env"]}' if 'env' in settings else '')]
        if entry.get('cpu'):
            # A model that needs no GPU (the synthetic narrators of V0): a CPU node, no GPU wait.
            cmd[-1] += ',SKIP_GPU_WAIT=1'
            cmd += ['--partition=' + settings.get('cpu_partition', 'normal'), '--gres=none']
        else:
            for option in ('partition', 'gres'):
                if option in settings:
                    cmd.append(f'--{option}=' + settings[option])
        if 'node' in settings and not entry.get('cpu'):
            cmd.append('--nodelist=' + settings['node'])
        if settings.get('parallel') and after:
            cmd.append('--dependency=afterany:' + after)
        elif settings.get('parallel'):
            pass   # each job asks for its own GPU; the jobs of one submission run side by side
        elif previous:
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


def free_mb(memory, settings):
    """MiB that must be free on the GPU before a task starts: the model's budget plus a margin, or, for an
    entry without a budget, the settings' common threshold."""
    if memory is None:
        return settings['min_free_mb']
    return int(memory * 1024) + settings.get('free_margin_mb', 1024)


def model_key(config, entry):
    """Entries with the same class and parameters are one model: one engine serves them."""
    return json.dumps([entry.get('class'), {**config['model_defaults'], **entry['parameters']}], sort_keys=True)


def pending(role, ids, runs, ready=False):
    """(run, id, narrator) of the work a chain still has: narrators without narratives, judge cells without
    results. ``ready`` keeps only what can run now, the cells whose narrator has written its narratives."""
    left = []
    for run in runs:
        config = json.loads((run / 'matrix.json').read_text())['configuration']
        generators = [entry['id'] for entry in config['generators']]
        judges = [entry['id'] for entry in config['judges']]
        for tag in ids:
            if role == 'generator':
                if tag in generators and not (run / 'generated' / tag / 'generation.json').exists():
                    left.append((run, tag, tag))
                continue
            if role == 'self':
                cells = [(tag, 'self')] if tag in generators and 'self_evaluation' in config else []
            else:
                cells = [(g, tag) for g in generators] if tag in judges else []
            for g, j in cells:
                if any((run / 'probes' / g / j).glob('*/summary.json')):
                    continue
                if ready and not (run / 'generated' / g / 'generation.json').exists():
                    continue
                left.append((run, j, g))
    return left


def fleet_tasks(runs, tier=None, memory=None):
    """The chains of a fleet with work left: per role and model, its ids and the runs that have it.

    ``tier`` keeps one memory class: its narrators and their self-evaluations, and the judges of that class and
    of the smaller ones, which only evaluate what is still missing (e.g. the small judge over the narrators of
    a larger class). ``memory`` replaces the budget of an id, in GiB."""
    memory = dict(memory or {})
    tasks = {}
    for run in runs:
        config = json.loads((run / 'matrix.json').read_text())['configuration']
        tiers = config['slurm'].get('tiers', [])
        if tier is not None and tier not in tiers:
            raise ValueError(f'Unknown tier {tier} for {run} (slurm.tiers: {tiers})')
        for role, entries in (('generator', config['generators']), ('judge', config['judges']),
                              ('self', config['generators'] if 'self_evaluation' in config else [])):
            for entry in entries:
                if tier is not None:
                    if entry.get('tier') not in tiers:
                        raise ValueError(f'{entry["id"]} has no tier of {tiers}')
                    own, asked = tiers.index(entry['tier']), tiers.index(tier)
                    if own > asked or role != 'judge' and own != asked:
                        continue
                task = tasks.setdefault((role, model_key(config, entry)), {
                    'role': role, 'cpu': bool(entry.get('cpu')), 'ids': [], 'runs': [], 'memory': None,
                    'model': entry['parameters']['model']})
                if entry['id'] not in task['ids']:
                    task['ids'].append(entry['id'])
                if run not in task['runs']:
                    task['runs'].append(run)
                budget = memory.get(entry['id'], entry.get('gpu_memory_gb'))
                if budget is not None:
                    task['memory'] = max(task['memory'] or 0, budget)
    unknown = set(memory) - {tag for task in tasks.values() for tag in task['ids']}
    if unknown:
        raise ValueError('--memory for models that this fleet does not queue: ' + ', '.join(sorted(unknown)))
    return [task for task in tasks.values() if pending(task['role'], task['ids'], task['runs'])]


def queue_array(name, lines, listing, settings, maximum, cpu=False, node=None, after=None, dry_run=False):
    """One throttled Slurm array, a task per line of the listing. Returns its job id."""
    import shlex
    export = (f'--export=ALL,MAX_WAIT={settings["max_wait"]},HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1'
              + (f',GRETEL_ENV={settings["env"]}' if 'env' in settings else '') + (',SKIP_GPU_WAIT=1' if cpu else ''))
    cmd = ['sbatch', '--parsable', '--chdir=' + str(ROOT), '--time=' + settings['time'], '--mem=' + settings['mem'],
           f'--array=0-{len(lines) - 1}%{maximum}', '--job-name=' + name, export]
    cmd += (['--partition=' + settings.get('cpu_partition', 'normal'), '--gres=none'] if cpu else
            [f'--{o}=' + settings[o] for o in ('partition', 'gres') if o in settings])
    if node and not cpu:
        cmd.append('--nodelist=' + node)
    if after:
        cmd.append('--dependency=afterany:' + after)
    cmd += ['scripts/slurm_paper_probes.sh', 'array', str(listing)]
    if dry_run:
        print(shlex.join(cmd))
        for line in lines:
            print('   ', line)
        return f'<{name}>'
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text(''.join(line + '\n' for line in lines))
    (ROOT / 'lab/output/logs').mkdir(parents=True, exist_ok=True)
    job = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Unexpected sbatch output: ' + job)
    return job


def fleet(runs, maximum, dry_run=False, tier=None, memory=None, node=None):
    """The work left in several prepared runs as throttled Slurm arrays, launched once and left running.

    A task is a chain: one model over every run, in the order given (e.g. Tree-Cycles, then BBBP, then AIDS),
    loaded once, so a model that had to wait for GPU memory keeps it until it has finished everywhere. One
    array of narrators and one of evaluators (the judges that share a model, and each narrator's
    self-evaluation), which waits for the narrators' array (afterany: a failed narrator leaves its cells
    pending, the others go on). Each task waits for its own model's memory (gpu_memory_gb), the smallest
    first, and at most ``maximum`` tasks of an array run at a time. Everything is idempotent (see the module
    text), so the same fleet can be launched again later and queues only the chains with work left. CPU models
    (the synthetic narrators of V0) get their own array on the CPU partition. Lists of the tasks:
    <first run>/fleet/<time>-<tier>-<array>.list (role, GiB, MiB to wait for, ids, runs)."""
    if not dry_run:
        clear_stale_claims(runs)
    tasks = fleet_tasks(runs, tier, memory)
    if not tasks:
        print('Nothing left to queue' + (f' in tier {tier}' if tier else '') + ': every narrator and cell is done', flush=True)
        return
    settings = json.loads((runs[0] / 'matrix.json').read_text())['configuration']['slurm']
    stamp, label = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), tier or 'all'
    generators_job = None
    for name, roles in (('generators', ('generator',)), ('evaluators', ('judge', 'self'))):
        for cpu in (True, False):
            part = sorted((t for t in tasks if t['role'] in roles and t['cpu'] == cpu), key=lambda t: t['memory'] or 0)
            if not part:
                continue
            lines = [' '.join([t['role'], '-' if t['memory'] is None else str(t['memory']),
                               str(free_mb(t['memory'], settings)), ','.join(t['ids'])] + [str(run) for run in t['runs']])
                     for t in part]
            listing = runs[0] / 'fleet' / f'{stamp}-{label}-{name}{"-cpu" if cpu else ""}.list'
            job = queue_array(f'pm-{name[:3]}-{label}', lines, listing, settings, maximum, cpu, node,
                              generators_job if name == 'evaluators' else None, dry_run)
            if not dry_run:
                for index, task in enumerate(part):
                    for run in task['runs']:
                        with (run / 'jobs.jsonl').open('a') as stream:
                            for tag in task['ids']:
                                stream.write(json.dumps({'role': task['role'], 'id': tag, 'model': task['model'],
                                                         'job_id': f'{job}_{index}', 'fleet': listing.name}) + '\n')
                print(f'{label} {name}{" (cpu)" if cpu else ""}: array {job}, {len(part)} tasks, at most {maximum} at a time', flush=True)
                for index, task in enumerate(part):
                    print(f'  {job}_{index}: {task["role"]} {",".join(task["ids"])}'
                          + ('' if task['memory'] is None else f', {task["memory"]} GiB'), flush=True)
            if name == 'generators':
                generators_job = job if generators_job is None else f'{generators_job}:{job}'


def fit(config_path, sources, maximum, tiers=None, models=None, memory=None, node=None, dry_run=False):
    """Queue a load test of GPU models of a configuration, before any run is prepared from it: each task waits
    for the model's memory, loads it as the matrix will (run-fit) and answers the longest prompt of the pairs."""
    config = json.loads(Path(config_path).read_text())
    validate_resources(config)
    memory = dict(memory or {})
    seen, lines = set(), []
    for entry in sorted(config['generators'] + config['judges'], key=lambda e: memory.get(e['id'], e.get('gpu_memory_gb')) or 0):
        key = model_key(config, entry)
        if (entry.get('cpu') or key in seen or tiers and entry.get('tier') not in tiers
                or models and entry['id'] not in models):
            continue
        seen.add(key)
        budget = memory.get(entry['id'], entry.get('gpu_memory_gb'))
        lines.append(' '.join(['fit', '-' if budget is None else str(budget), str(free_mb(budget, config['slurm'])),
                               entry['id'], str(Path(config_path).resolve())] + [str(Path(x).resolve()) for x in sources]))
    if not lines:
        raise ValueError('No GPU model selected')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    listing = ROOT / 'lab/output/results' / config['scope'] / 'fit' / f'{stamp}.list'
    job = queue_array('pm-fit', lines, listing, config['slurm'], maximum, node=node, dry_run=dry_run)
    if not dry_run:
        print(f'fit: array {job}, {len(lines)} models, at most {maximum} at a time; results in {listing.parent}', flush=True)


def run_fit(config_path, tag, sources):
    """Load one model with its matrix configuration and answer a short prompt and the longest generation prompt
    of the pairs. Writes <scope>/fit/<id>.json and prints FIT OK or FIT FAILED."""
    import time
    from scripts.run_paper_probes import load_dumps
    config = json.loads(Path(config_path).read_text())
    role, entry = next((role, entry) for role, entries in (('generator', config['generators']), ('judge', config['judges']))
                       for entry in entries if entry['id'] == tag)
    directory = ROOT / 'lab/output/results' / config['scope'] / 'fit'
    write_json(directory / (tag + '.config.json'), model_config(config, entry, role))
    requests = [record[direction + '_generation']['judge_request']
                for root in sources for _, payload, _ in load_dumps(root)
                for record in (payload.get('data') or {}).get('counterfactuals', [])
                for direction in record.get('directions', ('direct', 'inverse'))
                if isinstance(record.get(direction + '_generation', {}).get('judge_request'), dict)]
    longest = max(requests, key=lambda r: len(r['system']) + len(r['prompt']))
    result = {'id': tag, 'model': entry['parameters']['model'], 'gpu_memory_gb': os.environ.get('GRETEL_GPU_MEMORY_GB'),
              'longest_prompt_characters': len(longest['system']) + len(longest['prompt'])}
    try:
        context, conf = run_config(directory / (tag + '.config.json'))
        start = time.time()
        model = Engine().get(context, conf[role])
        result['load_seconds'] = round(time.time() - start)
        answers = []
        for system, prompt in (('You answer in one short sentence.', 'Name the three primary colours.'),
                               (longest['system'], longest['prompt'])):
            start = time.time()
            answer = model.explain_counterfactual(system, prompt)
            answers.append({'seconds': round(time.time() - start), 'characters': len(answer), 'text': str(answer)[:400],
                            **{k: v for k, v in getattr(answer, 'llm_metadata', {}).items() if k in ('finish_reason', 'output_token_count')}})
        result.update(status='ok', answers=answers)
        if any(not a['characters'] for a in answers):
            raise ValueError('Empty answer')
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(directory / (tag + '.json'), result)
        traceback.print_exc()
        print(f'FIT FAILED {tag}: {result["error"]}', flush=True)
        return 2
    write_json(directory / (tag + '.json'), result)
    print(f'FIT OK {tag}: loaded in {result["load_seconds"]} s, longest prompt ({result["longest_prompt_characters"]} '
          f'characters) answered in {answers[1]["seconds"]} s', flush=True)
    return 0


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


def run_generator(run, tag, engine=None):
    from src.utils.context import clean_cfg
    from src.utils.probe_generation import regenerate_saved
    from src.utils.probe_common import counts
    directory = run / 'generated' / tag
    if (directory / 'generation.json').exists():
        print(tag, run.name, 'narratives already written', flush=True)
        return 0
    if not claim(directory / 'claim.json'):
        print(tag, run.name, 'is being narrated by another running job', flush=True)
        return 0
    try:
        context, config = run_config(run / 'configs' / ('generator-' + tag + '.json'))
        dumps = frozen_dumps(run)
        generator = (engine or Engine()).get(context, config['generator'])
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
                direction: counts([row[direction + '_generation'] for row in outcomes
                                   if direction + '_generation' in row])['status_counts']
                for direction in sorted({d for row in outcomes for d in row.get('directions', ('direct', 'inverse'))})}})
    finally:
        release(directory / 'claim.json')
    return 0


def evaluate_cell(run, g, tag, engine, context, config, role_config, keep, manifest):
    """One cell (narrator g, judge tag) if it has no results, its narrator is done and no running job holds it.
    Returns 'complete', 'waiting' (narrator not done), 'held', 'done' or 'error'."""
    from scripts.run_paper_probes import execute, load_dumps
    output = run / 'probes' / g / tag
    if any(output.glob('*/summary.json')):
        return 'complete'
    if not (run / 'generated' / g / 'generation.json').exists():
        return 'waiting'
    if not claim(output / 'claim.json'):
        return 'held'
    try:
        (output / 'error.json').unlink(missing_ok=True)
        try:
            judge = engine.get(context, role_config)
        except Exception as exc:
            raise RuntimeError(f'Judge initialization: {type(exc).__name__}: {exc}') from exc
        generation = json.loads((run / 'generated' / g / 'generation.json').read_text())
        if generation['source_fingerprint'] != manifest['source_fingerprint']:
            raise ValueError('Different graph pairs used by this generator')
        for source in generation.get('reused_outputs', []):
            if hashlib.sha256((run / source['path']).read_bytes()).hexdigest() != source['sha256']:
                raise ValueError('Reused generator output was modified')
        dumps = load_dumps(run / 'generated' / g)
        if len(dumps) != len(manifest['sources']):
            raise ValueError('Incomplete generator outputs')
        destination, errors = execute(judge, [dump for dump in dumps if keep(dump[1])], config, context,
                                      list(config['probes']), output)
        print(g, tag, 'Results:', destination, 'attempt_errors:', errors, flush=True)
        return 'done'
    except Exception as exc:
        message = str(exc) if isinstance(exc, RuntimeError) and exc.__cause__ is not None else f'{type(exc).__name__}: {exc}'
        write_json(output / 'error.json', {'generator': g, 'judge': tag, 'error': message})
        print(g, tag, 'ERROR:', message, flush=True)
        return 'error'
    finally:
        release(output / 'claim.json')


def run_judge(run, tag, engine=None):
    """The judge over every narrator of the run that is done and that it has not evaluated yet. The model is
    loaded by the first cell with work, so a job with nothing to do ends at once."""
    context, config = run_config(run / 'configs' / ('judge-' + tag + '.json'))
    manifest = json.loads((run / 'matrix.json').read_text())
    frozen_dumps(run)
    entry = next(e for e in manifest['configuration']['judges'] if e['id'] == tag)
    keep = source_filter(manifest, entry.get('source_filter'))
    engine = engine or Engine()
    states = {}
    for generator in manifest['configuration']['generators']:
        g = generator['id']
        states[g] = evaluate_cell(run, g, tag, engine, context, config, config['judge'], keep, manifest)
    print(tag, run.name, 'cells:', states, flush=True)
    collect(run)
    return 2 if 'error' in states.values() else 0


def source_filter(manifest, substrings):
    """Whether a generated dump comes from a source whose original path contains one of the substrings."""
    if not substrings:
        return lambda payload: True
    original = {source['frozen']: source['path'] for source in manifest['sources']}
    return lambda payload: any(text in original[payload['matrix_source']['path']] for text in substrings)


def run_self(run, tag, engine=None):
    """The generator judges its own outputs on the self-evaluation probes."""
    context, config = run_config(run / 'configs' / ('self-' + tag + '.json'))
    manifest = json.loads((run / 'matrix.json').read_text())
    frozen_dumps(run)
    keep = source_filter(manifest, manifest['configuration']['self_evaluation'].get('source_filter'))
    state = evaluate_cell(run, tag, 'self', engine or Engine(), context, config, config['judge'], keep, manifest)
    print(tag, run.name, 'self:', state, flush=True)
    collect(run)
    return 2 if state == 'error' else 0


def run_chain(role, ids, runs):
    """One model, loaded once, over several runs in order: a narrator (role generator), the judges that share a
    model (role judge, e.g. the primary judge and its budget sweep) or a narrator's self-evaluation (self)."""
    step = {'generator': run_generator, 'judge': run_judge, 'self': run_self}[role]
    engine, failed = Engine(), False
    for run in runs:
        config = json.loads((run / 'matrix.json').read_text())['configuration']
        known = [entry['id'] for entry in config['judges' if role == 'judge' else 'generators']]
        for tag in ids:
            if tag not in known or role == 'self' and 'self_evaluation' not in config:
                continue
            try:
                failed |= bool(step(run, tag, engine))
            except Exception as exc:
                failed = True
                traceback.print_exc()
                print(role, tag, run.name, 'ERROR:', type(exc).__name__, str(exc), flush=True)
    return 2 if failed else 0


def collect(run):
    config = json.loads((run / 'matrix.json').read_text())['configuration']
    rows, cells = [], []
    judges = [judge['id'] for judge in config['judges']] + (['self'] if 'self_evaluation' in config else [])
    for generator in config['generators']:
        for j in judges:
            g = generator['id']
            cell = run / 'probes' / g / j
            summaries = sorted(cell.glob('*/summary.json'))
            status = 'complete' if summaries else 'error' if (cell / 'error.json').exists() else 'pending'
            cells.append({'generator': g, 'judge': j, 'status': status})
            for path in summaries:
                for row in json.loads(path.read_text()):
                    rows.append({'generator': g, 'judge': j, 'result': str(path), **row})
    write_json(run / 'matrix-summary.json', {'comparisons': cells, 'results': rows})
    columns = ['generator', 'judge', 'probe', 'condition', 'n', 'n_success', 'status_counts',
               'edges_f1_valid_mean', 'n_semantic_valid', 'semantic_yes_count', 'n_valid', 'consistent_rate_valid',
               'with_explanation_success_rate_all_attempts', 'without_explanation_success_rate_all_attempts',
               'with_explanation_n_reused_original', 'without_explanation_n_reused_original',
               'success_rate_difference_all_attempts', 'result']
    with (run / 'matrix-summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, dict) else value for key, value in row.items()})
    done = [g['id'] for g in config['generators'] if (run / 'generated' / g['id'] / 'generation.json').exists()]
    print('Narrators done:', done, 'missing:', [g['id'] for g in config['generators'] if g['id'] not in done], flush=True)
    print('Comparisons:', {status: sum(cell['status'] == status for cell in cells) for status in ('complete', 'error', 'pending')}, flush=True)
    print('Summary:', run / 'matrix-summary.csv', flush=True)
    return rows, cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'submit'):
        command = commands.add_parser(name)
        command.add_argument('--config', default=DEFAULT_CONFIG)
        command.add_argument('--source', action='append', help='first-pass results root; repeat for several')
        command.add_argument('--output')
        if name == 'submit':
            command.add_argument('--dry-run', action='store_true')
            command.add_argument('--run-root', help='queue into this prepared run instead of preparing one')
            command.add_argument('--only', help='comma-separated role:id jobs (role generator, judge or self)')
            command.add_argument('--after', help='Slurm job the first queued job waits for')
    rerun = commands.add_parser('rerun-judges')
    rerun.add_argument('--from-matrix', required=True)
    rerun.add_argument('--config', default=DEFAULT_CONFIG)
    rerun.add_argument('--output')
    rerun.add_argument('--judge', action='append')
    rerun.add_argument('--probe', action='append')
    rerun.add_argument('--dry-run', action='store_true')
    command = commands.add_parser('fleet')
    command.add_argument('--run-root', action='append', required=True, help='prepared run; repeat, in run order')
    command.add_argument('--max', type=int, default=4, help='tasks of one array running at a time')
    command.add_argument('--tier', help='queue one memory class of the models (slurm.tiers of the configuration)')
    command.add_argument('--memory', action='append', default=[], metavar='ID=GIB', help='another GPU budget for a model')
    command.add_argument('--node', help='run the GPU tasks on this node')
    command.add_argument('--dry-run', action='store_true')
    command = commands.add_parser('fit')
    command.add_argument('--config', required=True)
    command.add_argument('--source', action='append', required=True, help='first-pass results root; repeat for several')
    command.add_argument('--tier', action='append', help='memory class to test; repeat for several (default: all)')
    command.add_argument('--model', action='append', help='model id to test; repeat for several (default: all)')
    command.add_argument('--max', type=int, default=4)
    command.add_argument('--memory', action='append', default=[], metavar='ID=GIB')
    command.add_argument('--node')
    command.add_argument('--dry-run', action='store_true')
    command = commands.add_parser('run-fit')
    command.add_argument('--config', required=True)
    command.add_argument('--model', required=True)
    command.add_argument('--source', action='append', required=True)
    command = commands.add_parser('extend')
    command.add_argument('--config', required=True)
    command.add_argument('--run-root', action='append', required=True)
    for name in ('run-chain', 'pending'):
        command = commands.add_parser(name)
        command.add_argument('--role', required=True, choices=('generator', 'judge', 'self'))
        command.add_argument('--ids', required=True, help='comma-separated ids that share one model')
        command.add_argument('--run-root', action='append', required=True)
        if name == 'pending':
            command.add_argument('--ready', action='store_true', help='only the work that can run now')
    for name in ('run-generator', 'run-judge', 'run-self', 'summary'):
        command = commands.add_parser(name)
        command.add_argument('--run-root', required=True)
        if name in ('run-generator', 'run-judge'):
            command.add_argument('--' + name.split('-')[1], required=True)
        elif name == 'run-self':
            command.add_argument('--generator', required=True)
    args = parser.parse_args(argv)
    # Before NumPy/PyTorch imports: respect the cluster's CPU thread policy.
    # Prepare/submit/summary run on the access node with one thread; model jobs
    # use their allocated CPUs. This also covers already queued batch scripts
    # which invoke the current Python file without the newer shell exports.
    os.environ['OMP_NUM_THREADS'] = (os.environ.get('SLURM_CPUS_PER_TASK', '1')
        if args.command in ('run-generator', 'run-judge', 'run-self', 'run-chain', 'run-fit') else '1')
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'
    if args.command in ('fleet', 'fit'):
        memory = {tag: float(value) for tag, value in (item.split('=', 1) for item in args.memory)}
        memory = {tag: int(value) if value == int(value) else value for tag, value in memory.items()}
        if args.command == 'fleet':
            fleet([Path(r).resolve() for r in args.run_root], args.max, args.dry_run, args.tier, memory, args.node)
        else:
            fit(args.config, args.source, args.max, args.tier, args.model, memory, args.node, args.dry_run)
        return 0
    if args.command == 'run-fit':
        return run_fit(args.config, args.model, args.source)
    if args.command == 'extend':
        for root in args.run_root:
            extend(Path(root).resolve(), args.config)
        return 0
    if args.command in ('run-chain', 'pending'):
        runs, ids = [Path(r).resolve() for r in args.run_root], args.ids.split(',')
        if args.command == 'run-chain':
            return run_chain(args.role, ids, runs)
        left = pending(args.role, ids, runs, args.ready)
        for run, tag, generator in left:
            print(run.name, tag, generator)
        return 0 if left else 10   # 10: nothing to do (the Slurm script ends the task without loading the model)
    if args.command == 'submit' and args.run_root:
        submit(Path(args.run_root).resolve(), args.dry_run, only=args.only.split(',') if args.only else None,
               after=args.after)
        return 0
    if args.command in ('prepare', 'submit'):
        run = prepare(args.config, args.source or [DEFAULT_SOURCE], args.output)
        if args.command == 'submit':
            submit(run, args.dry_run, only=args.only.split(',') if args.only else None, after=args.after)
        return 0
    if args.command == 'rerun-judges':
        run = prepare_judge_rerun(args.config, args.from_matrix, args.output, args.judge, args.probe)
        submit(run, args.dry_run, include_generators=False)
        return 0
    run = Path(args.run_root).resolve()
    if args.command == 'run-generator':
        return run_generator(run, args.generator)
    elif args.command == 'run-judge':
        return run_judge(run, args.judge)
    elif args.command == 'run-self':
        return run_self(run, args.generator)
    else:
        collect(run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
