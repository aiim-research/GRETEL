#!/usr/bin/env python3
"""Score the judges of the validation V0 against what the synthetic narratives contain by construction.

The V0 matrix (lab/config/narratives/matrix_v0.json) narrates real pairs with the synthetic narrators
(src/LLMexplaneability/synthetic.py), one per variant, and runs every candidate judge over them. Here each
judge's answers are compared with the known content (minimized arm):

- Reconstruction: the extraction must be exactly the edits the text states (``fidelity``); also the mean
  precision and recall against the true delta, next to the values the text allows (``expected``).
- Consistency: the verdict against the expected one of each variant and test (``accuracy``).
- Recourse: success rate with and without the explanation per variant and dataset; on Tree-Cycles the
  true mechanism (``exact``) should help more than a false one or none.

Parse failures count as errors in every rate, and their rate is reported. Run it on the local machine once the
run is back (or as a CPU job): it reads JSON only.

    python scripts/v0_judge_validation.py --run-root lab/output/results/narr-v0/main
"""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.LLMexplaneability.synthetic import edit_key, expected_verdict, stated_edits  # noqa: E402


def dataset_of(saved):
    """Dataset of a first-pass record set, from its generator class (the frozen file names say nothing)."""
    generator = saved.get('dataset', {}).get('parameters', {}).get('generator', {})
    cls = generator.get('class', '')
    if 'treecycles' in cls.lower():
        return 'tcr'
    if cls.endswith('MolGenerator'):
        return 'bbbp'
    return generator.get('parameters', {}).get('alias', cls.rsplit('.', 1)[-1]).lower()


def load_probe_files(run, generator, judge):
    """(probe, condition tag, generated dump payload, probe records) for every source file of the cell."""
    for path in sorted((run / 'probes' / generator / judge).glob('*/*/*/*.json')):
        if path.name in ('summary.json', 'manifest.json'):
            continue
        result = json.loads(path.read_text())
        source = json.loads(Path(result['source']).read_text())
        yield path.parts[-3], path.parts[-2], source, result['counterfactuals']


def rate(values):
    return sum(values) / len(values) if values else float('nan')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args(argv)
    run = Path(args.run_root)
    config = json.loads((run / 'matrix.json').read_text())['configuration']
    variants = {g['id']: g['parameters']['variant'] for g in config['generators']}
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for generator, variant in variants.items():
        for judge in [j['id'] for j in config['judges']]:
            for probe, tag, source, records in load_probe_files(run, generator, judge):
                dataset = dataset_of(source['data'])
                for item, record in zip(source['data']['counterfactuals'], records):
                    if item.get('arm') != 'minimized' or record['status'] == 'not_applicable':
                        continue
                    key = (judge, probe, tag, variant, dataset)
                    rows[key]['parsed'].append(record['status'] == 'success')
                    if probe == 'reconstruction':
                        request = item['direct_generation']['judge_request']
                        stated = {edit_key(f, v) for f, v in stated_edits(variant, request['system'], request['prompt'])}
                        truth = {edit_key(f, v) for f, v in stated_edits('exact', request['system'], request['prompt'])}
                        extraction = record.get('extraction') or {}
                        got = {edit_key(f, v) for f in extraction for v in extraction[f]} if record['status'] == 'success' else None
                        rows[key]['fidelity'].append(got == stated)
                        if got is not None:
                            rows[key]['precision'].append(len(got & truth) / len(got) if got else float(not truth))
                            rows[key]['recall'].append(len(got & truth) / len(truth) if truth else 1.0)
                        rows[key]['expected_precision'].append(len(stated & truth) / len(stated) if stated else float(not truth))
                        rows[key]['expected_recall'].append(len(stated & truth) / len(truth) if truth else 1.0)
                    elif probe == 'consistency':
                        test = tag.split('test-')[1].split('_')[0]
                        rows[key]['accuracy'].append(record.get('verdict') == expected_verdict(variant, test))
                    elif probe == 'recourse':
                        for arm in ('with_explanation', 'without_explanation'):
                            outcome = record.get(arm) or {}
                            rows[key][arm].append(bool(outcome.get('successful')))
    lines = ['# V0: judges against the synthetic narratives', '',
             f'Run `{run}`. Minimized arm; rates over every record, parse failures counted as errors.', '',
             '| Judge | Probe | Condition | Variant | Dataset | n | Parsed | Fidelity / accuracy | P (exp.) | R (exp.) | Recourse with / without |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for (judge, probe, tag, variant, dataset), m in sorted(rows.items()):
        n = len(m['parsed'])
        main_rate = rate(m['fidelity'] if probe == 'reconstruction' else m['accuracy']) if probe != 'recourse' else float('nan')
        pr = (f"{rate(m['precision']):.2f} ({rate(m['expected_precision']):.2f})", f"{rate(m['recall']):.2f} ({rate(m['expected_recall']):.2f})") \
            if probe == 'reconstruction' else ('', '')
        rec = f"{rate(m['with_explanation']):.2f} / {rate(m['without_explanation']):.2f}" if probe == 'recourse' else ''
        lines.append(f"| {judge} | {probe} | {tag} | {variant} | {dataset} | {n} | {rate(m['parsed']):.2f} | "
                     f"{'' if probe == 'recourse' else f'{main_rate:.2f}'} | {pr[0]} | {pr[1]} | {rec} |")
    totals = collections.defaultdict(list)
    for (judge, probe, _, _, _), m in rows.items():
        if probe != 'recourse':
            totals[(judge, probe)] += m['fidelity'] if probe == 'reconstruction' else m['accuracy']
    lines += ['', '| Judge | Reconstruction fidelity | Consistency accuracy |', '|---|---|---|']
    for judge in sorted({j for j, _ in totals}):
        lines.append(f"| {judge} | {rate(totals[(judge, 'reconstruction')]):.2f} | {rate(totals[(judge, 'consistency')]):.2f} |")
    report = '\n'.join(lines) + '\n'
    (run / 'v0-report.md').write_text(report)
    print(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
