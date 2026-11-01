"""Reversal: Alejandra's semantic comparison plus Rodrigo's inverse-edit test."""
from src.utils import reconstruction_probe
from src.utils.probe_common import call_many, inverse_delta, select_text
from src.utils.reconstruction_metrics import score_delta

# The criteria and YES/NO task are those of the existing contrastive stage.
SYSTEM_PROMPT = '''You are given two texts about a graph counterfactual process.
Goal: Decide if they describe the SAME process but in OPPOSITE DIRECTIONS
(A: input -> counterfactual, B: counterfactual -> input).
Criteria for YES (all must hold):
1) Same graph/entities/features throughout.
2) Each modification in one text has an inverse in the other.
3) Start/end classes are swapped consistently.
If any criterion is not met or information is insufficient, answer NO.
Do not explain or justify. Output exactly one word: YES or NO.'''


def run(judge, items, use_context=False, mode='dict', feature_match='transition', require_structured=False):
    """Items: direct raw, inverse raw, factual text, inverse factual text, directed.

    This probe reads no true delta and invokes extraction independently, even
    when Reconstruction was never run or used another judge.
    """
    forward_items = [(a, ga, None, d) for a, b, ga, gb, d in items]
    backward_items = [(b, gb, None, d) for a, b, ga, gb, d in items]
    forward = reconstruction_probe.run(judge, forward_items, use_context, mode, feature_match, require_structured)
    backward = reconstruction_probe.run(judge, backward_items, use_context, mode, feature_match, require_structured)
    records, prompts, pending = [], [], []
    for i, (a, b, ga, gb, directed) in enumerate(items):
        record = {'status': 'partial_error', 'forward': forward[i], 'backward': backward[i],
                  'structural_scores': None, 'semantic': None}
        if forward[i]['status'] == backward[i]['status'] == 'success':
            record['structural_scores'] = score_delta(inverse_delta(forward[i]['extraction']),
                                                       backward[i]['extraction'], directed, feature_match)
        records.append(record)
        try:
            ta = select_text(a, mode, require_structured, directed)
            tb = select_text(b, mode, require_structured, directed)
        except ValueError as exc:
            record.update(status='unparsed', error=str(exc))
            continue
        prompts.append((SYSTEM_PROMPT, f'--- TEXT A ---\n{ta}\n--- TEXT B ---\n{tb}'))
        pending.append(i)
    for index, answer in zip(pending, call_many(judge, prompts)):
        record = records[index]
        if answer['status'] == 'success':
            response = answer['judge_output'].strip().upper()
            if response not in ('YES', 'NO'):
                answer.update(status='judge_unparsed', error='Expected exactly YES or NO')
            else:
                answer['consistent'] = response == 'YES'
        record['semantic'] = answer
        if (record['forward']['status'] == record['backward']['status'] == answer['status'] == 'success'):
            record['status'] = 'success'
    return records
