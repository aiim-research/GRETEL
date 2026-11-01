"""RECONSTRUCTION probe, independent of the pipeline: the judge prompt, the
narrative selection (MODE), and the per-item bookkeeping. Used by the
Reconstruction stage (in-pipeline), by Reversal, and by
scripts/run_paper_probes.py (offline), so all score exactly the same way.

An item is prepared, sent to the judge, then finished:

    record, prompt = prepare(output, graph_text, use_context, mode)
    judge_output = judge(prompt)              # skipped when prompt is None
    finish(record, judge_output, truth, directed)
"""
from src.utils.reconstruction_metrics import score_delta
from src.utils.probe_common import call_many, select_text, read_object, validate_delta

JUDGE_SYSTEM_PROMPT = """You are a Graph Modification Extractor.
Your task is to read a text about a graph counterfactual and list, as structured data, the TECHNICAL MODIFICATIONS it mentions: which edges were added, which edges were removed, and which node features were changed.

RULES:
- Extract only modifications that the text states. Do not infer or invent any.
- Ignore statements about the classification and about why the class changed.
- Use the exact node identifiers and feature names that appear in the text.
- Return ONLY a JSON object, no Markdown, no comments, with exactly this schema:
{
  "nodes_added": [id, ...],
  "nodes_removed": [id, ...],
  "edges_added": [[u, v], ...],
  "edges_removed": [[u, v], ...],
  "features_changed": [{"node": id, "feature": "name", "from": value, "to": value}, ...]
}
- Use empty lists for modification types the text does not mention.
- Include numeric from/to values when explicitly given by the text. Never invent them.
"""

def build_prompt(narrative, graph_text, use_context):
    prompt = ''
    if use_context:
        prompt += '--- FACTUAL GRAPH ---\n' + (graph_text or '') + '\n'
    prompt += ('--- TEXT ---\n' + narrative.strip() + '\n\n'
               '--- INSTRUCTION ---\nExtract the JSON delta based strictly on the text above.\n')
    return prompt


def prepare(output, graph_text, use_context, mode, require_structured=False, directed=False):
    """Pick the narrative for the condition and build the judge prompt.

    Returns (record, prompt); prompt is None when there is nothing to judge,
    in which case the record is already final (status "unparsed")."""
    record = {'status': None, 'narrative_used': None, 'judge_output': None,
              'extraction': None, 'scores': None}
    try:
        narrative = select_text(output, mode, require_structured, directed)
    except ValueError as exc:
        record['status'] = 'unparsed'
        record['error'] = str(exc)
        return record, None
    record['narrative_used'] = narrative
    return record, build_prompt(narrative, graph_text, use_context)


def finish(record, judge_output, truth, directed, feature_match='transition'):
    """Parse the judge's answer and score it against the ground truth."""
    record['judge_output'] = judge_output
    try:
        extraction = validate_delta(read_object(judge_output), directed, feature_match=feature_match)
    except ValueError as exc:
        record.update(status='judge_unparsed', error=str(exc))
        return record
    record['status'] = 'success'
    record['extraction'] = extraction
    if truth is not None:
        record['scores'] = score_delta(truth, extraction, directed, feature_match)
    return record


def run(judge, items, use_context, mode, feature_match='transition', require_structured=False):
    """Run the probe over items = [(output, graph_text, truth, directed), ...].

    Prompts are sent together through judge.explain_many when the backend
    offers it (batched under vllm), one by one otherwise."""
    records, prompts, pending = [], [], []
    for output, graph_text, truth, directed in items:
        record, prompt = prepare(output, graph_text, use_context, mode, require_structured, directed)
        records.append(record)
        if prompt is not None:
            system = JUDGE_SYSTEM_PROMPT
            if feature_match == 'transition':
                system += '\nThis condition requires numeric from/to values for every reported feature transition.\n'
            prompts.append((system, prompt))
            pending.append(len(records) - 1)

    if prompts:
        answers = call_many(judge, prompts)
        for index, answer in zip(pending, answers):
            _, _, truth, directed = items[index]
            records[index].update(answer)
            if answer['status'] == 'success':
                finish(records[index], answer['judge_output'], truth, directed, feature_match)
    return records
