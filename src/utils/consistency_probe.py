"""CONSISTENCY probe: is the narrator's stated mechanism stable under three tests?

Each test pairs the direct narrative (A) with a second one (B) of the same
narrator, and an external judge says whether the two mechanisms are compatible,
that is, whether both could be true of the same change:

- inversion: B narrates the counterfactual back to the input (G' -> G); its
  mechanism must be compatible with the negation of A's.
- renaming: B narrates the same pair with the nodes renamed; its mechanism
  must be compatible with A's once the names are mapped back (the mapping is
  given to the judge).
- identity: B narrates the same input again with another sampling seed; its
  mechanism must be compatible with A's. This is the noise floor of the other
  two tests.

A narrative that states no mechanism (only what changed, not why the class
changes) fails the test. Parsing is strict: anything but the JSON object below
is ``judge_unparsed``, never repaired.
"""
from src.utils.probe_common import call_many, read_object, select_text

TESTS = ('inversion', 'renaming', 'identity')
VERDICTS = ('compatible', 'incompatible', 'no_mechanism')

SYSTEM_PROMPT = """You compare two explanations, A and B, of graph counterfactuals: changes to a graph that change the class a model predicts for it.
The MECHANISM of an explanation is its stated reason why the change makes the model predict the new class (for example, which structure or property the change creates or destroys, and why that matters for the class). A list of the edits, or a restatement that the class changed, is not a mechanism.

TASK:
1. Summarize the mechanism of A in one sentence (all its causes together), or null if A states none.
2. Summarize the mechanism of B in one sentence, or null if B states none.
3. Decide, under the relation given in RELATION, whether the two mechanisms are compatible: whether both could be true of the same model and the same change.
   - Different wording, detail, emphasis or number of causes does not make mechanisms incompatible.
   - Contradictory reasons do: one attributes the change of class to something the other denies, reverses or rules out.
   - Judge only the agreement between A and B, not whether either is true of the domain.

Return ONLY a JSON object, no Markdown and no other text, with exactly these three fields:
{"mechanism_a": "..." or null, "mechanism_b": "..." or null, "verdict": "compatible" | "incompatible" | "no_mechanism"}
Use "no_mechanism" when, and only when, A or B states no mechanism (its field is then null)."""

RELATIONS = {
    'inversion': ('A explains the change from an input graph to a counterfactual graph. B explains the reverse change, '
                  'from that counterfactual back to the input: its edits are the inverse of A\'s and its two classes '
                  'are swapped, which is expected. B is compatible with A when its mechanism is the negation of A\'s: '
                  'what A says brings about the new class, B says undoing it brings back the original class.'),
    'renaming': ('A and B explain the same change, but in B the nodes have other identifiers. NODE MAPPING gives, for '
                 'every node of A, its identifier in B. B is compatible with A when, with the identifiers mapped, '
                 'its mechanism is compatible with A\'s; a node named differently is the same node.'),
    'identity': ('A and B explain exactly the same change, written twice by the same author. B is compatible with A '
                 'when its mechanism is compatible with A\'s.'),
}


def build_prompt(test, text_a, text_b, mapping=None):
    prompt = f'--- RELATION ---\n{RELATIONS[test]}\n'
    if test == 'renaming':
        pairs = ', '.join(f'{a}->{b}' for a, b in enumerate(mapping))
        prompt += f'--- NODE MAPPING (A->B) ---\n{pairs}\n'
    return prompt + f'--- TEXT A ---\n{text_a.strip()}\n--- TEXT B ---\n{text_b.strip()}\n'


def parse(output):
    obj = read_object(output)
    if set(obj) != {'mechanism_a', 'mechanism_b', 'verdict'}:
        raise ValueError('Expected exactly mechanism_a, mechanism_b and verdict')
    for key in ('mechanism_a', 'mechanism_b'):
        if obj[key] is not None and (not isinstance(obj[key], str) or not obj[key].strip()):
            raise ValueError(key + ' must be a sentence or null')
    if obj['verdict'] not in VERDICTS:
        raise ValueError('Unknown verdict')
    if (obj['verdict'] == 'no_mechanism') != (obj['mechanism_a'] is None or obj['mechanism_b'] is None):
        raise ValueError('no_mechanism exactly when a mechanism is null')
    return obj


def run(judge, items, test, mode='dict', require_structured=False):
    """Items: dicts with output_a, output_b (raw generator answers), directed and,
    for renaming, mapping (permutation[i] = identifier in B of node i of A)."""
    if test not in TESTS:
        raise ValueError('Unknown consistency test')
    records, prompts, pending = [], [], []
    for item in items:
        record = {'status': None, 'test': test, 'verdict': None, 'consistent': None}
        records.append(record)
        try:
            a = select_text(item['output_a'], mode, require_structured, item['directed'])
            b = select_text(item['output_b'], mode, require_structured, item['directed'])
        except ValueError as exc:
            record.update(status='unparsed', error=str(exc))
            continue
        prompts.append((SYSTEM_PROMPT, build_prompt(test, a, b, item.get('mapping'))))
        pending.append(len(records) - 1)
    for index, answer in zip(pending, call_many(judge, prompts)):
        record = records[index]
        record.update(answer)
        if answer['status'] != 'success':
            continue
        try:
            verdict = parse(answer['judge_output'])
        except ValueError as exc:
            record.update(status='judge_unparsed', error=str(exc))
            continue
        record.update(status='success', verdict=verdict['verdict'], mechanism_a=verdict['mechanism_a'],
                      mechanism_b=verdict['mechanism_b'], consistent=verdict['verdict'] == 'compatible')
    return records
