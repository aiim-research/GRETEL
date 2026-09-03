import re
from typing import List, Dict, Any
import numpy as np
from src.dataset.instances.graph import GraphInstance

class FlipRateEvaluator():
    def __init__(self, graph_input:str, graph_counterfactual:str, domain:str, counterfactual_explanation:str):
        self.graph = graph_input
        self.graph_counterfactual = graph_counterfactual
        self.counterfactual_explanation = counterfactual_explanation
        self.domain = domain
        self.PROPOSAL_HEADER_RE = re.compile(r'^\s*Proposal\s+(\d+)\s*$', re.IGNORECASE)
        self.SECTION_LABELS = {
            "edges_added": re.compile(r'^\s*Edges\s+added\s*:\s*\{(.*)\}\s*$', re.IGNORECASE),
            "edges_removed": re.compile(r'^\s*Edges\s+removed\s*:\s*\{(.*)\}\s*$', re.IGNORECASE),
            "node_feats_hdr": re.compile(r'^\s*Node\s+Features\s+Modified\s*:\s*$', re.IGNORECASE),
        }

        # Matches edge tuples like: (v1 -- v2) or (vA_1 -> vB-2)
        self.EDGE_TUPLE_RE = re.compile(r'\(\s*([^()\s,]+)\s*(--|->)\s*([^()\s,]+)\s*\)')

        # Matches a node feature line: v1: [f1, f2, f3]
        self.NODE_FEAT_LINE_RE = re.compile(r'^\s*([^:\s]+)\s*:\s*\[(.*?)\]\s*$')
    

    def flip_rate_prompt(self):
        system = "**Goal.** Propose ONE alternative counterfactual edit sets (K={{K}}) that are **different from the edits in G'**, but that would **also flip the oracle's prediction in the same direction** as G' *for the same underlying reasons stated in the explanation*, and **respect the domain knowledge**.\n\n\
**Important requirements**\n\
1. Your edits must operationalize the causal mechanisms described in the Explanation not just arbitrary perturbations.\n\
2. **Do not reuse** the exact edge or feature changes used in G'. Propose **different** edits that achieve the same effect via alternative paths/motifs/features.\n\
3. Respect all domain constraints. If a constraint prevents a natural edit, find a compliant alternative.\n\
4. Prefer **minimal** edits (fewest changes) that plausibly flip the oracle prediction.\n\
5. If your reasoning implies necessary co-changes (e.g., when adding an edge requires synchronizing attributes), include them explicitly.\n\
6. Output **only** the requested format for each proposal, with no extra text. Use the exact node and feature names from G. Use `--` for undirected edges or `->` for directed graphs (match {{DOMAIN_KNOWLEDGE}}). If a section is empty, write an empty set `{ }`.\n\
7. For node features ALWAYS keep the same vector length given in the original (the number and order of the features is the same).\n\
\n\
**Requested output format (labeled Proposal):**\n\
\n\
Proposal 1  \n\
Edges added: { (1 -- 2), (1 -- 4), ... }  \n\
Edges removed: { (1 -- 2), (1 -- 4), ... }  \n\
Node Features Modified:  \n\
1: [1, 2, 3, ...]  \n\
2: [1, 2, 3, ...]  \n\
...\n\
\n\
**Notes for you (the model):**\n\
- Treat the oracle as a black box but align with the Explanation's mechanisms.\n\
- Avoid any edits that would violate graph invariants in {{DOMAIN_KNOWLEDGE}}.\n\
- Keep each proposal self-contained; do not reference G' explicitly.\n\
- Do **not** include any prose outside the required format.\n\
\n\
Return exactly ONE proposal in the specified format.\n\n"
        prompt = self.domain
        prompt += self.graph
        prompt += self.graph_counterfactual
        prompt += self.counterfactual_explanation
        return system, prompt
    

    def _parse_edge_block(self, block_text: str) -> List[Dict[str, str]]:
        """
        Parse the text inside { ... } for edge lists and return a list of dicts:
        [{"u": "v1", "v": "v2", "type": "undirected"|"directed"}, ...]
        """
        # Empty set or whitespace-only
        if block_text.strip() == "" or block_text.strip() == "}":
            return []
        edges = []
        for u, arrow, v in self.EDGE_TUPLE_RE.findall(block_text):
            edges.append({
                "u": u,
                "v": v,
                "type": "directed" if arrow == "->" else "undirected"
            })
        # If nothing matched but it wasn't empty, you likely have a formatting issue
        if not edges and block_text.strip():
            # Try to be forgiving: remove trailing ellipses or stray commas and re-try
            cleaned = re.sub(r'\.\.\.|…', '', block_text).strip()
            for u, arrow, v in self.EDGE_TUPLE_RE.findall(cleaned):
                edges.append({
                    "u": u,
                    "v": v,
                    "type": "directed" if arrow == "->" else "undirected"
                })
        return edges

    def _parse_feature_list(self, list_text: str) -> List[str]:
        """
        Parse a feature list like 'f1, f2, f3' into ['f1','f2','f3'].
        Accepts optional quotes and trailing commas/spaces.
        """
        raw = [tok.strip() for tok in list_text.split(",") if tok.strip()]
        # Remove optional surrounding quotes
        feats = [t[1:-1] if (len(t) >= 2 and ((t[0] == t[-1] == "'") or (t[0] == t[-1] == '"'))) else t for t in raw]
        return feats

    def parse_proposals(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM output (with one or more 'Proposal N' blocks) into a list of dicts.
        Each proposal has:
        {
            "label": "Proposal 1",
            "number": 1,
            "edges_added": [{"u":..., "v":..., "type":"undirected"|"directed"}, ...],
            "edges_removed": [...],
            "node_features_modified": { node: [feat1, feat2, ...], ... }
        }
        """
        lines = text.splitlines()
        proposals: List[Dict[str, Any]] = []

        i = 0
        current = None
        mode = None  # None | "node_feats"
        while i < len(lines):
            line = lines[i]

            # Detect new proposal header
            m = self.PROPOSAL_HEADER_RE.match(line)
            if m:
                # Flush previous
                if current:
                    proposals.append(current)
                current = {
                    "label": f"Proposal {m.group(1)}",
                    "number": int(m.group(1)),
                    "edges_added": [],
                    "edges_removed": [],
                    "node_features_modified": {}
                }
                mode = None
                i += 1
                continue

            if current is not None:
                # Edges added / removed single-line sections
                m_add = self.SECTION_LABELS["edges_added"].match(line)
                if m_add:
                    current["edges_added"] = self._parse_edge_block(m_add.group(1))
                    mode = None
                    i += 1
                    continue

                m_rem = self.SECTION_LABELS["edges_removed"].match(line)
                if m_rem:
                    current["edges_removed"] = self._parse_edge_block(m_rem.group(1))
                    mode = None
                    i += 1
                    continue

                # Node features header toggles into node-feats mode
                m_nf_hdr = self.SECTION_LABELS["node_feats_hdr"].match(line)
                if m_nf_hdr:
                    mode = "node_feats"
                    i += 1
                    continue

                # Parse node feature lines while in node-feats mode
                if mode == "node_feats":
                    # Stop if we hit a blank line or a new section/proposal
                    if not line.strip() or self.PROPOSAL_HEADER_RE.match(line) or \
                    self.SECTION_LABELS["edges_added"].match(line) or \
                    self.SECTION_LABELS["edges_removed"].match(line) or \
                    self.SECTION_LABELS["node_feats_hdr"].match(line):
                        # Don't consume this line here; loop will re-process if needed
                        # (except blank line, which we do consume)
                        if not line.strip():
                            i += 1
                        continue

                    m_nf = self.NODE_FEAT_LINE_RE.match(line)
                    if m_nf:
                        node = m_nf.group(1)
                        feats = self._parse_feature_list(m_nf.group(2))
                        current["node_features_modified"][node] = feats
                        i += 1
                        continue
                    else:
                        # If line doesn't match a node-feat pattern, skip harmlessly
                        i += 1
                        continue

            i += 1

        # Flush last proposal
        if current:
            proposals.append(current)

        # Sort proposals by their number, just in case
        proposals.sort(key=lambda p: p.get("number", 0))
        return proposals
    
    def edit_graph(self, graph, proposals):
        new_data = np.copy(graph.data)
        new_features = np.copy(graph.node_features)

        edges_added = proposals[0]['edges_added']
        for edge in edges_added:
            new_data[int(edge['u'])][int(edge['v'])] = 1
            if edge['type'] == 'undirected':
                new_data[int(edge['v'])][int(edge['u'])] = 1

        edges_removed =  proposals[0]['edges_removed']
        for edge in edges_removed:
            new_data[int(edge['u'])][int(edge['v'])] = 0
            if edge['type'] == 'undirected':
                new_data[int(edge['v'])][int(edge['u'])] = 0

        node_features = proposals[0]['node_features_modified']
        for key in node_features.keys():
            if len( new_features[int(key)]) == len( node_features[key]):
                for i in range(0, len( node_features[key])):
                    new_features[int(key)][i] = float(node_features[key][i])

        new_g = GraphInstance(id=graph.id,
                                label=0,
                                data=new_data,
                                directed=graph.directed,
                                node_features= new_features)


        return new_g


   


    
    