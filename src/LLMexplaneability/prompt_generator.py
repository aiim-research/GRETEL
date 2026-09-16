from .feature_extractor import FeatureExtractor

class PromptGenerator:
    def __init__(self, GraphFeatures: FeatureExtractor, CounterfactualFeatures: FeatureExtractor, Domain: str):
        self.GraphFeatures = GraphFeatures
        self.CounterfactualFeatures = CounterfactualFeatures
        self.Domain = Domain
        
    def export_prompt(self, file = "salida.txt"):
        sys, prompt = self.generate_prompt()
        with open(file, "w", encoding="utf-8") as f:
            f.write(prompt)          # sobrescribe si existe


    def graph_to_text(self):
        return "GRAPH:\n" + self.graph_nodes_prompt() + "\n"  + self.graph_edges_prompt() + "\n" +  self.graph_node_features_prompt() + "\n" + "ORACLE PREDICTED CLASS:" + str(self.GraphFeatures.oracle_prediction) + "\n\n"
   
    def modifications_to_text(self):
        modifications = self.graph_modifications_prompt() 
        return "MODIFICATIONS FOR GENERATING COUNTERFACTUAL:\n" + modifications[0] + "\n" + modifications[1] + "\n" + modifications[2] + "\n" + "ORACLE PREDICTED CLASS:" + str(self.CounterfactualFeatures.oracle_prediction) + "\n\n"


    def generate_prompt(self, add_features = False):
        system = f"You are an analyst who explains classification changes in graphs to experts in the specified DOMAIN.\n\
Your task: explain, in DOMAIN terms, why the graph changed class after certain modifications (a counterfactual).\n\n\
\
RULES:\n\
- Do not invent data; use only the information provided in INPUT.\n\
- Explain using DOMAIN concepts, but ground your claims in graph evidence: modifications and features.\n\
- Prioritize the top 3-5 root causes, ordered by estimated impact.\n\
- For each cause, explicitly link:\n\
(a) the relevant modification(s) and/or feature change,\n\
(b) how that change affects a DOMAIN mechanism,\n\
(c) how that shifts the oracle's classification from the original to the counterfactual.\n\
- If something is ambiguous, say so and include a confidence tag (High/Medium/Low). Do not show your internal chain of thought—only conclusions and evidence.\n\
- If the graph is directed/weighted, respect direction/weights in your reasoning.\n\
- Keep the explanation concrete and avoid unnecessary jargon.\n\
\n\
OUTPUT FORMAT:\n\
1) 2-3 sentence summary of the class change (Original → Counterfactual) in DOMAIN terms.\n\
2) Main causes (numbered bullets in DOMAIN terms). For each:\n\
- Cause (DOMAIN terms)\n\
- Evidence (modification/feature → quantitative change)\n\
- Mechanism (why this pushes the class)\n\
- Confidence: High/Medium/Low\n\
3) Secondary signals (optional)\n\
4) Limitations/Assumptions (1-2 bullets)\n\
\n\
Now process the following INPUT:\n\n"

        system += self.Domain + "\n"
        prompt =  self.graph_to_text()   
        prompt += self.modifications_to_text()
        if add_features:
            features = self.graph_extracted_features_prompt()
            prompt += features[0] + "\n\n" + features[1]

        return system, prompt
    
    
    def graph_nodes_prompt(self) -> str:
        GraphNodes = "Nodes: {" if self.GraphFeatures.is_directed else "Vertex: {"
        for node in self.GraphFeatures.graph.nodes():
            GraphNodes += f"{node}, "
        GraphNodes = GraphNodes.rstrip(", ") + "}"
        return GraphNodes
    
    def graph_edges_prompt(self) -> str:
        GraphEdges = "Edges: {" if not self.GraphFeatures.is_directed else "Arcs: {"
        for node in self.GraphFeatures.graph.nodes():
            for neighbor in self.GraphFeatures.graph.neighbors(node):
                if self.GraphFeatures.is_directed:
                    GraphEdges += f"({node} -> {neighbor}), "
                else:
                    if node < neighbor:  # Avoid duplicates in undirected graphs
                        GraphEdges += f"({node} -- {neighbor}), "
        
        GraphEdges = GraphEdges.rstrip(", ") + "}"
        return GraphEdges

    def graph_node_features_prompt(self) -> str:
        GraphNodeFeatures = "Node Features:\n"  if self.GraphFeatures.is_directed else "Vertex Features:\n"
        for node in self.GraphFeatures.graph.nodes():
            features = self.GraphFeatures.graph.node_features[node]
            GraphNodeFeatures += f"{node}: {features.tolist()}\n"
        return GraphNodeFeatures
    
    def graph_modifications_prompt(self):
        EdgesAdded = "Edges Added: {" if not self.GraphFeatures.is_directed else "Arcs Added: {"
        for vertex in self.GraphFeatures.graph.nodes():
            for neig in self.CounterfactualFeatures.graph.neighbors(vertex):
                if neig not in self.GraphFeatures.graph.neighbors(vertex):
                    if self.GraphFeatures.is_directed:
                        EdgesAdded += f"({vertex} -> {neig}), "
                    elif vertex < neig:
                        EdgesAdded += f"({vertex} -- {neig}), "
        
        EdgesAdded = EdgesAdded.rstrip(", ") + "}"

        EdgesRemoved = "Edges Removed: {" if not self.GraphFeatures.is_directed else "Arcs Removed: {"
        for vertex in self.GraphFeatures.graph.nodes():
            for neig in self.GraphFeatures.graph.neighbors(vertex):
                if neig not in self.CounterfactualFeatures.graph.neighbors(vertex):
                    if self.GraphFeatures.is_directed:
                        EdgesRemoved += f"({vertex} -> {neig}), "
                    elif vertex < neig:
                        EdgesRemoved += f"({vertex} -- {neig}), "
        
        EdgesRemoved = EdgesRemoved.rstrip(", ") + "}"

        FeaturesModified = "Vertex Features Modified (vertex: new Feature list):\n" if not self.GraphFeatures.is_directed else "Node Features Modified (node: new Feature list):\n"
        for vertex in self.GraphFeatures.graph.nodes():
            for (feat, feat_mod) in zip(self.GraphFeatures.graph.node_features[vertex], self.CounterfactualFeatures.graph.node_features[vertex]):
                if feat != feat_mod:
                    FeaturesModified += f"{vertex}: {self.CounterfactualFeatures.graph.node_features[vertex].tolist()}\n"
                    break

        return EdgesAdded, EdgesRemoved, FeaturesModified
        
    def _graph_features_prompt(self, Features: FeatureExtractor, graph: str):
        return "FEATURES EXTRACTED " + graph +"\n" +\
                f"Max Degree: {Features.max_degree[0]}\n" +\
                f"Max Degree Nodes: {Features.max_degree[1]}\n" +\
                f"Min Degree: {Features.min_degree[0]}\n" +\
                f"Min Degree Nodes: {Features.min_degree[1]}\n" +\
                f"Conex Components: {Features.conex_components}\n" +\
                f"Graph Density: {Features.density}\n" +\
                f"Mean Degree: {Features.mean_degree}\n" +\
                f"Degree Variance: {Features.variance_degree}\n" +\
                f"Wedge Number: {Features.wedge_count}\n" +\
                f"Number of Trianglea: {Features.triangle_count}\n" +\
                f"Approximate Diameter: {Features.approximate_diameter}\n" +\
                f"Degree Asortativity: {Features.degree_asortativity}\n" +\
                f"Has Cycles: {Features.is_ciclic}"  



    def graph_extracted_features_prompt(self):
        features_input = self._graph_features_prompt(self.GraphFeatures, "Graph G INPUT")
        features_counterfactual = self._graph_features_prompt(self.CounterfactualFeatures, "Graph G' COUNTERFACTUAL")
        return features_input, features_counterfactual

        