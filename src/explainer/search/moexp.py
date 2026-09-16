from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.utils.cfg_utils import retake_oracle
import networkx as nx
import torch
from operator import itemgetter
import numpy as np
from typing import List, Tuple, Dict, Any


class GNNMOExp(Explainer):
    """
    GNN Multi-Objective Explanation (MOExp) explainer.

    Liu et al. Multi-objective explanations of GNN predictions. 
    In 2021 IEEE International Conference on Data Mining (ICDM) 2021 Dec 7 (pp. 409-418). IEEE.
    
    This explainer generates counterfactual explanations for graph neural networks
    using a multi-objective approach based on KL divergence and graph structure metrics.
    """
    
    def init(self):
        super().init()
        self.oracle = retake_oracle(self.local_config)
        self.local_params = self.local_config['parameters']
        self.L = self.oracle.local_config['parameters']['model']['parameters']['num_conv_layers']
        self.max_num_nodes = self.local_params['max_num_nodes']

        # Initialize loss functions and activation functions
        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.softmax = torch.nn.Softmax(dim=0)
        self.log_softmax = torch.nn.LogSoftmax(dim=0)
    
    def explain(self, instance: GraphInstance) -> GraphInstance:
        """
        Generate a counterfactual explanation for the given graph instance.
        
        Args:
            instance: The graph instance to explain
            
        Returns:
            A counterfactual graph instance that changes the prediction
        """
        # Extract features and create NetworkX graph
        node_features_matrix = instance.node_features
        G = nx.from_numpy_array(instance.data)
        
        # Get original prediction probabilities
        y_i = self.softmax(self.oracle.predict_proba(instance))
        log_y_i = self.log_softmax(self.oracle.predict_proba(instance))
        
        # Find node with maximum degree centrality
        degree_dict = nx.degree_centrality(G)
        selected_node_index = max(degree_dict, key=degree_dict.get)
        
        # Create DFS tree and convert to undirected graph
        G_i = nx.dfs_tree(G, source=selected_node_index, depth_limit=self.L)
        G_i = G_i.to_undirected()
        
        # Generate subgraphs using DFS edges
        dfs_edge_list = list(nx.dfs_edges(G, source=selected_node_index, depth_limit=self.L))
        G_i_list = self._build_subgraph_list(G, selected_node_index, dfs_edge_list)
        
        # Calculate metrics for each subgraph
        nu_list, mu_list, log_y_i_prime_list = self._calculate_subgraph_metrics(
            G_i_list, node_features_matrix, log_y_i
        )
        
        # Generate candidate explanations
        candidates = self._generate_candidates(G_i_list, nu_list, mu_list)
        
        # Select best explanation based on ranking
        explanations = self._select_best_explanations(candidates, nu_list, mu_list)
        
        # Create and return the first counterfactual
        return self._create_counterfactual_instance(
            explanations[0][1], node_features_matrix
        )

    def _build_subgraph_list(self, G: nx.Graph, selected_node_index: int, 
                           dfs_edge_list: List[Tuple[int, int]]) -> List[nx.Graph]:
        """
        Build list of subgraphs from DFS exploration.
        
        Args:
            G: Original graph
            selected_node_index: Starting node for DFS
            dfs_edge_list: List of edges from DFS traversal
            
        Returns:
            List of subgraphs
        """
        # Start with the single selected node
        G_i_list = [G.subgraph([selected_node_index])]
        
        # Add subgraphs progressively based on DFS edges
        for i in range(len(dfs_edge_list)):
            edges_in_subgraph = dfs_edge_list[:i+1]
            next_subgraph = nx.Graph()
            next_subgraph.add_edges_from(edges_in_subgraph)
            if len(next_subgraph.nodes) <= self.max_num_nodes:
                G_i_list.append(next_subgraph)
        
        return G_i_list

    def _calculate_subgraph_metrics(self, G_i_list: List[nx.Graph], 
                                  node_features_matrix: np.ndarray, 
                                  log_y_i: torch.Tensor) -> Tuple[List[float], List[float], List[torch.Tensor]]:
        """
        Calculate nu and mu metrics for each subgraph.
        
        Args:
            G_i_list: List of subgraphs
            node_features_matrix: Node features of original graph
            log_y_i: Log probabilities of original prediction
            
        Returns:
            Tuple of (nu_list, mu_list, log_y_i_prime_list)
        """
        nu_list, mu_list, log_y_i_prime_list = [], [], []
        
        for i, subgraph in enumerate(G_i_list):
            # Create graph instance for subgraph
            subgraph_instance = self._create_subgraph_instance(
                subgraph, node_features_matrix
            )
            
            # Get prediction for subgraph
            log_y_i_prime = self.log_softmax(self.oracle.predict_proba(subgraph_instance))
            log_y_i_prime_list.append(log_y_i_prime)
            
            # Calculate nu metric (KL divergence)
            nu_G_i = -(self.kl(log_y_i, log_y_i_prime) + self.kl(log_y_i_prime, log_y_i))
            nu_list.append(nu_G_i)
            
            # Calculate mu metrics for previous subgraphs
            if i > 0:
                for j in range(i):
                    mu_G_i_G_i_tilde = (nu_list[i] - nu_list[j]) / (len(subgraph.edges) - len(G_i_list[j].edges))
                    mu_list.append(mu_G_i_G_i_tilde)
        
        return nu_list, mu_list, log_y_i_prime_list

    def _create_subgraph_instance(self, subgraph: nx.Graph, 
                                node_features_matrix: np.ndarray) -> GraphInstance:
        """
        Create a GraphInstance from a subgraph.
        
        Args:
            subgraph: NetworkX subgraph
            node_features_matrix: Node features of original graph
            
        Returns:
            GraphInstance for the subgraph
        """
        subgraph_adjacency_matrix = nx.adjacency_matrix(subgraph).toarray() + np.transpose(nx.adjacency_matrix(subgraph).toarray())
        subgraph_node_features = [node_features_matrix[j] for j in subgraph.nodes]
        return GraphInstance(
            id=-len(subgraph.nodes), 
            data=np.array(subgraph_adjacency_matrix), 
            node_features=np.array(subgraph_node_features), 
            label=0
        )

    def _generate_candidates(self, G_i_list: List[nx.Graph], 
                           nu_list: List[float], mu_list: List[float]) -> List[List]:
        """
        Generate candidate explanations from subgraph pairs.
        
        Args:
            G_i_list: List of subgraphs
            nu_list: List of nu metrics
            mu_list: List of mu metrics
            
        Returns:
            List of candidate explanations
        """
        candidates = []
        
        # Skip first subgraph as it has no predecessors
        for i in range(1, len(G_i_list)):
            for j in range(i):
                candidate = [G_i_list[i], G_i_list[j], nu_list[i]]
                mu_G_i_G_i_tilde = (nu_list[i] - nu_list[j]) / (len(G_i_list[i].edges) - len(G_i_list[j].edges))
                candidate.append(mu_G_i_G_i_tilde)
                candidates.append(candidate)
        
        return candidates

    def _select_best_explanations(self, candidates: List[List], 
                                nu_list: List[float], mu_list: List[float]) -> List[List]:
        """
        Select the best explanations based on ranking.
        
        Args:
            candidates: List of candidate explanations
            nu_list: List of nu metrics
            mu_list: List of mu metrics
            
        Returns:
            List of best explanations
        """
        # Sort metrics for ranking
        ordered_nu_list = sorted(nu_list, reverse=True)
        ordered_mu_list = sorted(mu_list, reverse=True)
        
        # Add ranking score R to each candidate
        for candidate in candidates:
            r_score = ordered_nu_list.index(candidate[2]) + ordered_mu_list.index(candidate[3])
            candidate.append(r_score)
        
        # Sort by ranking score (lower R = higher metrics)
        ordered_candidates = sorted(candidates, key=itemgetter(4))
        
        # Return all candidates with the best ranking score
        best_rank = ordered_candidates[0][4]
        return [candidate for candidate in ordered_candidates if candidate[4] == best_rank]

    def _create_counterfactual_instance(self, counterfactual_graph: nx.Graph, 
                                      node_features_matrix: np.ndarray) -> GraphInstance:
        """
        Create a GraphInstance for the counterfactual explanation.
        
        Args:
            counterfactual_graph: The counterfactual subgraph
            node_features_matrix: Node features of original graph
            
        Returns:
            GraphInstance for the counterfactual
        """
        adjacency_matrix = nx.adjacency_matrix(counterfactual_graph).toarray()
        node_features = [node_features_matrix[j] for j in counterfactual_graph.nodes]
        return GraphInstance(
            id=0, 
            data=np.array(adjacency_matrix), 
            node_features=np.array(node_features), 
            label=0
        )

    def check_configuration(self):
        """Validate and set default configuration parameters."""
        super().check_configuration()
        self.local_config['parameters']['max_num_nodes'] = self.local_config['parameters'].get(
            'max_num_nodes', self.dataset.num_nodes
        )

