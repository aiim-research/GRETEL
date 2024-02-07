README for dataset Github Stargazers


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

There are OPTIONAL files if the respective information is available:

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description of the dataset === 

Github Stargazers

The social networks of developers who starred popular machine learning and web development repositories (with at least 10 stars) until 2019 August. Nodes are users and links are follower relationships. The task is to decide whether a social network belongs to web or machine learning developers. We only included the largest component (at least with 10 users) of graphs.

Properties

- Number of graphs: 12,725
- Directed: No.
- Node features: No.
- Edge features: No.
- Graph labels: Yes. Binary-labeled.
- Temporal: No.

|  Stats   |  Min  |  Max  |
|   ---    |  ---  |  ---  |
|  Nodes   |   10  |  957  | 
| Density  | 0.003 | 0.561 | 
| Diameter |   2   |   18  | 


=== Source ===

If you find this dataset useful in your research, please consider citing the following paper:

>@misc{karateclub2020,
       title={An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs},
       author={Benedek Rozemberczki and Oliver Kiss and Rik Sarkar},
       year={2020},
       eprint={2003.04819},
       archivePrefix={arXiv},
       primaryClass={cs.LG}
}

And take a look at the project itself:

https://github.com/benedekrozemberczki/karateclub
