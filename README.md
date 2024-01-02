# GRETEL (v2.0): Graph Counterfactual Explanation Evaluation Framework
[![discord](https://img.shields.io/badge/Discord-blue?style=for-the-badge)](https://discord.gg/VhRmBt9aAH)
[![linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/artificial-intelligence-information-mining)
[![github](https://img.shields.io/github/stars/aiim-research/GRETEL?style=for-the-badge)](#)
[![python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge)](https://docs.python.org/release/3.9.0/)



## General Information:
Machine Learning (ML) systems are a building part of the modern tools that impact our daily life in several application domains. Due to their black-box nature, those systems are hardly adopted in application domains (e.g. health, finance) where understanding the decision process is of paramount importance. Explanation methods were developed to explain how the ML model has taken a specific decision for a given case/instance. Graph Counterfactual Explanations (GCE) is one of the explanation techniques adopted in the Graph Learning domain. The existing works on Graph Counterfactual Explanations diverge mostly in the problem definition, application domain, test data, and evaluation metrics, and most existing works do not compare exhaustively against other counterfactual explanation techniques present in the literature. Here, we release GRETEL [1,2], a unified framework to develop and test GCE methods in several settings. GRETEL [1,2] is an open-source framework for Evaluating Graph Counterfactual Explanation Methods. It is implemented using the Object-Oriented paradigm and the Factory Method design pattern. Our main goal is to create a generic platform that allows the researchers to speed up the process of developing and testing new Graph Counterfactual Explanation Methods.
GRETEL is a highly extensible evaluation framework that promotes Open Science and the reproducibility of the evaluation by providing a set of well-defined mechanisms to integrate and manage easily: both real and synthetic datasets, ML models, state-of-the-art explanation techniques, and evaluation measures. 


## Table of Contents

* [Team Information](#team-information)
* [General Information](#general-information)
* [Notable References](#citation-request)
* [Requirements](#requirements)
* [Installation](#installation)
* [Resources Provided with the Framework](#resources-provided-with-the-framework)
* [How to Use](#how-to-use)
* [References](#references)

## Team Information:
* Prof. Giovanni Stilo [project leader/investigator]
* Mario Alfonso Prado Romero [principal investigator]
* Dr. Bardh Prenkaj [notable investigator]
* Hiram Borbolla Hernández [notable contributor]
* Roberto Marti Cedeño [notable contributor]
* Alessandro Celi [administrative staff]

### Past Contributors:
* Ernesto Estevanell-Valladares [contributor]
* Daniel Alejandro Valdés-Pérez [contributor]

## General Information:

GRETEL [1, 2] is an open source framework for Evaluating Graph Counterfactual Explanation Methods. It is implemented using the Object Oriented paradigm and the Factory Method design pattern. Our main goal is to create a generic platform that allows the researchers to speed up the process of developing and testing new Graph Counterfactual Explanation Methods.

## Welcomed Citations:

Please cite our papers if you use GRETEL in your projects:

Mario Alfonso Prado-Romero and Giovanni Stilo. 2022. GRETEL: Graph Counterfactual Explanation Evaluation Framework. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM '22). Association for Computing Machinery, New York, NY, USA. [https://doi.org/10.1145/3511808.3557608](https://doi.org/10.1145/3511808.3557608)

Mario Alfonso Prado-Romero, Bardh Prenkaj, and Giovanni Stilo. 2023. Developing and Evaluating Graph Counterfactual Explanation with GRETEL. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (WSDM '23). Association for Computing Machinery, New York, NY, USA, 1180–1183.  [https://doi.org/10.1145/3539597.3573026](https://doi.org/10.1145/3539597.3573026)

Mario Alfonso Prado-Romero, Bardh Prenkaj, Giovanni Stilo, and Fosca Giannotti. 2023. A Survey on Graph Counterfactual Explanations: Definitions, Methods, Evaluation, and Research Challenges. ACM Comput. Surv. Just Accepted (September 2023). [https://doi.org/10.1145/3618105](https://doi.org/10.1145/3618105)

```latex:
@inproceedings{prado-romero2022gretel,
  title={GRETEL: Graph Counterfactual Explanation Evaluation Framework},
  author={Prado-Romero, Mario Alfonso and Stilo, Giovanni},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  isbn = {9781450392365},
  year={2022},
  doi = {10.1145/3511808.3557608},
  booktitle={Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  location = {Atlanta, GA, USA},
  series = {CIKM '22}
}
```

```latex:
@inproceedings{prado-romero2023developing,
author = {Prado-Romero, Mario Alfonso and Prenkaj, Bardh and Stilo, Giovanni},
title = {Developing and Evaluating Graph Counterfactual Explanation with GRETEL},
year = {2023},
isbn = {9781450394079},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3539597.3573026},
booktitle = {Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
pages = {1180–1183},
location = {Singapore, Singapore},
series = {WSDM '23}
} 
```

```latex:
@article{prado-romero2023survey,
author = {Prado-Romero, Mario Alfonso and Prenkaj, Bardh and Stilo, Giovanni and Giannotti, Fosca},
title = {A Survey on Graph Counterfactual Explanations: Definitions, Methods, Evaluation, and Research Challenges},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {0360-0300},
url = {https://doi.org/10.1145/3618105},
doi = {10.1145/3618105},
journal = {ACM Comput. Surv.},
month = {sep}
}
```
<!-- 
## Requirements:
* pytorch
* picologging
* exmol
* gensim
* joblib
* jsonpickle
* karateclub
* matplotlib
* networkx
* numpy
* pandas
* rdkit
* scikit-learn
* scipy
* selfies
* sqlalchemy
* black
* typing-extensions
* torch_geometric
* dgl
* flufl.lock
* jsonc-parser
-->
## First steps with GRETEL:

### Prepare the enviroment:
We suggest proceeding to create a (CONDA) environment in the following way (reffering also to the script launchers/env_install.sh):
```
conda update -n base -c defaults conda -y
conda create -n GRTL python=3.9 -y
```

Activate the created env:
```
conda activate GRTL
```

Install first Pytorch and then the other dependencies:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install picologging exmol gensim joblib jsonpickle karateclub matplotlib networkx numpy pandas rdkit scikit-learn scipy selfies sqlalchemy black typing-extensions torch_geometric dgl IPython ipykernel flufl.lock jsonc-parser

```

### How to run an experiment
The initialization mechanism in GRETEL 2.0 was completely refactored.
A slightly different logic comes in place with a more robust and flexible mechanism.

Thus, new configuration files are needed. Even if they are not yet final, you can take a look at the folder config.

To run a configuration (configure the enviroment upfront):
```
python main.py <CONFIG_FILE>
```

<!-- 
## Installation:
The easiest way to get Gretel up and running with all the dependencies is to pull the development Docker image available in [Docker Hub](https://hub.docker.com/):

```
docker pull gretel/gretel:latest
```

The image is based on `tensorflow/tensorflow:latest-gpu` and it's GPU ready. In order to setup the container we recommend you to run:

```
docker-compose run gretel
```

For simplicity we provide several **makefile** rules for easy interaction with the Docker interface:

 * `make docker` - builds the development image from scratch
 * `make pull` - pull the development image
 * `make push` - push the development image
 * `make demo` - run the demo in the development image.-->

## Resources provided with the Framework:

### Datasets:

* **Tree-Cycles** [3]: Synthetic data set where each instance is a graph. The instance can be either a tree or a tree with several cycle patterns connected to the main graph by one edge

* **Tree-Infinity**: It follows the approach of the Tree-Cycles, but instead of cycles, there is an infinity shape.

* **ASD** [4]: Autism Spectrum Disorder (ASD) taken from the Autism Brain Imagine Data Exchange (ABIDE).

* **ADHD** [4]: Attention Deficit Hyperactivity Disorder (ADHD), is taken from the USC Multimodal Connectivity Database (USCD).

* **BBBP** [5]: Blood-Brain Barrier Permeation is a molecular dataset. Predicting if a molecule can permeate the blood-brain barrier.

* **HIV** [5]: It is a molecular dataset that classifies compounds based on their ability to inhibit HIV.


### Oracles:

* **KNN**

* **SVM**

* **GCN**

* **ASD Custom Oracle** [4] (Rules specific for the ASD dataset)

* **Tree-Cycles Custom Oracle** (Guarantees 100% accuracy on Tree-Cycles dataset)


### Explainers:

* **DCE Search**: Distribution Compliant Explanation Search,  mainly used as a baseline, does not make any assumption about the underlying dataset and searches for a counterfactual instance in it.

* **Oblivious Bidirectional Search (OBS)** [4]: It is an heuristic explanation method that uses a 2-stage approach.

* **Data-Driven Bidirectional Search (DDBS)** [4]: It follows the same logic as OBS. The main difference is that this method uses the probability (computed on the original dataset) of each edge to appear in a graph of a certain class to drive the counterfactual search process.

* **MACCS** [5]: Model Agnostic Counterfactual Compounds with STONED (MACCS) is specifically designed to work with molecules.

* **MEG** [6]: Molecular Explanation Generator is an RL-based explainer for molecular graphs.

* **CFF** [7] Is a learning-based method that uses Counterfactual and Factual Reasoning in the perturbation mask generation process.

* **CLEAR** [8] is a learning based explanation method that provides Generative Counterfactual Explanations on Graphs.

* **CounteRGAN** [9] is a porting of a GAN-based explanation method for images

<!-- 
First, we need to create a config json file with the option we want to use in our experiment. In the file config/CIKM/manager_config_example_all.json it is possible to find all options for each componnent of the framework.

```json
{
    "store_paths": [
        {"name": "dataset_store_path", "address": "/NFSHOME/mprado/CODE/GRETEL/data/datasets/"},
        {"name": "embedder_store_path", "address": "/NFSHOME/mprado/CODE/GRETEL/data/embedders/"},
        {"name": "oracle_store_path", "address": "/NFSHOME/mprado/CODE/GRETEL/data/oracles/"},
        {"name": "explainer_store_path", "address": "/NFSHOME/mprado/CODE/GRETEL/data/explainers/"},
        {"name": "output_store_path", "address": "/NFSHOME/mprado/CODE/GRETEL/output/"}
    ],
    "datasets": [
        {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 300, "n_in_cycles": 200} },
        {"name": "tree-cycles-balanced", "parameters": {"n_inst_class": 250, "n_per_inst": 300, "n_in_cycles": 200} },
        {"name": "tree-cycles-dummy", "parameters": {"n_inst_class": 250, "n_per_inst": 300, "n_in_cycles": 200} },
        {"name": "autism", "parameters": {} },
        {"name": "adhd", "parameters": {} },
        {"name": "tree-infinity", "parameters": {"n_inst": 500, "n_per_inst": 300, "n_infinities": 10, "n_broken_infinities": 10}},
        {"name": "bbbp", "parameters": {"force_fixed_nodes": true}},
        {"name": "bbbp", "parameters": {"force_fixed_nodes": false}},
        {"name": "hiv", "parameters": {"force_fixed_nodes": false}}
    ],
    "oracles": [
        {"name": "knn", "parameters": { "embedder": {"name": "graph2vec", "parameters": {} }, "k": 5 } },
        {"name": "svm", "parameters": { "embedder": {"name": "graph2vec", "parameters": {} } } },
        {"name": "asd_custom_oracle", "parameters": {} },
        {"name": "svm", "parameters": { "embedder": {"name": "rdk_fingerprint", "parameters": {} } } },
        {"name": "gcn-tf", "parameters": {} }
    ],
    "explainers": [
        {"name": "dce_search", "parameters":{"graph_distance": {"name": "graph_edit_distance", "parameters": {}} } },
        {"name": "dce_search_oracleless", "parameters":{"graph_distance": {"name": "graph_edit_distance", "parameters": {}} } },
        {"name": "bidirectional_oblivious_search", "parameters":{"graph_distance": {"name": "graph_edit_distance", "parameters": {}} } },
        {"name": "bidirectional_data-driven_search", "parameters":{"graph_distance": {"name": "graph_edit_distance", "parameters": {}} } },
        {"name": "maccs", "parameters":{"graph_distance": {"name": "graph_edit_distance", "parameters": {}} } }
    ],
    "evaluation_metrics": [ 
        {"name": "graph_edit_distance", "parameters": {}},
        {"name": "oracle_calls", "parameters": {}},
        {"name": "correctness", "parameters": {}},
        {"name": "sparsity", "parameters": {}},
        {"name": "fidelity", "parameters": {}},
        {"name": "oracle_accuracy", "parameters": {}}
    ]
}
```

Then to execute the experiment from the main the code would be something like this:

```python
from src.evaluation.evaluator_manager import EvaluatorManager

config_file_path = '/NFSHOME/mprado/CODE/Themis/config/linux-server/set-1/config_autism_custom-oracle_dce.json'

print('Creating the evaluation manager.......................................................')
eval_manager = EvaluatorManager(config_file_path, run_number=0)

print('Creating the evaluators...................................................................')
eval_manager.create_evaluators()

print('Evaluating the explainers..................................................................')
eval_manager.evaluate()
```

Once the result json files are generated it is possible to use the result_stats.py module to generate the tables with the results of the experiments. The tables will be generated as CSV and LaTex. In the examples folder there are some jupyter notebooks, and associated configuration files, that show how to use the framework for evaluating an explainer. Furthermore, they show how to extend GRETEL with new datasets and explainers.
-->

## References

1. Prado-Romero, M.A. and Stilo, G., 2022, October. Gretel: Graph counterfactual explanation evaluation framework. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (pp. 4389-4393).

2. Prado-Romero, M.A., Prenkaj, B. and Stilo, G., 2023, February. Developing and Evaluating Graph Counterfactual Explanation with GRETEL. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (pp. 1180-1183).

3. Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. 2019. Gnnexplainer: Generating explanations for graph neural networks. Ad-
vances in neural information processing systems 32 (2019)

4. Carlo Abrate and Francesco Bonchi. 2021. Counterfactual Graphs for Explainable Classification of Brain Networks. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2495–2504

5. Geemi P Wellawatte, Aditi Seshadri, and Andrew D White. 2022. Model agnostic generation of counterfactual explanations for molecules. Chemical science 13, 13
(2022), 3697–370

6. Numeroso, D. and Bacciu, D., 2021, July. Meg: Generating molecular counterfactual explanations for deep graph networks. In 2021 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.

7. Tan, J., Geng, S., Fu, Z., Ge, Y., Xu, S., Li, Y. and Zhang, Y., 2022, April. Learning and evaluating graph neural network explanations based on counterfactual and factual reasoning. In Proceedings of the ACM Web Conference 2022 (pp. 1018-1027).

8. Ma, J., Guo, R., Mishra, S., Zhang, A. and Li, J., 2022. Clear: Generative counterfactual explanations on graphs. Advances in Neural Information Processing Systems, 35, pp.25895-25907.

9. Nemirovsky, D., Thiebaut, N., Xu, Y. and Gupta, A., 2022, August. CounteRGAN: Generating counterfactuals for real-time recourse and interpretability using residual GANs. In Uncertainty in Artificial Intelligence (pp. 1488-1497). PMLR.
