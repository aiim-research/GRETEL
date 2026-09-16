# GRETEL: Graph Counterfactual Explanation Evaluation Framework

[![discord](https://img.shields.io/badge/Discord-blue?style=for-the-badge)](https://discord.gg/TdZWBDg7)
[![linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/artificial-intelligence-information-mining)
[![github](https://img.shields.io/github/stars/aiim-research/GRETEL?style=for-the-badge)](#)
[![python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge)](https://docs.python.org/release/3.9.0/)

GRETEL is an open-source framework for developing and evaluating Graph Counterfactual Explanation (GCE) methods. It is the code behind several published papers and is maintained as a platform others can extend.

Work on Graph Counterfactual Explanations diverges in problem definition, application domain, test data and evaluation metrics, and most papers do not compare exhaustively against the alternatives. GRETEL exists to make that comparison possible: datasets, ML models, explanation techniques and evaluation measures are all pluggable components, selected and parameterised from a single configuration file.

## Quick start

```bash
git clone https://github.com/aiim-research/GRETEL.git && cd GRETEL
conda env create -f environment.yml && conda activate GRETEL

# run one experiment: ASD, DCE generator, LocalSearch minimizer, fold 0
python main.py lab/config/generate_minimize/asd/dce/dce-lcls/generate_minimize0.jsonc 1
```

Results appear under `lab/output/results/<scope>/`. For CUDA, use `./scripts/setup-grtl-gpu.sh` instead of the conda step.

**[docs/reproducing-experiments.md](docs/reproducing-experiments.md)** is the guide: environment, data, how a configuration is assembled, how to run a batch, and how to read the result store. **[docs/README.md](docs/README.md)** maps the rest of the repository.

## Repository layout

```
src/        the framework: dataset/ oracle/ embedder/ explainer/ evaluation/ core/
lab/        the experiment workbench: config/ notebooks/ data/cache/ graphics/
scripts/    command-line entry points (runners, artefact trainers, figures)
tools/      repository integrity checks
tests/      smoke tests: the current matrix, and the published baselines
data/       the datasets that ship with the repository
docs/       this documentation
legacy/     earlier phases, kept reproducible
```

Each of `lab/`, `scripts/`, `tools/`, `legacy/` and the `legacy/` subpackages inside `src/` carries a README explaining what is live in it and what is kept for the record.

## How it works

A component is a class plus a parameter dictionary, named in the config by its dotted path:

```jsonc
"explainer": {
  "class": "src.explainer.future.search.dces.DCESExplainer",
  "parameters": { "epochs": 500 }
}
```

A factory instantiates it, and `Context` derives everything else. In particular, every cache entry, saved artefact and result directory is named by an MD5 of the component's resolved configuration, so `ASD-15273954d84e872cf0b021cd4477bfdc` identifies one dataset built one specific way. Two runs that agree on every parameter share a cache; a run that differs anywhere gets its own. Nothing has to be cleaned between experiments.

The consequence to know about before editing: a module's path is part of that identity. See [tools/README.md](tools/README.md).

Adding a component means writing the class and naming it in a config. Nothing needs to be registered.

## What ships with the framework

**Datasets.** Tree-Cycles and Tree-Infinity (synthetic), ASD and ADHD (brain networks, [4]), BBBP and HIV (molecules, [5]), plus any [TU dataset](https://chrsmrrs.github.io/datasets/) by name (PROTEINS, ENZYMES, BZR, AIDS, COLORS-3, Synthie, IMDB-BINARY, Cuneiform), downloaded on first use.

**Oracles.** KNN, SVM, GCN, and custom oracles for ASD [4] and Tree-Cycles (the latter is exact by construction).

**Explainers.**

| Method | What it does |
|---|---|
| DCE Search | searches the dataset for a counterfactual instance. Makes no assumption about the data, so it serves as the baseline |
| OBS / DDBS [4] | oblivious and data-driven bidirectional search, two-stage heuristics |
| Local Bounded Search | bounded local search over edge and attribute operations, the subject of the current revision |
| MACCS [5] | molecule-specific, counterfactual compounds with STONED |
| MEG [6] | reinforcement learning over molecular graphs |
| CFF [7] | learned perturbation masks from counterfactual and factual reasoning |
| CLEAR [8] | generative counterfactual explanations on graphs |
| CounteRGAN [9] | a GAN-based image method ported to graphs |
| Ensembles | aggregate several explainers by union, intersection, frequency, multi-criteria selection and others |

The older baselines are shipped, not archived: their configurations live under `legacy/config-v2/` because that is the generation they were published with, and `python tests/catalogue_smoke.py` runs one instance through each of them so they cannot rot unnoticed.

**Metrics.** Graph Edit Distance, Feature Edit Distance, Correctness, Sparsity, Fidelity, Oracle Calls, Oracle Accuracy, Runtime, Instability.

## Team

* Prof. Giovanni Stilo (project leader and investigator)
* Mario Alfonso Prado Romero (co-principal investigator)
* Dr. Bardh Prenkaj (co-principal investigator)
* Andrea D'Angelo (notable investigator)
* Efstratios Zaradoukas (contributor)
* Alessandro Celi (administrative staff)

Past contributors: Hiram Borbolla Hernández, Roberto Marti Cedeño, Ernesto Estevanell-Valladares, Daniel Alejandro Valdés-Pérez.

Further documentation lives in the [GRETEL wiki](https://github.com/aiim-research/GRETEL/wiki).

## Citing GRETEL

`CITATION.cff` carries the machine-readable metadata. Please cite the framework paper if you use GRETEL:

```bibtex
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

```bibtex
@inproceedings{prado-romero2023developing,
  author = {Prado-Romero, Mario Alfonso and Prenkaj, Bardh and Stilo, Giovanni},
  title = {Developing and Evaluating Graph Counterfactual Explanation with GRETEL},
  year = {2023},
  isbn = {9781450394079},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3539597.3573026},
  booktitle = {Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
  pages = {1180--1183},
  location = {Singapore, Singapore},
  series = {WSDM '23}
}
```

```bibtex
@article{prado-romero2023survey,
  author = {Prado-Romero, Mario Alfonso and Prenkaj, Bardh and Stilo, Giovanni and Giannotti, Fosca},
  title = {A Survey on Graph Counterfactual Explanations: Definitions, Methods, Evaluation, and Research Challenges},
  year = {2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  issn = {0360-0300},
  doi = {10.1145/3618105},
  journal = {ACM Comput. Surv.},
  month = {sep}
}
```

## References

1. Prado-Romero, M.A. and Stilo, G., 2022. GRETEL: Graph counterfactual explanation evaluation framework. CIKM '22, 4389-4393.
2. Prado-Romero, M.A., Prenkaj, B. and Stilo, G., 2023. Developing and Evaluating Graph Counterfactual Explanation with GRETEL. WSDM '23, 1180-1183.
3. Ying, Z., Bourgeois, D., You, J., Zitnik, M. and Leskovec, J., 2019. GNNExplainer: Generating explanations for graph neural networks. NeurIPS 32.
4. Abrate, C. and Bonchi, F., 2021. Counterfactual Graphs for Explainable Classification of Brain Networks. KDD '21, 2495-2504.
5. Wellawatte, G.P., Seshadri, A. and White, A.D., 2022. Model agnostic generation of counterfactual explanations for molecules. Chemical Science 13(13), 3697-3705.
6. Numeroso, D. and Bacciu, D., 2021. MEG: Generating molecular counterfactual explanations for deep graph networks. IJCNN 2021, 1-8.
7. Tan, J., Geng, S., Fu, Z., Ge, Y., Xu, S., Li, Y. and Zhang, Y., 2022. Learning and evaluating graph neural network explanations based on counterfactual and factual reasoning. WWW '22, 1018-1027.
8. Ma, J., Guo, R., Mishra, S., Zhang, A. and Li, J., 2022. CLEAR: Generative counterfactual explanations on graphs. NeurIPS 35, 25895-25907.
9. Nemirovsky, D., Thiebaut, N., Xu, Y. and Gupta, A., 2022. CounteRGAN: Generating counterfactuals for real-time recourse and interpretability using residual GANs. UAI 2022, 1488-1497.

## License

See [LICENSE](LICENSE).
