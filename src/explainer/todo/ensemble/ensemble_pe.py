import time
import numpy as np
import networkx as nx
import itertools
import scipy.sparse as sp

from abc import ABC
from typing import List
from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.explainer.ensemble.ensemble_bagging import EnsembleBaggingExplainer
from src.core.oracle_base import Oracle


class PEEnsembleExplainer(EnsembleBaggingExplainer):
    def __init__(self, id, config_dict=None, weak_explainers=None) -> None:
        super().__init__(id, config_dict, weak_explainers)
        self._name = "pe_ensemble"
        self._config_dict = config_dict
        self._proba_model: sp.spmatrix = None

        enable_params = config_dict is not None and "parameters" in config_dict.keys()
        param_dict = config_dict["parameters"] if enable_params else None

        self._learning_rate = (
            param_dict["learning_rate"]
            if enable_params and "learning_rate" in param_dict.keys()
            else 0.5
        )
        self._population_size = (
            param_dict["population_size"]
            if enable_params and "population_size" in param_dict.keys()
            else 16
        )
        self._select_ratio = (
            param_dict["select_ratio"]
            if enable_params and "select_ratio" in param_dict.keys()
            else 0.5
        )

        # stop conditions
        self._timeout = (
            param_dict["timeout"]
            if enable_params and "timeout" in param_dict.keys()
            else 100
        )  # in seconds
        self._win_threshold = (
            param_dict["win_threshold"]
            if enable_params and "win_threshold" in param_dict.keys()
            else 0.9
        )

        # random seed
        self._seed = (
            param_dict["seed"]
            if enable_params and "seed" in param_dict.keys()
            else None
        )

    @property
    def proba_model(self):
        return self._proba_model

    @proba_model.setter
    def proba_model(self, proba_model):
        self._proba_model = proba_model

    def __evaluate_cf__(self, graph, oracle: Oracle, original_prediction):
        instance = DataInstance(graph=graph)
        prediction = oracle.predict(instance)
        proba_estimates = oracle.predict_proba(instance)

        if (prediction != original_prediction):
            print("Found counterfactual")

        # here we need to include some metric for graph similarity (thus optimizing minimality)
        # so far our metric is quite simple, max of another class minus original class estimated probability
        
        estimates_index = [(proba_estimates[i], i) for i in range(len(proba_estimates))]
        original_class_proba = estimates_index.pop(original_prediction)[0]

        estimates_index.sort(key= lambda x: x[1], reverse = True)
        max_estimate = estimates_index[0][0]
        return max_estimate - original_class_proba

    def __build_graph__(self, sample, instance: DataInstance):
        graph: nx.Graph = instance.graph.copy()
        g = nx.Graph(sample)
        return g

    def __make_symetric__(self, matrix):
        symm = np.maximum(matrix, matrix.transpose())
        return symm

    def _sample_(self):
        # samples are supposed to be adj matrices
        samples = []
        for i in range(self._population_size):
            proba_model = self._proba_model.toarray()
            random_values = np.random.random(self._proba_model.shape)
            sample = np.where(random_values >= proba_model, 1, 0)

            # ignore values in lower triangular matrix
            triupper = sp.triu(sample)
            triupper.setdiag(1)

            # save symmetrical matrix (we are working with undirected graphs)
            symm = np.asmatrix(self.__make_symetric__(triupper.toarray()))
            samples.append(sp.csr_matrix(symm, symm.shape))

        return samples

    def _rank_(
        self, samples: List[sp.spmatrix], instance: DataInstance, oracle: Oracle
    ):
        original_prediction = oracle.predict(instance)

        e_samples = [
            (
                sample,
                self.__evaluate_cf__(
                    self.__build_graph__(sample, instance), oracle, original_prediction
                ),
            )
            for sample in samples
        ]

        e_samples.sort(key = lambda x: x[1])
        return e_samples

    def _build_proba_model_(self, samples: List[sp.spmatrix]):
        model = samples[0]
        for sample in samples[1:]:
            model += sample

        model = model * (1 / len(samples))
        return model

    def _interpolate_proba_models_(self, original_model, new_model):
        om = original_model.toarray()
        nm = new_model.toarray()
        return sp.csr_matrix(om + (nm - om) * self._learning_rate)

    def aggregate(
        self,
        instance: DataInstance,
        oracle: Oracle,
        dataset: Dataset,
        explanations: List[DataInstance],
    ):
        # sum up all edges occurences
        adj_sum: sp.spmatrix = nx.adjacency_matrix(explanations[0].graph).tocsr()
        for instance in explanations[1:]:
            adj_sum += nx.adjacency_matrix(instance.graph).tocsr()

        # add regularization (enable creating unforeseen edges)
        ones = sp.coo_matrix(np.ones(adj_sum.shape)).tocsr()
        adj_sum += ones

        # turn into probabilistic model
        adj_sum *= 1 / (len(explanations) + 2)
        adj_sum.setdiag(1)
        return adj_sum

    def explain_aggregate(
        self,
        instance: DataInstance,
        oracle: Oracle,
        dataset: Dataset,
        explanations: List[DataInstance],
        aggregate,
    ):
        init_time = time.time()
        current_time = time.time()

        # set initial probabilistic model
        self.proba_model = aggregate

        # set baseline
        baseline = []
        baseline = self._rank_([nx.adjacency_matrix(instance.graph).tocsr()], instance, oracle)[0]
        
        best_explanation = None #(baseline[0], baseline[1])
        cicle_count = 0
        while current_time - init_time < self._timeout:
            # optimize
            pop = self._sample_()
            ranked_pop = self._rank_(pop, instance, oracle)

            # update best explanation found so far
            if (best_explanation is None or ranked_pop[0][1] > best_explanation[1]):
                best_explanation = ranked_pop[0]

                # good enough stop condition
                if (best_explanation[1] >= self._win_threshold):
                    break

            proba_model = self._build_proba_model_(
                [x[0] for x in ranked_pop[: int(self._select_ratio * self._population_size)]]
            )

            self.proba_model = self._interpolate_proba_models_(self.proba_model, proba_model)

            cicle_count+=1
            current_time = time.time()

        print(cicle_count)
        return DataInstance(graph = nx.Graph(best_explanation[0]))
