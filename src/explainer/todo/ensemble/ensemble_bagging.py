from typing import List
from src.explainer.ensemble.ensemble_base import EnsembleExplainer
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle

import sklearn

from abc import ABC, abstractmethod


class EnsembleBaggingExplainer(EnsembleExplainer):

    def __init__(self, id, config_dict=None, weak_explainers=None) -> None:
        super().__init__(id, config_dict, weak_explainers)
        self._name = 'ensemble_bagging_explainer'

        enable_params = config_dict is not None and "parameters" in config_dict.keys()
        param_dict = config_dict["parameters"] if enable_params else None

        # sampling_multiplier should be less or equal to the amount of week explainers and higher or equal to 1
        self._sampling_multiplier = param_dict["sampling_multiplier"] if enable_params and "sampling_multiplier" in param_dict.keys() else 1

    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        n_instances = len(dataset.instances)
        n_explainers = len(self._weak_explainers)
        n_samples = int(self._sampling_multiplier * n_instances/n_explainers)

        samples = [sklearn.utils.resample(dataset.instances, n_samples = n_samples)] * n_explainers
        
        datasets = []
        for i in range(len(samples)):
            sub_dataset = Dataset(f"subdataset_{i}", dataset._config_dict)
            sub_dataset.instances = samples[i]
            datasets.append(sub_dataset)
        
        cfs = [self._weak_explainers[i].explain(instance, oracle, datasets[i]) for i in range(n_explainers)]
        aggregate = self.aggregate(instance, oracle, dataset, cfs)
        return self.explain_aggregate(instance, oracle, dataset, cfs, aggregate)

    @abstractmethod
    def aggregate(self, instance, oracle, dataset, *explanations):
        pass

    @abstractmethod    
    def explain_aggregate(self, instance, oracle, dataset, *explanations, aggregate):
        pass
