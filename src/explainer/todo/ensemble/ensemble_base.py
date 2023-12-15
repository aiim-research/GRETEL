from typing import List
from src.core.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle

from abc import ABC


class EnsembleExplainer(Explainer):

    def __init__(self, id, config_dict=None, weak_explainers: Explainer=None) -> None:
        super().__init__(id)
        self._id = id
        self._name = 'ensemble_explainer'
        self._config_dict=config_dict
        self._weak_explainers=weak_explainers