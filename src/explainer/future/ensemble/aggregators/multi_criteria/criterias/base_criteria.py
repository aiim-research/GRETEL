from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

from src.core.configurable import Configurable
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.dataset.instances.base import DataInstance
from src.explainer.future.ensemble.aggregators.multi_criteria.criterias.gain_direction import (
    GainDirection,
)

T = TypeVar("T", bound=DataInstance)


class BaseCriteria(Generic[T], Configurable, metaclass=ABCMeta):
    @abstractmethod
    def gain_direction(self) -> GainDirection:
        raise NotImplementedError

    @abstractmethod
    def calculate(
        self,
        first_instance: T,
        second_instance: T,
        oracle: Oracle,
        explainer: Explainer,
        dataset: Dataset,
    ) -> float:
        raise NotImplementedError

    def init(self):
        super().init()
