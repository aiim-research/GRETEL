from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

from src.core.configurable import Configurable
from src.dataset.instances.base import DataInstance
from src.explainer.ensemble.aggregators.criterias.gain_direction import GainDirection

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
    ) -> float:
        raise NotImplementedError
