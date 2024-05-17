from abc import ABCMeta, abstractmethod
from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle

from src.dataset.instances.base import DataInstance


class Explanation(metaclass=ABCMeta):
    """The common logic shared between all explanation types should be in this class"""
    
    def __init__(self, explainer_class: str) -> None:
        self._explainer_class = explainer_class

    @property
    def explainer_class(self) -> str:
        return self._explainer_class

    @explainer_class.setter
    def explainer_class(self, new_explainer_class) -> None:
        self._explainer_class = new_explainer_class


    