from abc import ABCMeta, abstractmethod

from src.explanation.base import Explanation
from src.core.configurable import Configurable


class Stage(Configurable, metaclass=ABCMeta):

    @abstractmethod
    def process(self, explanation: Explanation) -> Explanation:
        pass


    def write_into_explanation(self, exp: Explanation, value):
        exp._stages_info[self.__class__.name] = value