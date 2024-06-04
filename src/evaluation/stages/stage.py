from src.explanation.base import Explanation
from src.core.configurable import Configurable
from abc import abstractmethod, ABC

class Stage(Configurable, ABC):

    @abstractmethod
    def process(self, exp: Explanation) -> Explanation:
        pass