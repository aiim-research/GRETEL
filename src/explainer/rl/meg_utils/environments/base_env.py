import collections
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, Set, TypeVar

from src.dataset.instances.base import DataInstance


class Result(collections.namedtuple("Result", ["state", "reward", "terminated"])):
    """
    A namedtuple defines the result of a step taken.
    The namedtuple contains the following fields:
        state: The instance reached after taking the action.
        reward: Float. The reward get after taking the action.
        terminated: Boolean. Whether this episode is terminated.
    """


T = TypeVar("T", bound=DataInstance)


class BaseEnvironment(Generic[T], ABC):
    def __init__(
        self,
        target_fn: Optional[Callable[[T], Any]] = None,
        max_steps: int = 10,
    ):
        self._name = "base_environment"

        self._state: Optional[T] = None  # It is a data instace
        self._init_instance: Optional[T] = None  # It is a data instace
        self._counter = 0
        self.max_steps = max_steps
        self._target_fn = target_fn

    @property
    def init_instance(self) -> Optional[T]:
        return self._init_instance

    @abstractmethod
    def set_instance(self, new_instance: Optional[T]) -> None:
        pass

    @property
    def state(self) -> Optional[T]:
        return self._state

    @state.setter
    def state(self, new_state: Optional[T]):
        self._state = new_state

    @property
    def num_steps_taken(self):
        return self._counter

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def reward(self) -> Any:
        pass

    @abstractmethod
    def step(self, action) -> Result:
        pass

    @abstractmethod
    def get_valid_actions(
        self,
        state: Optional[T] = None,
        force_rebuild: bool = False,
    ) -> Set[T]:
        pass

    def goal_reached(self):
        if not self._target_fn:
            return False
        return self._target_fn(self._state)
