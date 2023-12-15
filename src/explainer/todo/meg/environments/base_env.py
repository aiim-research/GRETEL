import collections
from abc import ABC, abstractmethod
from typing import Set

from src.dataset.data_instance_base import DataInstance

class Result(
    collections.namedtuple('Result', ['state', 'reward', 'terminated'])):
    """
    A namedtuple defines the result of a step taken.
    The namedtuple contains the following fields:
        state: The instance reached after taking the action.
        reward: Float. The reward get after taking the action.
        terminated: Boolean. Whether this episode is terminated.
    """
    
class BaseEnvironment(ABC):
    
    def __init__(self, target_fn=None, max_steps=10):
        self._name = 'base_environment'
        
        self._state: DataInstance = None
        self._init_instance: DataInstance = None
        self._counter = 0
        self.max_steps = max_steps
        self._target_fn = target_fn
        
        
    @property
    def init_instance(self):
        return self._init_instance
    
    @abstractmethod
    def set_instance(self, new_instance):
        pass
        
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, new_state):
        self._state = new_state
    
    @property
    def num_steps_taken(self):
        return self._counter
    
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def reward(self) -> any:
        pass
    
    @abstractmethod
    def step(self, action) -> Result:
        return None
    
    @abstractmethod
    def get_valid_actions(self, state=None, force_rebuild=False) -> Set[DataInstance]:
        return None
    
    def goal_reached(self):
        if not self._target_fn:
            return False
        return self._target_fn(self._state)
    
   