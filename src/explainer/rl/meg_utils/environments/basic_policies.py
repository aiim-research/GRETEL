import copy
from typing import Any, Callable, List, Optional, Set

from src.dataset.instances.graph import GraphInstance
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.explainer.rl.meg_utils.environments.base_env import BaseEnvironment, Result


class AddRemoveEdgesEnvironment(BaseEnvironment[GraphInstance]):
    def __init__(
        self,
        target_fn: Optional[Callable[[GraphInstance], Any]] = None,
        max_steps: int = 10,
        record_path: bool = False,
    ):
        super().__init__(target_fn=target_fn, max_steps=max_steps)
        self._valid_actions: Set[GraphInstance] = {}
        self.record_path = record_path
        self._path: List[GraphInstance] = []
        self.reward_fn = GraphEditDistanceMetric().evaluate

    def get_path(self) -> List[GraphInstance]:
        return self._path

    def initialize(self) -> None:
        self._state = self._init_instance
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0

    def get_valid_actions(
        self,
        state: Optional[GraphInstance] = None,
        force_rebuild: bool = False,
    ) -> Set[GraphInstance]:
        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state
        self._valid_actions = self._get_valid_actions(state)
        return copy.deepcopy(self._valid_actions)

    def _get_valid_actions(
        self,
        state: Optional[GraphInstance],
    ) -> Set[GraphInstance]:
        """
        Params:
        State: A data instance
        Returns: A set of data instances
        """
        # adj_matrix = state.to_numpy_array()
        nodes = list(range(state.data.shape[0]))
        valid_actions = []
        # Iterate through each node
        for node in nodes:
            # Iterate through neighbouring nodes and check for valid actions
            for neighbour in nodes:
                if neighbour > node:
                    # Adding/removal of edges
                    state.sync_features_and_weights()
                    temp_inst = copy.deepcopy(state)
                    temp_inst.id = state.id + neighbour + 1
                    temp_inst.data[node][neighbour] = 1 - temp_inst.data[node][neighbour]
                    temp_inst.data[neighbour][node] = temp_inst.data[node][neighbour]
                    temp_inst.sync_features_and_weights()
                    valid_actions.append(temp_inst)
                    # temp_inst = DataInstance(self.state._id + neighbour + 1)
                    # adj_matrix[node][neighbour] = 1 - adj_matrix[node][neighbour]
                    # adj_matrix[neighbour][node] = adj_matrix[node][neighbour]
                    # temp_inst.from_numpy_array(adj_matrix)
                    # valid_actions.append(temp_inst)
        state.sync_features_and_weights()
        return set(valid_actions)

    def set_instance(self, new_instance: Optional[GraphInstance]) -> None:
        self._init_instance = new_instance

    def reward(self):
        return {"reward": self.reward_fn(self._state, self._init_instance)}

    def step(self, action: GraphInstance) -> Result:
        if self.num_steps_taken >= self.max_steps or self.goal_reached():
            raise ValueError("This episode is terminated.")
        if action.id not in [inst.id for inst in self._valid_actions]:
            raise ValueError("Invalid action.")
        self._state = action
        if self.record_path:
            self._path.append(self._state)
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1
        result = Result(
            state=self._state,
            reward=self.reward(),
            terminated=(self._counter >= self.max_steps) or self.goal_reached(),
        )
        return result
