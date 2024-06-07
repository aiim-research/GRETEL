from random import sample as random_sample
from typing import cast

import numpy as np
import torch

from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.rl.meg_utils.environments.base_env import BaseEnvironment
from src.explainer.rl.meg_utils.utils.encoders import ActionEncoderAB
from src.explainer.rl.meg_utils.utils.queue import SortedQueue
from src.explainer.rl.meg_utils.utils.sorters import SorterSelector
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.context import Context


class MEGExplainer(Explainer):
    def check_configuration(self):
        super().check_configuration()
        dst_metric = "src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric"
        environment = "src.explainer.rl.meg_utils.environments.basic_policies.AddRemoveEdgesEnvironment"
        action_encoder = "src.explainer.rl.meg_utils.utils.encoders.IDActionEncoder"
        sorter = "src.explainer.rl.meg_utils.utils.sorters.RewardSorterSelector"
        init_dflts_to_of(self.local_config, "distance_metric", dst_metric)
        init_dflts_to_of(self.local_config, "env", environment)
        init_dflts_to_of(self.local_config, "action_encoder", action_encoder)
        init_dflts_to_of(self.local_config, "sorter_selector", sorter)
        params = self.local_config["parameters"]
        params["num_input"] = params.get("num_input", 5)
        params["batch_size"] = params.get("batch_size", 1)
        params["lr"] = params.get("lr", 1e-4)
        params["replay_buffer_size"] = params.get("replay_buffer_size", 10000)
        params["epochs"] = params.get("epochs", 10)
        params["max_steps_per_episode"] = params.get("max_steps_per_episode", 1)
        params["update_interval"] = params.get("update_interval", 1)
        params["gamma"] = params.get("gamma", 0.95)
        params["polyak"] = params.get("polyak", 0.995)
        params["num_counterfactuals"] = params.get("num_counterfactuals", 10)
        self.context.logger.info("Configuration checked and parameters set.")

    def init(self):
        super().init()
        params = self.local_config["parameters"]
        self.distance_metric = cast(
            EvaluationMetric,
            get_instance_kvargs(
                params["distance_metric"]["class"],
                params["distance_metric"]["parameters"],
            ),
        )
        self.context.logger.info("Distance metric initialized.")
        params["env"]["parameters"]["context"] = self.context
        self.environment = cast(
            BaseEnvironment,
            get_instance_kvargs(params["env"]["class"], params["env"]["parameters"]),
        )
        self.context.logger.info("Environment initialized.")
        self.action_encoder = cast(
            ActionEncoderAB,
            get_instance_kvargs(
                params["action_encoder"]["class"],
                params["action_encoder"]["parameters"],
            ),
        )
        self.context.logger.info("Action encoder initialized.")
        self.sorter_selector = cast(
            SorterSelector,
            get_instance_kvargs(
                params["sorter_selector"]["class"],
                params["sorter_selector"]["parameters"],
            ),
        )
        self.context.logger.info("Sorter selector initialized.")
        self.num_input = cast(int, params["num_input"])
        self.batch_size = cast(int, params["batch_size"])
        self.lr = cast(float, params["lr"])
        self.replay_buffer_size = cast(int, params["replay_buffer_size"])
        self.epochs = cast(int, params["epochs"])
        self.max_steps_per_episode = cast(int, params["max_steps_per_episode"])
        self.update_interval = cast(int, params["update_interval"])
        self.gamma = cast(float, params["gamma"])
        self.polyak = cast(float, params["polyak"])
        self.num_counterfactuals = cast(int, params["num_counterfactuals"])
        self.context.logger.info("Initialization complete with all parameters set.")

    def explain(self, instance):
        num_nodes = instance.data.shape[0]
        num_edges = np.count_nonzero(instance.data)
        assert len(instance.node_features) == num_nodes
        assert len(instance.edge_features) == num_edges
        assert len(instance.edge_weights) == num_edges
        self.instance = instance
        self.context.logger.info(f"Explaining instance with {num_nodes} nodes and {num_edges} edges.")
        # dataset = self.converter.convert(dataset)
        self.explainer = MEGAgent(
            self.context,
            num_input=self.num_input + 1,
            num_output=1,
            lr=self.lr,
            replay_buffer_size=self.replay_buffer_size,
        )

        self.__fit()
        instance = self.dataset.get_instance(instance.id)

        with torch.no_grad():
            inst = self.cf_queue.get(0)  # get the best counterfactual
            self.context.logger.info("Best counterfactual obtained.")
            return inst["next_state"]

    def __fit(self):
        # explainer_name = f"meg_fit_on_{self.dataset.name}_instance={self.instance.id}_fold_id={self.fold_id}"
        # self.name = explainer_name

        self.cf_queue = SortedQueue(
            self.num_counterfactuals,
            sort_predicate=self.sorter_selector.predicate(),
        )
        self.context.logger.info("Counterfactual queue initialized.")
        self.environment.set_instance(self.instance)
        self.environment.oracle = self.oracle

        self.environment.initialize()
        self.context.logger.info("Environment initialized for fitting.")
        self.__real_fit()

    def __real_fit(self):
        eps = 1.0
        batch_losses = []
        episode = 0
        it = 0
        while episode < self.epochs:
            steps_left = self.max_steps_per_episode - self.environment.num_steps_taken
            valid_actions = list(self.environment.get_valid_actions())

            self.context.logger.info(f"Amount of valid actions: {len(valid_actions)}")
            self.context.logger.info(f"Steps left: {steps_left}")

            self.context.logger.info("Observations being prepared.")

            observations = np.vstack(
                [
                    np.append(self.action_encoder.encode(action), steps_left)
                    for action in valid_actions
                ]
            )

            self.context.logger.info("Observations prepared.")

            observations = torch.as_tensor(observations).float()
            a = self.explainer.action_step(observations, eps)
            action = valid_actions[a]
            self.context.logger.info(f"Action taken: {action}")

            result = self.environment.step(action)

            action_embedding = np.append(self.action_encoder.encode(action), steps_left)

            next_state, out, done = result

            steps_left = self.max_steps_per_episode - self.environment.num_steps_taken

            action_embeddings = np.vstack(
                [
                    np.append(self.action_encoder.encode(action), steps_left)
                    for action in self.environment.get_valid_actions()
                ]
            )

            self.explainer.replay_buffer.push(
                torch.as_tensor(action_embedding).float(),
                torch.as_tensor(out["reward"]).float(),
                torch.as_tensor(action_embeddings).float(),
                float(result.terminated),
            )
            self.context.logger.info("Experience pushed to replay buffer.")

            if (
                it % self.update_interval == 0
                and len(self.explainer.replay_buffer) >= self.batch_size
            ):
                loss = self.explainer.train_step(
                    self.batch_size, self.gamma, self.polyak
                )
                loss = loss.item()
                batch_losses.append(loss)
                self.context.logger.info(f"Training step completed. Loss: {loss}")

            it += 1

            if done:
                episode += 1

                self.context.logger.info(f'Episode {episode}> Reward = {out["reward"]}')
                self.cf_queue.insert(
                    {
                        "marker": "cf",
                        "id": lambda action: action,
                        "next_state": next_state,
                        **out,
                    }
                )

                eps *= 0.9987

                batch_losses = []
                self.environment.initialize()
                self.context.logger.info(f"Episode {episode} finished. Environment reinitialized.")


class MEGAgent:
    def __init__(
        self,
        context: Context,
        num_input: int = 5,
        num_output: int = 10,
        lr: float = 1e-3,
        replay_buffer_size: int = 10,
    ):
        self.context = context
        self.num_input = num_input
        self.num_output = num_output
        self.replay_buffer_size = replay_buffer_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dqn, self.target_dqn = (
            DQN(self.context, num_input, num_output).to(self.device),
            DQN(self.context, num_input, num_output).to(self.device),
        )

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayMemory(self.context, replay_buffer_size)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        self.context.logger.info("MEGAgent initialized with DQN and target DQN.")

    def action_step(self, observations, epsilon_threshold):
        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
            self.context.logger.info("Random action taken due to epsilon threshold.")
        else:
            q_value = self.dqn(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).detach().numpy()
            self.context.logger.info("Action taken based on Q-values.")

        return action

    def train_step(self, batch_size: int, gamma: float, polyak: float) -> torch.Tensor:
        experience = self.replay_buffer.sample(batch_size)
        states_ = torch.stack([S for S, *_ in experience])
        next_states_ = [S for *_, S, _ in experience]

        q = self.dqn(states_)
        q_target = torch.stack(
            [self.target_dqn(S).max(dim=0).values.detach() for S in next_states_]
        )

        rewards = (
            torch.stack([R for _, R, *_ in experience])
            .reshape((1, batch_size))
            .to(self.device)
        )
        dones = (
            torch.tensor([D for *_, D in experience])
            .reshape((1, batch_size))
            .to(self.device)
        )

        q_target = rewards + gamma * (1 - dones) * q_target
        td_target = q - q_target

        loss = torch.where(
            torch.abs(td_target) < 1.0,
            0.5 * td_target * td_target,
            1.0 * (torch.abs(td_target) - 0.5),
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(
                self.dqn.parameters(), self.target_dqn.parameters()
            ):
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)

        self.context.logger.info(f"Training step: loss = {loss.item()}")
        return loss


class ReplayMemory:
    def __init__(self, context: Context, capacity: int):
        self.context = context
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.context.logger.info(f"Replay memory initialized with capacity {capacity}.")

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity
        self.context.logger.info(f"Memory pushed at position {self.position}.")

    def sample(self, batch_size: int):
        self.context.logger.info(f"Sampling {batch_size} experiences from memory.")
        return random_sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):
    def __init__(
        self,
        context: Context,
        num_input: int,
        num_output: int,
        hidden_state_neurons: list[int] = [1024, 512, 128, 32],
    ):
        super(DQN, self).__init__()

        self.context = context

        self.layers = torch.nn.ModuleList([])

        hs = hidden_state_neurons

        N = len(hs)

        for i in range(N - 1):
            h, h_next = hs[i], hs[i + 1]
            dim_input = num_input if i == 0 else h

            self.layers.append(torch.nn.Linear(dim_input, h_next))

        self.out = torch.nn.Linear(hs[-1], num_output)
        self.context.logger.info(f"DQN initialized with input {num_input}, output {num_output}, and layers {hidden_state_neurons}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        x = self.out(x)
        self.context.logger.info("Forward pass through DQN.")
        return x
