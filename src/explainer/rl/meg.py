import os

import numpy as np
import torch
import random

from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.core.trainable_base import Trainable

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset

from src.explainer.rl.meg_utils.utils.queue import SortedQueue


class MEGExplainer(Explainer, Trainable):
    
    def __init__(self,
                id,
                environment,
                action_encoder,
                num_input=5,
                lr=1e-3,
                replay_buffer_size=10,
                num_epochs=500,
                max_steps_per_episode=5,
                update_interval=10,
                batch_size=25,
                num_counterfactuals=5,
                gamma=5,
                polyak=5,
                fold_id=0,
                sort_predicate=None,
                config_dict=None) -> None:
        
        super().__init__(id, config_dict)
        
        self.num_epochs = num_epochs
        self.max_steps_per_episode = max_steps_per_episode
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.num_counterfactuals = num_counterfactuals
        self.gamma = gamma
        self.polyak = polyak
        self.action_encoder = action_encoder
        self.sort_predicate = sort_predicate
        self.fold_id = fold_id
        self.num_input = num_input
        self.replay_buffer_size = replay_buffer_size
        self.environment = environment
        self.lr = lr
        
        

    def explain(self, instance):
        #dataset = self.converter.convert(dataset)              
        self.explainer = MEGAgent(num_input=self.num_input + 1,
                                  num_output=1,
                                  lr=self.lr,
                                  replay_buffer_size=self.replay_buffer_size)
        
        self.fit(self.oracle, self.dataset, instance, self.fold_id)
        instance = self.dataset.get_instance(instance.id)
        
        with torch.no_grad():
            inst = self.cf_queue.get(0) # get the best counterfactual
            return inst['next_state']

    def save_explainers(self):
        self.explainer.save(os.path.join(self.explainer_store_path, self.name))
 
    def load_explainers(self):
        self.explainer.load(os.path.join(self.explainer_store_path, self.name))

    
    def real_fit(self):
        pass


    def fit(self, oracle: Oracle, dataset : Dataset, instance: DataInstance, fold_id=0):
        explainer_name = f'meg_fit_on_{dataset.name}_instance={instance.id}_fold_id={fold_id}'
        self.name = explainer_name
                
        self.cf_queue = SortedQueue(self.num_counterfactuals, sort_predicate=self.sort_predicate)
        self.environment.set_instance(instance)
        self.environment.oracle = oracle
        
        self.environment.initialize()                    
        self.__fit(oracle, dataset, instance, fold_id)

    def __fit(self, oracle, dataset, instance, fold_id):
        eps = 1.0
        batch_losses = []
        episode = 0
        it = 0
        while episode < self.num_epochs:
            steps_left = self.max_steps_per_episode - self.environment.num_steps_taken
            valid_actions = list(self.environment.get_valid_actions())
                        
            observations = np.vstack(
                [
                    np.append(self.action_encoder.encode(action), steps_left) for action in valid_actions
                ]
            )
            
            observations = torch.as_tensor(observations).float()
            a = self.explainer.action_step(observations, eps)
            action = valid_actions[a]
            
            result = self.environment.step(action)
                        
            action_embedding = np.append(
                self.action_encoder.encode(action),
                steps_left
            )
            
            next_state, out, done = result
            
            steps_left = self.max_steps_per_episode - self.environment.num_steps_taken
            
            action_embeddings = np.vstack(
                [
                    np.append(self.action_encoder.encode(action), steps_left) for action in self.environment.get_valid_actions()
                ]
            )
            
            self.explainer.replay_buffer.push(
                torch.as_tensor(action_embedding).float(),
                torch.as_tensor(out['reward']).float(),
                torch.as_tensor(action_embeddings).float(),
                float(result.terminated)
            )
            
            if it % self.update_interval == 0 and len(self.explainer.replay_buffer) >= self.batch_size:
                loss = self.explainer.train_step(
                    self.batch_size,
                    self.gamma,
                    self.polyak
                )
                loss = loss.item()
                batch_losses.append(loss)
            
            it += 1
            
            if done:
                episode += 1
                
                #print(f'Episode {episode}> Reward = {out["reward"]} (pred: {out["reward_pred"]}, sim: {out["reward_sim"]})')
                print(f'Episode {episode}> Reward = {out["reward"]}')
                self.cf_queue.insert({
                    'marker': 'cf',
                    'id': lambda action : action,
                    'next_state': next_state,
                    **out
                })
                                
                eps *= 0.9987
                
                batch_losses = []
                self.environment.initialize()
        
class MEGAgent:
    
    def __init__(self,
                num_input=5,
                num_output=10,
                lr=1e-3,
                replay_buffer_size=10):
        
        self.num_input = num_input
        self.num_output = num_output
        self.replay_buffer_size = replay_buffer_size
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
        self.dqn, self.target_dqn = (
           DQN(num_input, num_output).to(self.device),
           DQN(num_input, num_output).to(self.device)
        )
        
        for p in self.target_dqn.parameters():
            p.requires_grad = False
            
        self.replay_buffer = ReplayMemory(replay_buffer_size)
        
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        
    def action_step(self, observations, epsilon_threshold):
        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).detach().numpy()
            
        return action
    
    def train_step(self, batch_size, gamma, polyak):
        experience = self.replay_buffer.sample(batch_size)
        states_ = torch.stack([S for S, *_ in experience])        
        next_states_ = [S for *_, S, _ in experience]
        
        q = self.dqn(states_)
        q_target = torch.stack([self.target_dqn(S).max(dim=0).values.detach() for S in next_states_])
        
        rewards = torch.stack([R for _, R, *_ in experience]).reshape((1, batch_size)).to(self.device)
        dones = torch.tensor([D for *_, D in experience]).reshape((1, batch_size)).to(self.device)
        
        q_target = rewards + gamma * (1-dones) * q_target
        td_target = q - q_target
        
        loss = torch.where(
            torch.abs(td_target) < 1.0,
            0.5 * td_target * td_target,
            1.0 * (torch.abs(td_target) - 0.5)
        ).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        with torch.no_grad():
            for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)
                
        return loss
    
class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
class DQN(torch.nn.Module):
    
    def __init__(self, num_input, num_output,
                 hidden_state_neurons=[1024, 512, 128, 32]):
        
        super(DQN, self).__init__()
        
        self.layers = torch.nn.ModuleList([])
        
        hs = hidden_state_neurons
        
        N = len(hs)
        
        for i in range(N-1):
            h, h_next = hs[i], hs[i+1]
            dim_input = num_input if i == 0 else h
                        
            self.layers.append(
                torch.nn.Linear(dim_input, h_next)
            )
            
        self.out = torch.nn.Linear(hs[-1], num_output)
        
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        x = self.out(x)
        return x