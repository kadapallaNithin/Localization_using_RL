from .base import Agent, QAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

model_cache = {}

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent(QAgent):
    def param_names(self):
        return ['state_dim', 'action_dim', 'lr', 'gamma',
                'target_update_freq', 'buffer_capacity',
                'min_replay_size', 'update_every',
                'eps_start', 'eps_min', 'tau']

    def __init__(
        self,
        state_dim,
        action_dim,
        # lr=3e-4,
        lr=0.001,
        gamma=0.99,
        target_update_freq=100,
        buffer_capacity=50_000,
        batch_size=64,
        min_replay_size=10_000,
        update_every=4,
        load_if_exists=False,
        eps_start=1.0,
        eps_min=0.05,
        tau=4e-7,
        path="dqn_model.pth"
    ):
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.min_replay_size = min_replay_size
        self.batch_size = batch_size
        self.step_count = 0
        self.update_every = update_every
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.tau = tau
        self.episode = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_freq = target_update_freq
        self.steps = 0

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        print('Created DQN agent', path)
        if not path.endswith('.pth'):
            path += '.pth'
        self.path = path
        if load_if_exists:
            self.load(load_if_exists=load_if_exists)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def act_batch(self, states):
        states = np.asarray(states, dtype=np.float32)
        actions = np.random.randint(self.action_dim, size=len(states))
        greedy_mask = np.random.rand(len(states)) >= self.epsilon
        if np.any(greedy_mask):
            state_tensor = torch.tensor(states[greedy_mask], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            actions[greedy_mask] = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions.tolist()

    def update_batch(self, transitions):
        for transition in transitions:
            self.replay_buffer.add(transition)
            self.step_count += 1
            if len(self.replay_buffer) > self.min_replay_size and self.step_count % self.update_every == 0:
                self.learn()


    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))
        self.step_count += 1
        if len(self.replay_buffer) > self.min_replay_size and self.step_count % self.update_every == 0:
            self.learn()



    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of arrays to numpy arrays first for efficiency
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN
        next_actions = self.q_network(next_states).argmax(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        # print('loss', loss)
        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self):
        torch.save(self.q_network.state_dict(), self.path)

    def load(self, load_if_exists=False):
        print("Load")
        if load_if_exists:
            if not os.path.exists(self.path):
                return
        if self.path in model_cache:
            print("Model loaded from cache.")
            self.q_network, self.target_network = model_cache[self.path]
        else:
            print(f"Loading file {self.path}")
            self.q_network.load_state_dict(torch.load(self.path))
            self.target_network.load_state_dict(self.q_network.state_dict())
            model_cache[self.path] = (self.q_network, self.target_network)

class DQNWrapperAgent(Agent):
    def __init__(self, *args, **kwargs):
        # print(args, kwargs)
        self.action_dim = kwargs['action_dim']
        kwargs['action_dim'] += 1
        self.dqn = DQNAgent(**kwargs)

    def act(self, state):
        action = self.dqn.act(state)
        if action == self.action_dim:
            random_action = np.random.randint(0, self.action_dim)
            # print('random_action', random_action)
            return random_action
        return action

    def update(self, *args):
        return self.dqn.update(*args)

    def act_batch(self, states):
        actions = np.array(self.dqn.act_batch(states))
        random_mask = actions == self.action_dim
        if np.any(random_mask):
            actions[random_mask] = np.random.randint(0, self.action_dim, size=np.sum(random_mask))
        return actions.tolist()

    def update_batch(self, transitions):
        return self.dqn.update_batch(transitions)

    def end_episode(self):
        return self.dqn.end_episode()

    def print_intvl(self, *args, **kwargs):
        return self.dqn.print_intvl(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.dqn.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self.dqn.load(*args, **kwargs)
    
    def maybe_learn(self, *args, **kwargs):
        return self.dqn.maybe_learn(*args, **kwargs)
