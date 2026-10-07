import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """The brain: takes the 34-number observation, returns one score per basket."""

    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayMemory:
    """A box of past days: (what I saw, what I picked, reward, what I saw next, game over?)."""

    def __init__(self, capacity=10_000):
        self.memories = deque(maxlen=capacity)  # oldest memories fall out when full

    def add(self, obs, action, reward, next_obs, done):
        self.memories.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.memories, batch_size)  # random days, not in order
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))
        return (
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.int64),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(next_obs, dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memories)


class DQNAgent:
    """The player: picks baskets, remembers days and learns from them."""

    def __init__(
        self,
        n_inputs,
        n_actions,
        gamma=0.95,  # how much tomorrow's score matters compared to today's
        lr=0.001,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.95,  # applied once per EPISODE (the old notebook did it per step)
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_net = QNetwork(n_inputs, n_actions)  # learns every step
        self.target_net = QNetwork(n_inputs, n_actions)  # a frozen copy, used to make steady targets
        self.update_target()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayMemory()

    def act(self, obs, greedy=False):
        """Pick a basket. greedy=True never explores (use it for the backtest)."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # explore: try a random basket
        with torch.no_grad():
            scores = self.q_net(torch.as_tensor(obs, dtype=torch.float32))
        return int(scores.argmax())  # exploit: pick the highest score

    def remember(self, obs, action, reward, next_obs, done):
        self.memory.add(obs, action, reward, next_obs, done)

    def learn(self):
        """Train on one random handful of memories. Returns the loss, or None if memory is too small."""
        if len(self.memory) < self.batch_size:
            return None
        obs, actions, rewards, next_obs, dones = self.memory.sample(self.batch_size)

        # What the brain currently guesses for the baskets it actually picked.
        guess = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # What it should have guessed: today's reward + gamma × best score tomorrow (0 if game over).
        with torch.no_grad():
            best_tomorrow = self.target_net(next_obs).max(1).values
            target = rewards + self.gamma * best_tomorrow * (1 - dones)

        loss = nn.functional.mse_loss(guess, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        """Copy the learning brain into the frozen copy."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """Explore a bit less. Call once at the end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.update_target()
