import json
import os
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque


class CallCounter:
    def __init__(self, env):
        self.n = 0
        self._orig = env._calc_next_state
        env._calc_next_state = self._wrap
    def _wrap(self, a):
        self.n += 1
        return self._orig(a)

    
NORM_CENTER = torch.tensor([2.0, 2.0, 0.0, 0.0])
NORM_SCALE  = torch.tensor([2.5, 2.5, 4.0, 4.0])

def normalize(s):
    return (s - NORM_CENTER) / NORM_SCALE


class Actor(nn.Module):
    def __init__(self, s_dim=4, a_dim=2, hidden=256, a_max=1.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_dim)
        )
        self.a_max = a_max

    def forward(self, s):
        return self.a_max * torch.tanh(self.net(normalize(s)))


class Critic(nn.Module):
    def __init__(self, sdim=4, adim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sdim + adim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, s, a):
        return self.net(torch.cat([normalize(s), a], dim=1)).squeeze(1)


class Buffer():
    def __init__(self) -> None:
        self.buffer = deque(maxlen=100000)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch=128):
        self.batch_array = random.sample(self.buffer, batch)
        states, actions, rewards, next_states, dones = zip(*self.batch_array)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(np.array(dones), dtype=torch.float32))