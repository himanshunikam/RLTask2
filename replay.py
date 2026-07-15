from asteroid import AsteroidStatic
from action_space import ACTIONS, NUM_ACTIONS, get_action
from collections import deque
import random
import numpy as np
import torch
class Replay():
    def __init__(self) -> None:
        self.buffer = deque(maxlen=50000)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch=64):

        self.batch_array = random.sample(self.buffer, batch)
        states, actions, rewards, next_states, dones = zip(*self.batch_array)

        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64).unsqueeze(1),       # action is the discrete index 0..8
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)



