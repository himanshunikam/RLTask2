from asteroid import AsteroidStatic
from action_space import ACTIONS, NUM_ACTIONS, get_action
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, n_in=4, n_out=9, hidden=256):
        super().__init__()
        self.linear1 = nn.Linear(n_in, hidden)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden, n_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return self.linear3(x)   # linear output: Q-values must be free to go negative




if __name__ =='__main__':
    print()