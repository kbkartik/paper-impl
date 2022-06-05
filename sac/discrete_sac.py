import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import random
from copy import deepcopy
import itertools

class DiscreteSAC(nn.Module):

    def __init__(self, obs_dim, n_actions):
        super(DiscreteSAC, self).__init__()

        # Define current Q nets
        self.curr_Q1 = MLPNet(obs_dim, n_actions)
        self.curr_Q2 = MLPNet(obs_dim, n_actions)

        # Define target Q nets
        self.target_Q1 = deepcopy(self.curr_Q1)
        self.target_Q1.eval()

        self.target_Q2 = deepcopy(self.curr_Q2)
        self.target_Q2.network.eval()

        # Define policy
        self.policy = MLPNet(obs_dim, n_actions, output_activation=F.gumbel_softmax)

        self.q_net_params = itertools.chain(self.curr_Q1.parameters(), self.curr_Q2.network.parameters())

class MLPNet(nn.Module):

    def __init__(self, obs_dim, n_actions, output_activation=F.softmax):
        super(MLPNet, self).__init__()

        self.output_activation = output_activation
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        ).to(device)
    
    def forward(self, x):

        x = x.to(device)
        x = self.network(x)
        x = self.output_activation(x, dim=1)
        return x