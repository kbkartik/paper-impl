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
        self.policy = PolicyNet(obs_dim, n_actions)

        self.q_net_params = itertools.chain(self.curr_Q1.parameters(), self.curr_Q2.network.parameters())

class MLPNet(nn.Module):

    def __init__(self, obs_dim, n_actions):
        super(MLPNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.ReLU()
        ).to(device)
    
    def forward(self, x):

        x = x.to(device)
        x = self.network(x)
        return x

class PolicyNet(nn.Module):

    def __init__(self, obs_dim, n_actions):
        super(MLPNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        ).to(device)

    def get_logpi_and_one_hot(self, logits, soft_categorical, dim):
        """
        Get the sampled action through straight through estimator trick for the 
        sampled gumbel softmax. Also, return the log pi for entropy term in 
        Qnet and policy net losses.
        """

        index = soft_categorical.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        one_hot_action_vec = y_hard - soft_categorical.detach() + soft_categorical
        logpi_action = torch.log(torch.multiply(soft_categorical.detach(), one_hot_action_vec.detach()).sum(dim))
        
        return logpi_action, one_hot_action_vec
    
    def forward(self, x):
        
        x = x.to(device)
        x = self.network(x)
        soft_categorical = F.gumbel_softmax(x, dim=1)

        dim = 1 if soft_categorical.shape[0] > 1 else -1
        logpi_action, one_hot_action_vec = self.get_logpi_and_one_hot(x, soft_categorical, dim)

        return logpi_action, one_hot_action_vec