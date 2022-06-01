import gym
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import yaml

from discrete_sac import DiscreteSAC

# Define seeding
seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Defining env
env = gym.make('CartPole-v1')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a class for replay buffer
class ExperienceRelay:

    def __init__(self, replay_buffer_cap, minibatch_size):
        
        self.Transition = namedtuple('Transition', 'current_phi action reward next_phi done') # Define transitions
        self.minibatch_size = minibatch_size
        self.buffer = deque([], maxlen=replay_buffer_cap)

    def store(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample_minibatch(self):
        return random.sample(self.buffer, self.minibatch_size)

class Agent:

    def __init__(self, HYPERPARAMS):
        self.HYPERPARAMS = HYPERPARAMS
        self.model = DiscreteSAC(self.HYPERPARAMS)

    def select_action(self, curr_st):

        if self.HYPERPARAMS['action_space'] == 'discrete':
            
            with torch.no_grad():
                action_probs = self.model.current_policy(curr_st)
        
        else:



    def update_current_nets(self, ):

    def update_target_nets(self, ):
    
    def train(self):

        for ep in range(self.HYPERPARAMS['n_episodes']):

            done = False
            curr_st = env.reset()
            action = self.select_action(curr_st)
            env_steps = 0
            
            while not done and env_steps < self.HYPERPARAMS['max_env_steps']:

                next_st, reward, done, _ = env.step(action)
                self.replay_buffer.store(curr_st, action, reward, next_st, done)

                if not done:
                    curr_st = next_st
                    action = self.select_action(curr_st)

                 # determine some condition for update
                if done:
                    for j in range(self.HYPERPARAMS['n_model_updates']):
                        minibatch = self.replay_buffer.sample_minibatch()
                        self.update_current_nets(minibatch)
                        self.update_target_nets()

        env.close()


if __name__ == "__main__":
    with open('./discrete_sac.yml') as f:
        hyper_params = yaml.load(f)
        agent = Agent(hyper_params)
        agent.train()
        agent.eval()