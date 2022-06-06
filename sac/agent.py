import gym
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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

writer = SummaryWriter('runs/experiment1')

# Define a class for replay buffer
class ExperienceRelay:

    def __init__(self, replay_buffer_cap, minibatch_size):
        
        self.Transition = namedtuple('Transition', 'curr_st action reward next_st') # Define transitions
        self.minibatch_size = minibatch_size
        self.buffer = deque([], maxlen=replay_buffer_cap)

    def store(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample_minibatch(self):
        return random.sample(self.buffer, self.minibatch_size)
    
    def get_length(self):
        return len(self.buffer)

class Agent:

    def __init__(self, HYPERPARAMS):
        self.HYPERPARAMS = HYPERPARAMS

        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.model = DiscreteSAC(n_states, n_actions)        

        self.replay_buffer = ExperienceRelay(self.HYPERPARAMS['replay_buffer_cap'], self.HYPERPARAMS['minibatch_size'])

        self.q_net_optimizer = optim.Adam(self.model.q_net_params, lr=0.0001)
        self.pi_optimizer = optim.Adam(self.model.policy.parameters(), lr=0.0001)
        self.loss_fn = nn.SmoothL1Loss()

        self.alpha = 0.3

    def select_action(self, curr_st):

        if self.HYPERPARAMS['action_space'] == 'discrete':
            
            with torch.no_grad():
                action_probs = self.model.policy(curr_st)

            if np.random.random() > self.HYPERPARAMS['epsilon']:
                action = action_probs.max().item()
            else:
                action = torch.random(action_probs).item()

    def optimize_model(self, minibatch, j):

        # reasoning behind TD3 DDQN clipping and overestimation bias?
        # reasoning behind number of policies
        # WHy replay buffer can't be used in PG algos.

        # creating minibatch
        transition_minibatch = self.replay_buffer.Transition(*zip(*minibatch))
        curr_st_mb = torch.stack(transition_minibatch.curr_st).to(device)
        action_mb = torch.stack(transition_minibatch.action)
        reward_mb = torch.stack(transition_minibatch.reward)

        # Getting q values for all actions in current states
        curr_st_mb_q1vals = self.model.curr_Q1(curr_st_mb)
        curr_st_mb_q2vals = self.model.curr_Q2(curr_st_mb)

        # Getting q value for current state, action
        curr_st_action_q1vals = curr_st_mb_q1vals.gather(1, action_mb)
        curr_st_action_q2vals = curr_st_mb_q2vals.gather(1, action_mb)

        non_terminal_idxs = []
        non_terminal_next_st_mb = []
        for i, ns in enumerate(transition_minibatch.next_st):
            if ns is not None:
                non_terminal_idxs.append(i)
                non_terminal_next_st_mb.append(ns)
        
        non_terminal_idxs = torch.tensor(non_terminal_idxs, dtype=torch.bool)
        non_terminal_next_st_mb = torch.stack(non_terminal_next_st_mb).to(device)

        with torch.no_grad():
            # sample next state max action using current q nets
            next_st_action_mb = self.model.policy(non_terminal_next_st_mb).detach()

            # evaluate sampled max action on target q nets
            next_st_action_tgt_eval_q1net = torch.zeros(self.replaybuffer.minibatch_size, device=device)
            next_st_action_tgt_eval_q2net = torch.zeros(self.replaybuffer.minibatch_size, device=device)

            next_st_action_tgt_eval_q1net[non_terminal_idxs] = self.model.target_Q1(non_terminal_next_st_mb).gather(1, next_st_action_mb).detach()
            next_st_action_tgt_eval_q2net[non_terminal_idxs] = self.model.target_Q2(non_terminal_next_st_mb).gather(1, next_st_action_mb).detach()

            # Target Q values
            backup = reward_mb + self.HYPERPARAMS['gamma'] * (torch.minimum(next_st_eval_max_action_q1net, next_st_eval_max_action_q2net) - self.alpha * log_pi_action)

        # Optimizing Q nets
        q_net_loss = self.loss_fn(y, curr_st_action_q1vals) + self.loss_fn(backup, curr_st_action_q2vals)
        self.q_net_optimizer.zero_grad()
        q_net_loss.backward()
        self.q_net_optimizer.step()

        if j % self.HYPERPARAMS['pi_tgt_nets_update_freq']:
            # Optimizing policy
            with torch.no_grad():
                curr_st_mb_q_vals = torch.minimum(curr_st_mb_q1vals.clone().detach(), curr_st_mb_q2vals.clone().detach())

            pi_loss = self.alpha * torch.log(self.model.policy(curr_st_mb)) - curr_st_mb_q_vals
            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            self.pi_optimizer.step()

    def update_target_nets(self):

        with torch.no_grad():
            updated_params = self.HYPERPARAMS['polyak'] * self.model.target_Q1.state_dict() + (1-self.HYPERPARAMS['polyak']) * self.model.curr_Q1.state_dict()
            self.model.target_Q1.load_state_dict(updated_params)

            updated_params = self.HYPERPARAMS['polyak'] * self.model.target_Q2.state_dict() + (1-self.HYPERPARAMS['polyak']) * self.model.curr_Q2.state_dict()
            self.model.target_Q2.load_state_dict(updated_params)
    
    def train(self):
        agent_lifetime_steps = 0

        for ep in range(self.HYPERPARAMS['n_episodes']):

            done = False
            curr_st = env.reset()
            action = self.select_action(curr_st)
            env_steps = 0
            episodic_return = 0
            
            while not done: # and env_steps < self.HYPERPARAMS['max_env_steps']:

                next_st, reward, done, _ = env.step(action)
                self.replay_buffer.store(curr_st, action, reward, next_st)
                env_steps += 1
                agent_lifetime_steps += 1
                episodic_return += reward

                if not done:
                    curr_st = next_st
                    action = self.select_action(curr_st)
                
                # optimize model
                if self.replay_buffer.get_length() > self.HYPERPARAMS['init_buffer_len'] or agent_lifetime_steps % self.HYPERPARAMS['steps_update_freq'] == 0:
                    for j in range(self.HYPERPARAMS['n_model_updates']):
                        minibatch = self.replay_buffer.sample_minibatch()
                        self.optimize_model(minibatch, j)
                        if j % self.HYPERPARAMS['pi_tgt_nets_update_freq']:
                            self.update_target_nets()
            
            writer.add_scalar('episodic_return', episodic_return, ep+1)

        env.close()


if __name__ == "__main__":
    with open('./discrete_sac.yml') as f:
        hyper_params = yaml.load(f)
        agent = Agent(hyper_params)
        agent.train()