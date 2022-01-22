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

# Define seeding
#np.random.seed(10)

# Defining env
env = gym.make('CartPole-v1')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create sequetial MLP network
class MLP(nn.Module):

    def __init__(self,):
        super(MLP, self).__init__()

    def create_network(self, input_dim, output_dim, net_arch=[32, 32], activation_fn=nn.ReLU(), final_layer_activation=nn.Identity()):
        mlp = []
        num_layers = len(net_arch)
        for i in range(num_layers):
            if i == 0:
                mlp.append(nn.Linear(input_dim, net_arch[i]))
                mlp.append(activation_fn)
            elif i == num_layers - 1:
                mlp.append(nn.Linear(net_arch[i], output_dim))
                mlp.append(final_layer_activation)
            else:
                mlp.append(nn.Linear(net_arch[i], net_arch[i+1]))
                mlp.append(activation_fn)
        return nn.Sequential(*mlp).to(device)

class DQNCNN(nn.Module):

    def __init__(self, input_image, action_space_dim,):
        super(Net, self).__init__()

        # Shape of input image
        C_in, H_in, W_in = input_image.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(C_in, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ).to(device)

        # One forward pass using an input image to calculate n_flatten neurons
        with torch.no_grad():
            n_flatten = self.cnn(input_image).shape[1]

        self.fc1 = nn.Linear(n_flatten, 512)
        self.head = nn.Linear(512, output_dim)

    def forward(self, x):

        x = x.to(device)
        x = self.cnn(x)
        x = F.relu(self.fc1(x))
        x = self.head(x)

# Define a class for replay buffer
class ExperienceRelay:

    def __init__(self, buffer_capacity, minibatch_size):
        
        self.Transition = namedtuple('Transition', 'current_phi action reward next_phi') # Define transitions
        self.minibatch_size = minibatch_size
        self.buffer = deque([], maxlen=buffer_capacity)

    def store(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample_minibatch(self):
        return random.sample(self.buffer, self.minibatch_size)

class DQNAgent(nn.Module):

    def __init__(self, env, net_arch, buffer_capacity, minibatch_size, n_episodes, epsilon, history_length, target_update_freq, gamma):

        super(DQN, self).__init__()

        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.env = env

        # Initial DQN networks
        self.mlp = MLP()
        self.Q_net = self.mlp.create_network(self.n_states, self.n_actions, net_arch=net_arch)
        self.target_Q_net = self.mlp.create_network(self.n_states, self.n_actions, net_arch=net_arch)
        self.target_Q_net.load_state_dict(self.Q_net.state_dict())
        self.target_Q_net.eval()

        self.er = ExperienceRelay(buffer_capacity, minibatch_size)

        self.optimizer = optim.RMSprop(self.Q_net.parameters())
        self.loss_fn = nn.SmoothL1Loss()
        
        self.n_episodes = n_episodes
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_delay = eps_delay
        self.experience_count = 0 # Number of experiences (or transitions) seen until now

        self.history_length = history_length
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        #self.max_grad_norm = max_grad_norm
    
    #def preprocessed_input(self, ):    

    def select_action(self, current_phi):

        eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.experience_count/self.eps_delay)

        self.experience_count += 1

        if random.random() > eps_thresh:
            with torch.no_grad():
                return self.Q_net(current_phi).max(1)[1].view(1, 1)
        else:
            rand_action = self.env.action_space.sample()
            return torch.tensor([rand_action], device=device)

    def optimize_model(self, minibatch):

        transition_minibatch = self.er.Transition(*zip(*minibatch))

        current_phi_mb = torch.stack(transition_minibatch.current_phi)
        action_mb = torch.stack(transition_minibatch.action)
        reward_mb = torch.stack(transition_minibatch.reward)

        # State action values of current_phi using Q-net
        current_phi_estimates = self.Q_net(current_phi_mb).gather(1, action_mb)

        # Boolean indexes where next_phi is not terminal
        non_terminal_idxs = torch.tensor(tuple(map(lambda next_phi: next_phi is not None, 
                            transition_minibatch.next_phi)), device=device, dtype=torch.bool)

        non_terminal_next_phi_mb = torch.stack([ns for ns in transition_minibatch.next_phi if ns is not None])
        next_phi_max_qvals = torch.zeros(self.er.minibatch_size, device=device)
        next_phi_max_qvals[non_terminal_idxs] = self.target_Q_net(non_terminal_next_phi_mb).max(1)[0].detach()

        # Target Q values
        y = reward_mb + self.gamma * next_phi_max_qvals

        self.loss_fn(y, current_phi_estimates)
        self.optmizer.zero_grad()
        self.loss_fn.loss.backward()

        ###
        # PyTorch has clamped gradient whereas SB3 has clipped grad norm
        # We do neither. We test how it works first with Huber (SmoothL1) loss.
        #https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
        ###

        self.optimizer.step()        

    def train(self,):

        for ep in range(self.n_episodes):
            z = np.zeros((self.history_length, self.n_states)).tolist()
            current_hist = deque(z, maxlen=self.history_length)
            current_hist.append(env.reset()) # Initial obs is the new current history when env is reset
            done = False
            
            current_phi = torch.as_tensor(current_hist).T
            #if env.type == 'Atari':
            #    self.current_phi = self.preprocessed_input(current_hist)

            while not done:
                action = self.select_action(current_phi)
                obs, reward, done, _ = env.step(action.item())
                if not done:
                    current_hist.append(list(obs))
                    next_phi = torch.as_tensor(current_hist).T
                else:
                    next_phi = None

                self.er.store(current_phi, action.item(), reward, next_phi)
                current_phi = next_phi
                if len(self.er.buffer) > 50 * self.er.minibatch_size:
                    minibatch = self.er.sample_minibatch()
                    self.optimize_model(minibatch)

                                            
        if ep % self.target_update_freq == 0:
            self.target_Q_net.load_state_dict(self.Q_net.state_dict())