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
from env_utils import Phi_fn_cartpole
import yaml

# Define seeding
#np.random.seed(10)

# Defining env
env = gym.make('CartPole-v1')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNetwork(nn.Module):

    def __init__(self, image_dim, output_dim,):
        super(Net, self).__init__()

        # Shape of input image
        C_in, H_in, W_in = image_dim
        image = torch.rand(image_dim, device=device)

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
            n_flatten = self.cnn(image).shape[1]

        self.fc1 = nn.Linear(n_flatten, 512)
        self.head = nn.Linear(512, output_dim)

    def forward(self, x):

        x = x.to(device)
        x = self.cnn(x)
        x = F.relu(self.fc1(x))
        x = self.head(x)

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

class DQN(nn.Module):

    def __init__(self, image_dim, replay_buffer_cap, minibatch_size, n_episodes, eps_start, eps_end, eps_delay, 
                history_length, skip_frames, target_update_freq, gamma):

        super(DQN, self).__init__()

        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        # Initial DQN networks
        self.Q_net = DQNetwork(image_dim, self.n_actions)
        self.target_Q_net.load_state_dict(self.Q_net.state_dict())
        self.target_Q_net.eval()

        self.er = ExperienceRelay(replay_buffer_cap, minibatch_size)

        self.optimizer = optim.RMSprop(self.Q_net.parameters())
        self.loss_fn = nn.SmoothL1Loss()
        
        self.n_episodes = n_episodes
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_delay = eps_delay
        self.experience_count = 0 # Number of experiences (or transitions) seen until now
        self.skip_frames = skip_frames
        self.target_update_freq = target_update_freq
        self.gamma = gamma

        self.preprocess_fn = Phi_fn_cartpole(env, history_length)

    def select_action(self, current_phi):

        eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.experience_count/self.eps_delay)

        self.experience_count += 1

        if random.random() > eps_thresh:
            with torch.no_grad():
                return self.Q_net(current_phi).max(1)[1].view(1, 1).item()
        else:
            return env.action_space.sample()

    def optimize_model(self, minibatch):

        transition_minibatch = self.er.Transition(*zip(*minibatch))

        current_phi_mb = torch.stack(transition_minibatch.current_phi).to(device)
        action_mb = torch.stack(transition_minibatch.action)
        reward_mb = torch.stack(transition_minibatch.reward)

        # State action values of current_phi using Q-net
        current_phi_estimates = self.Q_net(current_phi_mb).gather(1, action_mb)

        # Boolean indexes where next_phi is not terminal
        # non_terminal_idxs = torch.tensor(tuple(map(lambda next_phi: next_phi is not None, transition_minibatch.next_phi)), device=device, dtype=torch.bool)

        non_terminal_idxs = torch.tensor(transition_minibatch.done, dtype=torch.bool)

        non_terminal_next_phi_mb = torch.stack([ns for ns in transition_minibatch.next_phi if ns is not None]).to(device)
        next_phi_max_qvals = torch.zeros(self.er.minibatch_size, device=device)
        next_phi_max_qvals[non_terminal_idxs] = self.target_Q_net(non_terminal_next_phi_mb).max(1)[0].detach()

        # Target Q values
        y = reward_mb + self.gamma * next_phi_max_qvals

        self.loss_fn(y, current_phi_estimates)
        self.optmizer.zero_grad()
        self.loss_fn.loss.backward()

        ###
        # PyTorch clamps gradient whereas SB3 clips grad norm. We do neither. We test how it works first with Huber (SmoothL1) loss.
        # https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
        ###

        self.optimizer.step()        

    def train(self):

        for ep in range(self.n_episodes):

            current_phi, screen_buffer = self.preprocess_fn.get_initial_framestack(env)
            action = self.select_action(current_phi) # Item is used to extract the action from the tensor
            done = False

            while not done:

                cum_reward_skipped_frames = 0
                for sf_count in range(self.skip_frames):
                    obs, reward, done, _ = env.step(action)    
                    screen_buffer.append(env.render(mode='rgb_array'))
                    cum_reward_skipped_frames += reward
                    if done:
                        break

                self.preprocess_fn.update_framestack(screen_buffer)
                next_phi = torch.stack(self.preprocess_fn.framestack, dim=0)
                self.er.store(current_phi, action, float(np.clip(cum_reward_skipped_frames, -1, 1)), next_phi, done)

                if not done:
                    current_phi = next_phi
                    action = self.select_action(current_phi)
                
                if len(self.er.buffer) > 50 * self.er.minibatch_size:
                    minibatch = self.er.sample_minibatch()
                    self.optimize_model(minibatch)
             
        if ep % self.target_update_freq == 0:
            self.target_Q_net.load_state_dict(self.Q_net.state_dict())

env.render()
env.close()

if __name__ == "__main__":
    with open('./dqn.yml') as f:
        hyper_params = yaml.load(f)
        model = DQN(*hyper_params[dqn_algo].values())
        model.train()