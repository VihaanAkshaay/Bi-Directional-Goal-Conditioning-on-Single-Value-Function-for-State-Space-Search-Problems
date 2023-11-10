# All Learning Agents here

import Q_networks 
import Replay_Buffer

import numpy as np
import random

import torch
import numpy as np
from collections import deque, namedtuple

#import learning_agents
#from cube_env import Cube,visualise

import torch
import torch.nn as nn  
import torch.nn.functional as F

import numpy as np
import random

import torch
import numpy as np
from collections import deque, namedtuple

import torch.optim as optim

import torch
import torch.nn as nn  
import torch.nn.functional as F

# Setting device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Setting parameters:
BUFFER_SIZE = 100000 # replay buffer size
BATCH_SIZE = 32     # minibatch size
GAMMA = 0.99          # discount factor
LR = 5e-4              # learning rate 
UPDATE_EVERY = 1        # how often to update the network (When Q target is present)

############# SRE-DQN #######################

class Agent_SREDQN():
    """
    Encompases Scrambler and Repeater and Explorer
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = Q_networks.QNetwork_UVFA(state_size, action_size, seed).to(device) # Policy network used by the repeater
        self.qnetwork_target = Q_networks.QNetwork_UVFA(state_size, action_size, seed).to(device) # We use network_target as a less_frequently updated NN which is used by the scrambler
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = Replay_Buffer.ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.scramble_dict = []
        
        # tau and tau_inv goals
        

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            
    def add_to_buffer(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
    def train_call(self):

        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        """ +Q TARGETS PRESENT """
        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

            
    def itr_priority(self, s_0,c,g):
        num_q = self.qnetwork_target(torch.tensor(np.concatenate((c,g)), dtype=torch.float32).to(device)).detach().max().item()
        den_q = self.qnetwork_target(torch.tensor(np.concatenate((s_0,c)), dtype=torch.float32).to(device)).detach().max().item()

        if num_q*np.linalg.norm(s_0 - c)== 0:
            return 0.00001
        if np.linalg.norm(c - g) * den_q == 0:
            return 1000
        val = abs(np.linalg.norm(s_0 - c) * num_q )/abs((np.linalg.norm(c - g) * den_q))
        return val

    def scr_act(self,state,eps = 0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_target.eval()
        with torch.no_grad():
            action_values = self.qnetwork_target(state)
        self.qnetwork_target.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        """ +T TRUNCATION PRESENT
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        """
            
        self.optimizer.step()

#########################################
############# DQN #######################
#########################################

class Agent_DQN():

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = Q_networks.QNetwork(state_size, action_size, seed).to(device) # Policy network used by the repeater
        self.qnetwork_target = Q_networks.QNetwork(state_size, action_size, seed).to(device) # We use network_target as a less_frequently updated NN which is used by the scrambler
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = Replay_Buffer.ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.scramble_dict = []
        
        # tau and tau_inv goals
        

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Start the train call
        self.train_call()
            
    def add_to_buffer(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
    def train_call(self):

        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        """ +Q TARGETS PRESENT """
        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        
    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()