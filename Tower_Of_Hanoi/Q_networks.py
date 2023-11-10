# Q-Learning Agents


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,in_features,out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features,out_features)
        #self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(out_features,out_features)
        #self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self,x):
        identity = x
        out = self.fc1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,input_units,output_units,seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_units, 512)
        self.fc2 = nn.Linear(512, 256)
        self.res_block1 = ResidualBlock(256,256)
        #self.bn1 = nn.BatchNorm1d(5000)
        #self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(256,output_units)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out = self.fc1(state)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.bn2(out)
        out = self.relu(out)
        out = self.res_block1(out)

        return self.fc3(out)
    
class QNetwork_UVFA(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,input_units,output_units,seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork_UVFA, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(2*input_units, 512)
        self.fc2 = nn.Linear(512, 256)
        self.res_block1 = ResidualBlock(256,256)
        #self.bn1 = nn.BatchNorm1d(5000)
        #self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(256,output_units)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out = self.fc1(state)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.bn2(out)
        out = self.relu(out)
        out = self.res_block1(out)
        #out = self.res_block2(out)
        #out = self.res_block3(out)
        #out = self.res_block4(out)

        return self.fc3(out)

'''
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_units, output_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_units, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_units)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))  # Apply sigmoid activation to the output layer

class QNetwork_UVFA(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,input_units,output_units,seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork_UVFA, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(2*input_units, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_units)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))  # Apply sigmoid activation to the output layer
'''
