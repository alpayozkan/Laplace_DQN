# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import monotonicnetworks as lmn

class DQNNet(nn.Module):

    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.num_sensitivities = config.num_sensitivities
        output_dim = self.action_dim * self.num_sensitivities

        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(self.input_dim, 64) # originally 128
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)
        self.reshape_layer = lambda x: x.view(-1, self.action_dim, self.num_sensitivities)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) 
        x = self.reshape_layer(x)
        
        x = torch.cumsum(x, dim=2)
        x = torch.flip(x, dims=[2])
                
        diffs = torch.diff(x, dim=2)
        #print(x.shape, diffs.shape, x, diffs)
        # assert x == diffs
        assert torch.all(diffs <= 0), "Output Tensor is not non-increasing along dimension 2"
        
        return x
    
    # for multiple gammas 
class DQNNetMultgam(nn.Module):

    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim + 1 # add one for the discount factor 
        self.action_dim = config.action_dim
        self.num_sensitivities = config.num_sensitivities
        output_dim = self.action_dim * self.num_sensitivities

        super(DQNNetMultgam, self).__init__()
        self.layer1 = nn.Linear(self.input_dim, 64) # originally 128
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)
        self.reshape_layer = lambda x: x.view(-1, self.action_dim, self.num_sensitivities)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) 
        x = self.reshape_layer(x)
        
        x = torch.cumsum(x, dim=2)
        x = torch.flip(x, dims=[2])
                
        diffs = torch.diff(x, dim=2)
        #print(x.shape, diffs.shape, x, diffs)
        # assert x == diffs
        assert torch.all(diffs <= 0), "Output Tensor is not non-increasing along dimension 2"
        
        return x


# Build a Lipschitz-1 network
class RobustModel(torch.nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.model = torch.nn.Sequential(
            lmn.direct_norm(torch.nn.Linear(1, width), kind="one-inf"),
            lmn.GroupSort(2),
            lmn.direct_norm(torch.nn.Linear(width, width), kind="inf"),
            lmn.GroupSort(2),
            lmn.direct_norm(torch.nn.Linear(width, width), kind="inf"),
            lmn.GroupSort(2),
            lmn.direct_norm(torch.nn.Linear(width, 1), kind="inf"),
        )
        
    def forward(self, x):
        return self.model(x)

# Build a Lipschitz-1 network
class RobustModelTiny(torch.nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.model = torch.nn.Sequential(
            lmn.direct_norm(torch.nn.Linear(1, width), kind="one-inf"),
            # lmn.GroupSort(2),
            # lmn.direct_norm(torch.nn.Linear(width, width), kind="inf"),
            # lmn.GroupSort(2),
            # lmn.direct_norm(torch.nn.Linear(width, width), kind="inf"),
            lmn.GroupSort(2),
            lmn.direct_norm(torch.nn.Linear(width, 1), kind="inf"),
        )
        
    def forward(self, x):
        return self.model(x)
    
class DQNNetMultgamMonot(nn.Module):

    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim + 1 # add one for the discount factor 
        self.action_dim = config.action_dim
        self.num_sensitivities = config.num_sensitivities
        output_dim = self.action_dim * self.num_sensitivities

        super(DQNNetMultgamMonot, self).__init__()
        self.layer1 = nn.Linear(self.input_dim, 64) # originally 128
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)
        # self.reshape_layer = lambda x: x.view(-1, self.action_dim, self.num_sensitivities)

        # self.monot_net = lmn.SigmaNet(RobustModelTiny(), 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) 
        x = x.reshape(-1, self.action_dim, self.num_sensitivities)
        
        x = torch.cumsum(x, dim=2)
        x = torch.flip(x, dims=[2])
                
        # x = x.reshape(-1,1)
        # x = self.monot_net(x)
        x = x.reshape(-1, self.action_dim, self.num_sensitivities)

        diffs = torch.diff(x, dim=2)
        eps = 1e-4
        #print(x.shape, diffs.shape, x, diffs)
        # assert x == diffs
        assert torch.all(diffs <= eps), "Output Tensor is not non-increasing along dimension 2"
                
        return x 