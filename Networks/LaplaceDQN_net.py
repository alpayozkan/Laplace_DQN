# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    
class DQNNetMultgamAtari(nn.Module):

    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim # discount factor is not included in the input_dim for atari games`
        self.action_dim = config.action_dim
        self.num_sensitivities = config.num_sensitivities

        super(DQNNetMultgamAtari, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7*7 + 1, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.action_dim * self.num_sensitivities),
            nn.ReLU() # added ReLU to insure non-negative output (not in original implementation)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state, gamma):
        gamma = gamma.unsqueeze(1)
    
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float()
        
        if state.shape[0] == 0:
            return torch.zeros(0, self.action_dim, self.num_sensitivities, dtype=state.dtype, device=state.device)
        
        conv_out = self.conv(state).view(state.size()[0],-1)
        # concatenate the discount factor to the flattened convolutional output
        conv_out = torch.cat((conv_out, gamma), dim=1)
        lin_out = self.fc(conv_out)
        out = lin_out.reshape(lin_out.shape[0], self.action_dim, self.num_sensitivities)
        
        out = torch.cumsum(out, dim=2)
        out = torch.flip(out, dims=[2])
        
        diffs = torch.diff(out, dim=2)
        assert torch.all(diffs <= 0), "Output Tensor is not non-increasing along dimension 2"
        
        return out

class DQNNetMultgamInv(nn.Module):

    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim + 1 # add one for the discount factor 
        self.action_dim = config.action_dim
        self.num_sensitivities = config.num_sensitivities
        output_dim = self.action_dim * self.num_sensitivities

        super(DQNNetMultgamInv, self).__init__()
        self.low_net = DQNNetMultgam(config)
        self.med_net = DQNNetMultgam(config)
        self.high_net = DQNNetMultgam(config)

        self.low_range = 0.5
        self.med_range = 0.75
        self.high_range = 1.0
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x: [B, num_states + 1(gamma-dim)]
        gamm_tens = x[:,-1]
        low_gam_mask = (gamm_tens >= 0.0) & (gamm_tens < 0.5)
        med_gam_mask = (gamm_tens >= 0.5) &  (gamm_tens < 0.75)
        high_gam_mask = (gamm_tens >= 0.75) &  (gamm_tens <= 1)
        
        # assumed gammas in range [0,1]
        assert (low_gam_mask.sum()+med_gam_mask.sum()+high_gam_mask.sum()) == gamm_tens.shape[0]

        # Collect the outputs from each network
        low_out = self.low_net(x[low_gam_mask])
        med_out = self.med_net(x[med_gam_mask])
        high_out = self.high_net(x[high_gam_mask])

        # Initialize the result tensor with the same dtype and device as input x
        res_tens = torch.zeros(x.shape[0], self.action_dim, self.num_sensitivities, dtype=x.dtype, device=x.device)

        # Indexes for placing outputs in the result tensor
        low_indexes = low_gam_mask.nonzero(as_tuple=True)[0]
        med_indexes = med_gam_mask.nonzero(as_tuple=True)[0]
        high_indexes = high_gam_mask.nonzero(as_tuple=True)[0]

        # Place the outputs in the correct positions
        res_tens[low_indexes] = low_out
        res_tens[med_indexes] = med_out
        res_tens[high_indexes] = high_out
        
        return res_tens
    
class DQNNetMultgamInvAtari(nn.Module):

    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim + 1 # add one for the discount factor 
        self.action_dim = config.action_dim
        self.num_sensitivities = config.num_sensitivities
        output_dim = self.action_dim * self.num_sensitivities

        super(DQNNetMultgamInvAtari, self).__init__()
        self.low_net = DQNNetMultgamAtari(config)
        self.med_net = DQNNetMultgamAtari(config)
        self.high_net = DQNNetMultgamAtari(config)

        self.low_range = 0.5
        self.med_range = 0.75
        self.high_range = 1.0
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state, gamma):
        # x: [B, num_states + 1(gamma-dim)]
        gamma = gamma[:,0]
        low_gam_mask = (gamma >= 0.0) & (gamma < 0.5)
        med_gam_mask = (gamma >= 0.5) &  (gamma < 0.75)
        high_gam_mask = (gamma >= 0.75) &  (gamma <= 1)
        
        # assumed gammas in range [0,1]
        assert (low_gam_mask.sum()+med_gam_mask.sum()+high_gam_mask.sum()) == gamma.shape[0]

        # Collect the outputs from each network        
        low_out = self.low_net(state[low_gam_mask], gamma[low_gam_mask])
        med_out = self.med_net(state[med_gam_mask], gamma[med_gam_mask])
        high_out = self.high_net(state[high_gam_mask], gamma[high_gam_mask])

        # Initialize the result tensor with the same dtype and device as input x
        res_tens = torch.zeros(state.shape[0], self.action_dim, self.num_sensitivities, dtype=state.dtype, device=state.device)

        # Indexes for placing outputs in the result tensor
        low_indexes = low_gam_mask.nonzero(as_tuple=True)[0]
        med_indexes = med_gam_mask.nonzero(as_tuple=True)[0]
        high_indexes = high_gam_mask.nonzero(as_tuple=True)[0]

        # Place the outputs in the correct positions
        res_tens[low_indexes] = low_out
        res_tens[med_indexes] = med_out
        res_tens[high_indexes] = high_out
        
        return res_tens