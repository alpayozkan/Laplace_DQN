# -*- coding:utf-8 -*-
from utils.replay_memory import ReplayMemory, Transition
from utils.reproducibility import set_seed
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from Networks.LaplaceDQN_net import * # testing with 
from Networks.LaplaceDQN_net_monot_multgam import *
from utils.Inverse_Laplace import SVD_approximation_inverse_Laplace
from itertools import count
import random
import math
import torch
from envs.four_state_mdp import Simple4MDP
import os

import gym
from config import config_categ

import time
  

class LaplaceDQNAgentMultgam_4smdp:

    def __init__(self, game):
        #env = gym.make(game, None) # 4 state MDP
        env = Simple4MDP() 
        config = config_categ[game](env)
        self.config = config

        # self.envs = [env] + [gym.make(game, None) for _ in range(config.num_gamma-1)]
        self.envs = [env] + [Simple4MDP() for _ in range(config.num_gamma-1)] # NOTE 4 state MDP
        for env in self.envs:
            env.action_space.seed(config.seed)
        
        self.config = config 
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim

        self.num_episodes = config.num_episodes
        self.steps = config.initial_number_steps
        self.BATCH_SIZE = config.BATCH_SIZE
        self.LR = config.LR
        self.TAU = config.TAU
        self.device = config.device
        
        self.num_gamma = config.num_gamma 
        self.gamma_min = config.gamma_min
        self.gamma_max = config.gamma_max
        #start = 1 / np.log(self.gamma_min) 
        start = 1 / np.log(0.99) # with 1 gamma TODO
        end = 1 / np.log(self.gamma_max)
        #end = 1 / np.log(0.95)   
        self.gammas = torch.exp(torch.true_divide(1, torch.linspace(start, end, self.num_gamma)))
        self.num_sensitivities = config.num_sensitivities 
        self.rmin = config.rmin
        self.rmax = config.rmax
        assert self.rmin < self.rmax, "The reward range is not valid"
        self.sensitivity_step = (self.rmax - self.rmin) / self.num_sensitivities
        self.sensitivities = torch.arange(self.rmin, self.rmax, self.sensitivity_step)
        self.middle_sensitivities = torch.tensor([torch.true_divide(self.sensitivities[i] + self.sensitivities[i+1], 2) for i in range(self.sensitivities.shape[0]-1)])
        self.activ_sharpness = config.activ_sharpness
        
        self.num_gamma_to_tau = config.num_gamma_to_tau
        self.gamma_to_tau_min = config.gamma_to_tau_min
        self.gamma_to_tau_max = config.gamma_to_tau_max
        start = 1 / np.log(self.gamma_to_tau_min) 
        end = 1 / np.log(self.gamma_to_tau_max)   
        self.gammas_to_tau = torch.exp(torch.true_divide(1, torch.linspace(start, end, self.num_gamma_to_tau)))
        time_horizon_change = None # when time horizon changes during training
        
        # taus=torch.linspace(0.01,3,self.num_gamma) # as in implementation from Pablo Tano
        # self.gammas=torch.exp(-1/taus)
        print("Gammas : ", self.gammas)
        
        self.total_steps = [0 for _ in range(self.num_gamma)]

        # reproducibility
        self.seed = config.seed
        set_seed(self.seed)
        
        self.dir = os.getcwd()

        # self.env = None
        # copying weights of base_net to policy_net and target_net
        self.policy_net = DQNNetMultgamMonot(self.config) 
        self.target_net = DQNNetMultgamMonot(self.config)
        #self.policy_net = DQNNetMultgam(self.config)
        #self.target_net = DQNNetMultgam(self.config)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_memory = [ReplayMemory(self.replay_buffer_size) for _ in range(self.num_gamma)] 
        # have a replay buffer for every gamma (leads to different policies for each)
        #self.replay_buffers = [ReplayMemory(self.replay_buffer_size) for _ in range(self.num_gamma)]
        
        self.episode_durations = []
        self.check_model_improved = [torch.tensor([0]) for _ in range(self.num_gamma)]
        self.best_max = [torch.tensor([0]) for _ in range(self.num_gamma)]

        # for select action (epsilon-greedy)
        self.steps_done = [0 for _ in range(self.num_gamma)]
        
        # save for plotting evolution during training for each gamma
        self.model_reward_hist = [[] for _ in range(self.num_gamma)]
        self.model_loss_hist = [[] for _ in range(self.num_gamma)]
    
    def transition(self):
        """
        In transition, the agent simply plays and records
        [current_state, action, reward, next_state, done]
        in the replay_memory

        Updating the weights of the neural network happens
        every single time the replay buffer size is reached.

        done: boolean, whether the game has ended or not.
        """
        # Start time
        self.start_time = time.time()
        
        # multiple environments for every gamma
        # self.envs = [self.env] * self.num_gamma
        self.states = [torch.from_numpy(env.reset(seed=self.seed)[0]).float().unsqueeze(0) for env in self.envs] 
            
        for i_episode in range(self.num_episodes):
            #state, info = self.env.reset(seed=self.seed) # for 1 gammas
            #state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)


            for gamma_idx, gamma in enumerate(self.gammas):
                print('Episode: {} Reward: {} Max_Reward: {}'.format(i_episode, self.check_model_improved[gamma_idx].item(), self.best_max[gamma_idx].item()))
                print('-' * 64)
                self.check_model_improved[gamma_idx] = 0
                # print("DEBUG in transitions: ", t)
                for t in count():    
                    env = self.envs[gamma_idx]
                    state = self.states[gamma_idx]
                    
                    action = self.select_action(gamma_idx) 
                    observation, reward, terminated, truncated, _ = env.step(action.item())
                    reward = torch.tensor([reward], device=self.device)
                    # print("DEBUG Done terminated, truncated", terminated, truncated)
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Store the transition in memory
                    self.replay_memory[gamma_idx].push(state, action, next_state, reward)

                    # Move to the next state
                    self.states[gamma_idx] = next_state
                    #state = next_state 
                    self.total_steps[gamma_idx] += 1

                    # Perform one step of the optimization (on the policy network)
                    self.train_by_replay(gamma_idx) 

                    # Soft update of the target network's weights 
                    # θ′ ← τ θ + (1 −τ )θ′
                    # previous implementation updates were done for any episode where the reward is higher
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                    self.target_net.load_state_dict(target_net_state_dict)
                    
                    self.check_model_improved[gamma_idx] += reward # check if this solves the 4 state MDP problem
                    if done:
                        self.episode_durations.append(t + 1)
                        self.model_reward_hist[gamma_idx].append(self.check_model_improved[gamma_idx].detach().numpy())
                        self.states[gamma_idx] = torch.from_numpy(self.envs[gamma_idx].reset(seed=self.seed)[0]).float().unsqueeze(0)
                        break # can't break anymore -> need to reset to not have all episodes restart TODO
                    # else:
                    #     self.check_model_improved[gamma_idx] += reward

                if self.check_model_improved[gamma_idx] > self.best_max[gamma_idx]:
                    self.best_max[gamma_idx] = self.check_model_improved[gamma_idx]
                    
        folder = 'Results'
        subfolder_reward = 'Rewards'
        subfolder_loss = "Losses"
        subfolder_weights = "Weights"
        filename_rewards = 'rewards_laplace_mult.npy'
        filename_losses = 'losses_laplace_mult.npy'
        filename_weights = 'policy_net_weights_laplace_mult.pth'
        path_rewards = os.path.join(self.dir, folder, subfolder_reward, filename_rewards)
        path_losses = os.path.join(self.dir, folder, subfolder_loss, filename_losses)
        path_weights = os.path.join(self.dir, folder, subfolder_weights, filename_weights)
                
        # Save the losses and rewards to numpy arrays
        cum_reward_per_episode = np.array([self.model_reward_hist[i] for i in range(self.num_gamma)])
        np.save(path_rewards, cum_reward_per_episode)
        
        # arrange dimension difference due to difference transition lengths
        # append -1, to concatenate them, since loss cannot be -1
        vectors = self.model_loss_hist
        max_length = max(len(vec) for vec in vectors)
        padded_vectors = [np.pad(vec, (0, max_length - len(vec)), 'constant', constant_values=-1) for vec in vectors]
        loss_result = np.vstack(padded_vectors)
        np.save(path_losses, np.array(loss_result)) # check if I can save a numpy array of lists

        torch.save(self.policy_net.state_dict(), path_weights)
        
        # End time
        self.end_time = time.time()
        
    # for multiple gammas 
    def train_by_replay(self, gamma_idx):
        """
        TD update by replaying the history.
        """
        # step 1: generate replay samples (size = self.batch_size) from the replay buffer
        # e.g. uniform random replay or prioritize experience replay
        if len(self.replay_memory[gamma_idx]) < self.BATCH_SIZE:
            return
        transitions = self.replay_memory[gamma_idx].sample(self.BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        #print("state_batch[:10]", state_batch[:10])
        #print("reward_batch[:10]", reward_batch[:10])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        input = torch.cat((state_batch, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0).repeat(state_batch.shape[0], 1)), dim=1)  
        assert state_batch.shape[0] == input.shape[0]
        output = self.policy_net(input) 
        state_action_values = output[torch.arange(output.size(0)), action_batch.squeeze()]
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_output = torch.zeros(self.BATCH_SIZE, self.action_dim, self.num_sensitivities, device=self.device) # for 1 sensitivity 
        with torch.no_grad():
            next_inputs = torch.cat((non_final_next_states, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0).repeat(non_final_next_states.shape[0], 1)), dim=1) 
            next_output[non_final_mask] = self.target_net(next_inputs)
            action_values = torch.sum((next_output[:, :, :-1] - next_output[:, :, 1:]) * self.middle_sensitivities, dim=2)
            max_action_values = action_values.max(1).indices
            next_state_values = next_output[torch.arange(next_output.size(0)), max_action_values]        
        
        rewards_thresh = torch.nn.functional.sigmoid(self.activ_sharpness*(reward_batch.unsqueeze(-1).repeat(1, self.sensitivities.shape[0])-self.sensitivities.unsqueeze(0).repeat(reward_batch.shape[0], 1)))        
        
        assert torch.all((rewards_thresh <= 1) & (rewards_thresh >= 0)), "Rewards after activation should be between 0 or 1"
        
        # Compute the expected Q values 
        expected_state_action_values = (next_state_values * self.gammas[gamma_idx]) + rewards_thresh 
        
        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values) / float(self.num_gamma)
        self.model_loss_hist[gamma_idx].append(loss.detach().numpy())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def select_action(self, gamma_idx):
        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
            math.exp(-1. * self.steps_done[gamma_idx] / self.config.EPS_DECAY)
        self.steps_done[gamma_idx] += 1

        if sample > eps_threshold:
            with torch.no_grad():
                input = torch.cat((self.states[gamma_idx], torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0)), dim=1)
                
                assert self.states[gamma_idx].shape[0] == input.shape[0]
                
                z = self.policy_net(input) 
                action_values = torch.sum((z[:, :, :-1] - z[:, :, 1:]) * self.middle_sensitivities, dim=2) 
                
                assert action_values.shape[0] == self.states[gamma_idx].shape[0]
                
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return action_values.max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.envs[gamma_idx].action_space.sample()]], device=self.device, dtype=torch.long)
    


    # In the main script have a argument to do action selection in two different ways 
    # standard way before change in time horizon and this way after 
    # NOTE calls the inverse Laplace approximation
    def select_action_after_H_change(self, gamma_idx):
            sample = random.random()
            eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
                math.exp(-1. * self.steps_done[gamma_idx] / self.config.EPS_DECAY)
            self.steps_done[gamma_idx] += 1

            if sample > eps_threshold:
                with torch.no_grad():
                    input = torch.cat((self.states[gamma_idx], torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0)), dim=1)
                    
                    assert self.states[gamma_idx].shape[0] == input.shape[0]
                    # how to have it parallelized for batch size TODO 
                    # compute for every gamma (try more than what we track)
                    Q_gamma = np.zeros((self.BATCH_SIZE, self.action_dim, self.num_gamma_to_tau, self.num_sensitivities))
                    for idx, gamma in enumerate(self.num_gamma_to_tau):
                        input = torch.cat((self.states[gamma_idx], torch.tensor([gamma]).unsqueeze(0)), dim=1)
                        Q_gamma[:, :, idx, :] = self.policy_net(input) 
                    
                    tau_space = SVD_approximation_inverse_Laplace(self.config, Q_gamma)
                    
                    # eq. 13
                    action_values = torch.sum(np.dot(tau_space, self.middle_sensitivities[:-1]), dim=2)
                                        
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return action_values.max(1).indices.view(1, 1)
            else:
                return torch.tensor([[self.envs[gamma_idx].action_space.sample()]], device=self.device, dtype=torch.long)
      
      
        
    def eval_step(self, render=True):
        """
        Evaluation using the trained target network, no training involved
        :param render: whether to visualize the evaluation or not
        NOTE in this methods signal lines to change to evaluate for a different discount factor
        """
        folder = 'Results'
        subfolder = 'four_state_mdp_results'
        second_subfolder = 'large_gammas'
        
        for gamma_idx, gamma in enumerate(self.gammas):
            state_action_values = [] # 4MDP
            
            for each_ep in range(self.config.evaluate_episodes):
                state, info = self.envs[gamma_idx].reset(seed=self.seed) # NOTE evaluating for the ith gamma 
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                print('Episode: {} Reward: {} Training_Max_Reward: {}'.format(each_ep, self.check_model_improved[0].item(),
                                                                                self.best_max[0].item()))
                print('-' * 64)
                self.check_model_improved = 0

                for t in count():
                    action = self.select_action(gamma_idx=-1)
                    input = torch.cat((state, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0).repeat(state.shape[0], 1)), dim=1) # NOTE evaluating for the ith gamma
                    z = self.policy_net(input)
                    observation, reward, terminated, truncated, _ = self.envs[gamma_idx].step(action.item()) # NOTE evaluating for the ith gamma
                    reward = torch.tensor([reward], device=self.device)
                    done = terminated or truncated 

                    if render:
                        self.envs[gamma_idx].render() 
                    
                    self.check_model_improved += reward
                    if each_ep == self.config.evaluate_episodes-1:
                       state_action_values.append(z.detach().numpy())
                    if done:
                        break
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)  
                        state = next_state
                    # testing for four_state_MDP
                    print("ACTION SHAPE:", z.shape)
                    # save only for last episode since predictions will be the same 
                    # given the agent transits deterministically through the MDP states
                    # if each_ep == self.config.evaluate_episodes-1:
                    #    state_action_values.append(z.detach().numpy())
            filename_values = f'values_4_state_monot_{gamma:.4f}_{gamma_idx}'
            path_values = os.path.join(self.dir, folder, subfolder, second_subfolder, filename_values)
            np.save(path_values, np.array(state_action_values))
        
        print('Complete')
        
        # Calculate the runtime
        runtime = self.end_time - self.start_time

        print(f'Runtime: {runtime} seconds')
            