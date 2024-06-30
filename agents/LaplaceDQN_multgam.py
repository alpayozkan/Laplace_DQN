# -*- coding:utf-8 -*-
from utils.replay_memory import ReplayMemory, Transition
from utils.reproducibility import set_seed
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from Networks.LaplaceDQN_net import * 
from Networks.LaplaceDQN_net_monot_multgam import *
from utils.Inverse_Laplace import SVD_approximation_inverse_Laplace
from itertools import count
import random
import math
import torch
import os

import gymnasium as gym
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)
import pickle
from config import config_categ
import time

from stable_baselines3.common.monitor import Monitor

class LaplaceDQNAgentMultgam:

    def __init__(self, game, config):
        # env = gym.make(game, None) 
        # config = Config(env)
        self.config = config

        self.envs = [gym.make(game, None) for _ in range(config.num_gamma)]
        # self.envs = [env] + [gym.make(game, None) for _ in range(config.num_gamma-1)]

        for env in self.envs:
            env.action_space.seed(config.seed)
        
        self.config = config 
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim

        self.num_episodes = config.num_episodes
        self.initial_number_steps = config.initial_number_steps
        self.later_number_steps = config.later_number_steps
        self.max_steps = self.initial_number_steps
        self.BATCH_SIZE = config.BATCH_SIZE
        self.LR = config.LR
        self.TAU = config.TAU
        self.device = config.device
        
        self.num_gamma = config.num_gamma 
        self.gamma_min = config.gamma_min
        self.gamma_max = config.gamma_max
        start = 1 / np.log(self.gamma_min) 
        #start = 1 / np.log(0.99) # with 1 gamma TODO
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
        self.K = config.K
        self.time_horizon_change = config.time_horizon_change
        
        # taus=torch.linspace(0.01,3,self.num_gamma) # as in implementation from Pablo Tano
        # self.gammas=torch.exp(-1/taus)
        print("Gammas : ", self.gammas)
        
        self.total_steps = [0 for _ in range(self.num_gamma)]

        # reproducibility
        self.seed = config.seed
        set_seed(self.seed)
        
        self.dir = os.getcwd()

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
        
        self.episode_durations = []
        self.check_model_improved = [torch.tensor([0]) for _ in range(self.num_gamma)]
        self.best_max = [torch.tensor([0]) for _ in range(self.num_gamma)]

        # for select action (epsilon-greedy)
        self.steps_done = [0 for _ in range(self.num_gamma)]
        
        # save for plotting evolution during training for each gamma
        self.model_reward_hist = [[] for _ in range(self.num_gamma)]
        self.model_loss_hist = [[] for _ in range(self.num_gamma)]
    
        # monitoring parameters
        # also setup config or add to config before config is passed to the model
        self.check_freq = config.check_freq
        self.log_dir = config.log_dir
        self.verbose = config.verbose
        self.best_mean_reward = -np.inf

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
        self.states = [torch.from_numpy(env.reset(seed=self.seed)[0]).float().unsqueeze(0) for env in self.envs] 
            
        for i_episode in range(self.num_episodes):
            # No horizon change during training for now
            # if self.time_horizon_change is not None and i_episode == self.time_horizon_change:
            #     self.max_steps = self.later_number_steps

            for gamma_idx, gamma in enumerate(self.gammas):
                print('Episode: {} Reward: {} Max_Reward: {}'.format(i_episode, self.check_model_improved[gamma_idx].item(), self.best_max[gamma_idx].item()))
                print('-' * 64)
                self.check_model_improved[gamma_idx] = 0
                for t in count():    
                    env = self.envs[gamma_idx]
                    state = self.states[gamma_idx]
                    
                    action = self.select_action(gamma_idx)
                    # No horizon change during training for now
                    # if (self.time_horizon_change is None) or (self.time_horizon_change is not None and i_episode < self.time_horizon_change):
                    #     action = self.select_action(gamma_idx)
                    # else:
                    #     action = self.select_action_after_H_change(gamma_idx) 
                        
                    observation, reward, terminated, truncated, _ = env.step(action.item())
                    reward = torch.tensor([reward], device=self.device)
                    done = terminated or truncated

                    # if terminated or t == self.max_steps:
                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Store the transition in memory
                    self.replay_memory[gamma_idx].push(state, action, next_state, reward)

                    # Move to the next state
                    self.states[gamma_idx] = next_state
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
                    
                    self.check_model_improved[gamma_idx] += reward 
                    # if done or t == self.max_steps:
                    if done:
                        self.episode_durations.append(t + 1)
                        self.model_reward_hist[gamma_idx].append(self.check_model_improved[gamma_idx].detach().numpy())
                        self.states[gamma_idx] = torch.from_numpy(self.envs[gamma_idx].reset(seed=self.seed)[0]).float().unsqueeze(0)
                        break 

                    self._on_step()

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
        """
           Select action before the time horizon change
        """
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
    

    def select_action_after_H_change(self, gamma_idx):
        """
           Select action after the time horizon change using the inverse Laplace transform
        """
        # sample = random.random()
        # eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
        #     math.exp(-1. * self.steps_done[gamma_idx] / self.config.EPS_DECAY)
        # self.steps_done[gamma_idx] += 1

        # if sample > eps_threshold:
        with torch.no_grad():
            input = torch.cat((self.states[gamma_idx], torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0)), dim=1)
            
            assert self.states[gamma_idx].shape[0] == input.shape[0]
            Q_gamma = np.zeros((1, self.action_dim, self.num_gamma_to_tau, self.num_sensitivities))
            for idx, gamma in enumerate(self.gammas_to_tau):
                input = torch.cat((self.states[gamma_idx], torch.tensor([gamma]).unsqueeze(0)), dim=1)
                Q_gamma[:, :, idx, :] = self.policy_net(input) 
            
            tau_space = SVD_approximation_inverse_Laplace(self.config, Q_gamma)
            
            # eq. 13
            gammas_pow_tau = torch.tensor([self.gammas[gamma_idx]**tau for tau in range(self.K)])
            action_values = torch.matmul(torch.matmul(tau_space, self.middle_sensitivities[:-1]), gammas_pow_tau)
            
            print("Action Values shape after change in time horizon", action_values.shape)
                                
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return action_values.max(1).indices.view(1, 1)
        # else:
        #     return torch.tensor([[self.envs[gamma_idx].action_space.sample()]], device=self.device, dtype=torch.long)
      
        
    def eval_step(self, render=True):
        """
        Evaluation using the trained target network, no training involved
        :param render: whether to visualize the evaluation or not
        """
        self.max_steps = self.initial_number_steps
        
        for gamma_idx in range(self.gammas):
            
            for each_ep in range(self.config.evaluate_episodes):
                
                if self.time_horizon_change is not None and each_ep == self.time_horizon_change:
                   self.max_steps = self.later_number_steps
                
                state, info = self.envs[gamma_idx].reset(seed=self.seed) 
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                print('Episode: {} Reward: {} Training_Max_Reward: {}'.format(each_ep, self.check_model_improved[0].item(),
                                                                                self.best_max[0].item()))
                print('-' * 64)
                self.check_model_improved = 0

                for t in count(): 
                    if (self.time_horizon_change is None) or (self.time_horizon_change is not None and each_ep < self.time_horizon_change):
                        action = self.select_action(gamma_idx)
                    else:
                        action = self.select_action_after_H_change(gamma_idx) 
                        
                    action = self.select_action(gamma_idx=-1)
                    input = torch.cat((state, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0).repeat(state.shape[0], 1)), dim=1) # NOTE evaluating for the ith gamma
                    z = self.policy_net(input)
                    observation, reward, terminated, truncated, _ = self.envs[gamma_idx].step(action.item()) 
                    reward = torch.tensor([reward], device=self.device)
                    done = terminated or truncated 

                    if render:
                        self.envs[gamma_idx].render() 
                    
                    self.check_model_improved += reward
                
                    if done or t == self.max_steps:
                        break
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)  
                        state = next_state
        print('Complete')
        
        # Calculate the runtime
        runtime = self.end_time - self.start_time
        print(f'Runtime: {runtime} seconds')
            
    def _on_step(self) -> bool:
        # only log the last ie. the largest gamma

        if self.total_steps[-1] % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.log_dir))
                    # self.model.save(self.save_path)
                    filename = os.path.join(self.log_dir, 'best_model.pkl')
                    self.save_model(filename)
        return True

    def save_model(self, filename):
        # Save the object to a file
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj
    
class LaplaceDQNAgentMultgamIntv:

    def __init__(self, game, config, train=True):
        # env = gym.make(game, None) 
        # config = Config(env)
        print("Playing game: ", game)
        
        atari_game_list = ["Breakout-v0", "Defender-v0", "Hero-v0", "Tutankham-v0", "Seaquest-v0"]
        if game in atari_game_list:
            self.is_atari = True 
        else:
            self.is_atari = False
            
        self.game = game
        self.config = config
        
        self.num_gamma_per_module = config.num_gamma_per_module 
        self.num_gamma = self.num_gamma_per_module * 3 # three modules used currently
        
        if self.is_atari:
            self.envs = [gym.make(game, render_mode="rgb_array") for _ in range(self.num_gamma)]
        else:
            self.envs = [gym.make(game, None) for _ in range(self.num_gamma)]
        # self.envs = [env] + [gym.make(game, None) for _ in range(config.num_gamma-1)]
        
        if self.is_atari:
            for env_idx, env in enumerate(self.envs):
                #env = gym.wrappers.RecordEpisodeStatistics(env)
                self.envs[env_idx] = NoopResetEnv(env, noop_max=30)
                self.envs[env_idx] = MaxAndSkipEnv(self.envs[env_idx], skip=4)
                self.envs[env_idx] = EpisodicLifeEnv(self.envs[env_idx])

                if "FIRE" in self.envs[env_idx].unwrapped.get_action_meanings():
                    self.envs[env_idx] = FireResetEnv(self.envs[env_idx])
                
                self.envs[env_idx] = ClipRewardEnv(self.envs[env_idx])
                self.envs[env_idx] = gym.wrappers.ResizeObservation(self.envs[env_idx], (84, 84))
                self.envs[env_idx] = gym.wrappers.GrayScaleObservation(self.envs[env_idx])
                self.envs[env_idx] = gym.wrappers.FrameStack(self.envs[env_idx], 4)
                self.envs[env_idx].action_space.seed(config.seed)
                # env = gym.wrappers.RecordVideo(
                #     env, video_folder='./video/', episode_trigger=lambda episode_id: episode_id % 5 == 0) # only works with render_mode="human"
        
        self.config = config 
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim

        self.num_episodes = config.num_episodes
        self.initial_number_steps = config.initial_number_steps
        self.later_number_steps = config.later_number_steps
        self.max_steps = self.initial_number_steps
        self.BATCH_SIZE = config.BATCH_SIZE
        self.LR = config.LR
        self.TAU = config.TAU
        self.device = config.device
        
        # copying weights of base_net to policy_net and target_net
        if self.is_atari:
            self.policy_net = DQNNetMultgamInvAtari(self.config).to(self.device)
            self.target_net = DQNNetMultgamInvAtari(self.config).to(self.device)
        else:
            self.policy_net = DQNNetMultgamInv(self.config).to(self.device)
            self.target_net = DQNNetMultgamInv(self.config).to(self.device)
            #self.policy_net = DQNNetMultgam(self.config)
            #self.target_net = DQNNetMultgam(self.config)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # self.num_gamma = config.num_gamma 
        # self.gamma_min = config.gamma_min
        # self.gamma_max = config.gamma_max
        # start = 1 / np.log(self.gamma_min) 
        # #start = 1 / np.log(0.99) # with 1 gamma TODO
        # end = 1 / np.log(self.gamma_max)
        # #end = 1 / np.log(0.95)   
        # self.gammas = torch.exp(torch.true_divide(1, torch.linspace(start, end, self.num_gamma)))
        
        low_range = self.policy_net.low_range
        med_range = self.policy_net.med_range
        
        self.gamma_min = config.gamma_min
        self.gamma_max = config.gamma_max
        #start = 1 / np.log(self.gamma_min) 
        start = self.gamma_min
        #start = 1 / np.log(0.99) # with 1 gamma TODO
        #end = 1 / np.log(self.gamma_max)
        end = self.gamma_max
        #end = 1 / np.log(0.95)   
        #self.gammas = torch.exp(torch.true_divide(1, torch.linspace(start, end, self.num_gamma)))
        gammas_low = torch.linspace(start, low_range, self.num_gamma_per_module+1, device=self.device)
        gammas_med = torch.linspace(low_range, med_range, self.num_gamma_per_module+1, device=self.device)
        gammas_high = torch.linspace(med_range, end, self.num_gamma_per_module, device=self.device)
        
        self.gammas = torch.cat([gammas_low[:-1], gammas_med[:-1], gammas_high])

        self.num_sensitivities = config.num_sensitivities 
        self.rmin = config.rmin
        self.rmax = config.rmax
        assert self.rmin < self.rmax, "The reward range is not valid"
        self.sensitivity_step = (self.rmax - self.rmin) / self.num_sensitivities
        self.sensitivities = torch.arange(self.rmin, self.rmax, self.sensitivity_step)
        self.middle_sensitivities = torch.tensor([torch.true_divide(self.sensitivities[i] + self.sensitivities[i+1], 2) for i in range(self.sensitivities.shape[0]-1)], device=self.device)
        self.activ_sharpness = config.activ_sharpness
        
        self.num_gamma_to_tau = config.num_gamma_to_tau
        self.gamma_to_tau_min = config.gamma_to_tau_min
        self.gamma_to_tau_max = config.gamma_to_tau_max
        start = 1 / np.log(self.gamma_to_tau_min) 
        end = 1 / np.log(self.gamma_to_tau_max)   
        self.gammas_to_tau = torch.exp(torch.true_divide(1, torch.linspace(start, end, self.num_gamma_to_tau)))
        self.K = config.K
        self.time_horizon_change = config.time_horizon_change
        
        # taus=torch.linspace(0.01,3,self.num_gamma) # as in implementation from Pablo Tano
        # self.gammas=torch.exp(-1/taus)
        print("Gammas : ", self.gammas)
        
        self.total_steps = [0 for _ in range(self.num_gamma)]

        # reproducibility
        self.seed = config.seed
        set_seed(self.seed)
        
        self.dir = os.getcwd()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_memory = [ReplayMemory(self.replay_buffer_size) for _ in range(self.num_gamma)] 
        
        self.episode_durations = []
        self.check_model_improved = [torch.tensor([0]) for _ in range(self.num_gamma)]
        self.best_max = [torch.tensor([0]) for _ in range(self.num_gamma)]

        # for select action (epsilon-greedy)
        self.steps_done = [0 for _ in range(self.num_gamma)]
        
        # save for plotting evolution during training for each gamma
        self.model_reward_hist = [[] for _ in range(self.num_gamma)]
        self.model_loss_hist = [[] for _ in range(self.num_gamma)]
    
        # monitoring parameters
        # also setup config or add to config before config is passed to the model
        self.check_freq = config.check_freq
        self.log_dir = config.log_dir
        self.verbose = config.verbose
        self.best_mean_reward = -np.inf

        # instantiate the monitor for the largest gamma environment
        self.train = train
        if train:
            self.envs[-1] = Monitor(self.envs[-1], self.log_dir)

    # re-init with the config
    def re_init(self):
        # for consistency with the __init__
        # function is the same as init except we assume config is updated, and we re-assign params
        game = self.game
        config = self.config
        train = self.train

        atari_game_list = ["Breakout-v0", "Defender-v0", "Hero-v0", "Tutankham-v0", "Seaquest-v0"]
        if game in atari_game_list:
            self.is_atari = True 
        else:
            self.is_atari = False
            
        self.game = game
        self.config = config
        
        self.num_gamma_per_module = config.num_gamma_per_module 
        self.num_gamma = self.num_gamma_per_module * 3 # three modules used currently
        
        if self.is_atari:
            self.envs = [gym.make(game, render_mode="rgb_array") for _ in range(self.num_gamma)]
        else:
            self.envs = [gym.make(game, None) for _ in range(self.num_gamma)]
        # self.envs = [env] + [gym.make(game, None) for _ in range(config.num_gamma-1)]
        
        if self.is_atari:
            for env_idx, env in enumerate(self.envs):
                #env = gym.wrappers.RecordEpisodeStatistics(env)
                self.envs[env_idx] = NoopResetEnv(env, noop_max=30)
                self.envs[env_idx] = MaxAndSkipEnv(self.envs[env_idx], skip=4)
                self.envs[env_idx] = EpisodicLifeEnv(self.envs[env_idx])

                if "FIRE" in self.envs[env_idx].unwrapped.get_action_meanings():
                    self.envs[env_idx] = FireResetEnv(self.envs[env_idx])
                
                self.envs[env_idx] = ClipRewardEnv(self.envs[env_idx])
                self.envs[env_idx] = gym.wrappers.ResizeObservation(self.envs[env_idx], (84, 84))
                self.envs[env_idx] = gym.wrappers.GrayScaleObservation(self.envs[env_idx])
                self.envs[env_idx] = gym.wrappers.FrameStack(self.envs[env_idx], 4)
                self.envs[env_idx].action_space.seed(config.seed)
                # env = gym.wrappers.RecordVideo(
                #     env, video_folder='./video/', episode_trigger=lambda episode_id: episode_id % 5 == 0) # only works with render_mode="human"
        
        self.config = config 
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim

        self.num_episodes = config.num_episodes
        self.initial_number_steps = config.initial_number_steps
        self.later_number_steps = config.later_number_steps
        self.max_steps = self.initial_number_steps
        self.BATCH_SIZE = config.BATCH_SIZE
        self.LR = config.LR
        self.TAU = config.TAU
        self.device = config.device
        
        # copying weights of base_net to policy_net and target_net
        if self.is_atari:
            self.policy_net = DQNNetMultgamInvAtari(self.config).to(self.device)
            self.target_net = DQNNetMultgamInvAtari(self.config).to(self.device)
        else:
            self.policy_net = DQNNetMultgamInv(self.config).to(self.device)
            self.target_net = DQNNetMultgamInv(self.config).to(self.device)
            #self.policy_net = DQNNetMultgam(self.config)
            #self.target_net = DQNNetMultgam(self.config)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # self.num_gamma = config.num_gamma 
        # self.gamma_min = config.gamma_min
        # self.gamma_max = config.gamma_max
        # start = 1 / np.log(self.gamma_min) 
        # #start = 1 / np.log(0.99) # with 1 gamma TODO
        # end = 1 / np.log(self.gamma_max)
        # #end = 1 / np.log(0.95)   
        # self.gammas = torch.exp(torch.true_divide(1, torch.linspace(start, end, self.num_gamma)))
        
        low_range = self.policy_net.low_range
        med_range = self.policy_net.med_range
        
        self.gamma_min = config.gamma_min
        self.gamma_max = config.gamma_max
        #start = 1 / np.log(self.gamma_min) 
        start = self.gamma_min
        #start = 1 / np.log(0.99) # with 1 gamma TODO
        #end = 1 / np.log(self.gamma_max)
        end = self.gamma_max
        #end = 1 / np.log(0.95)   
        #self.gammas = torch.exp(torch.true_divide(1, torch.linspace(start, end, self.num_gamma)))
        gammas_low = torch.linspace(start, low_range, self.num_gamma_per_module+1, device=self.device)
        gammas_med = torch.linspace(low_range, med_range, self.num_gamma_per_module+1, device=self.device)
        gammas_high = torch.linspace(med_range, end, self.num_gamma_per_module, device=self.device)
        
        self.gammas = torch.cat([gammas_low[:-1], gammas_med[:-1], gammas_high])

        self.num_sensitivities = config.num_sensitivities 
        self.rmin = config.rmin
        self.rmax = config.rmax
        assert self.rmin < self.rmax, "The reward range is not valid"
        self.sensitivity_step = (self.rmax - self.rmin) / self.num_sensitivities
        self.sensitivities = torch.arange(self.rmin, self.rmax, self.sensitivity_step)
        self.middle_sensitivities = torch.tensor([torch.true_divide(self.sensitivities[i] + self.sensitivities[i+1], 2) for i in range(self.sensitivities.shape[0]-1)], device=self.device)
        self.activ_sharpness = config.activ_sharpness
        
        self.num_gamma_to_tau = config.num_gamma_to_tau
        self.gamma_to_tau_min = config.gamma_to_tau_min
        self.gamma_to_tau_max = config.gamma_to_tau_max
        start = 1 / np.log(self.gamma_to_tau_min) 
        end = 1 / np.log(self.gamma_to_tau_max)   
        self.gammas_to_tau = torch.exp(torch.true_divide(1, torch.linspace(start, end, self.num_gamma_to_tau)))
        self.K = config.K
        self.time_horizon_change = config.time_horizon_change
        
        # taus=torch.linspace(0.01,3,self.num_gamma) # as in implementation from Pablo Tano
        # self.gammas=torch.exp(-1/taus)
        print("Gammas : ", self.gammas)
        
        self.total_steps = [0 for _ in range(self.num_gamma)]

        # reproducibility
        self.seed = config.seed
        set_seed(self.seed)
        
        self.dir = os.getcwd()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_memory = [ReplayMemory(self.replay_buffer_size) for _ in range(self.num_gamma)] 
        
        self.episode_durations = []
        self.check_model_improved = [torch.tensor([0]) for _ in range(self.num_gamma)]
        self.best_max = [torch.tensor([0]) for _ in range(self.num_gamma)]

        # for select action (epsilon-greedy)
        self.steps_done = [0 for _ in range(self.num_gamma)]
        
        # save for plotting evolution during training for each gamma
        self.model_reward_hist = [[] for _ in range(self.num_gamma)]
        self.model_loss_hist = [[] for _ in range(self.num_gamma)]
    
        # monitoring parameters
        # also setup config or add to config before config is passed to the model
        self.check_freq = config.check_freq
        self.log_dir = config.log_dir
        self.verbose = config.verbose
        self.best_mean_reward = -np.inf

        # instantiate the monitor for the largest gamma environment
        self.train = train
        if train:
            self.envs[-1] = Monitor(self.envs[-1], self.log_dir)


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
        self.states = [torch.from_numpy(np.array(env.reset(seed=self.seed)[0])).float().unsqueeze(0) for env in self.envs] 
            
        for i_episode in range(self.num_episodes):
            # No horizon change during training for now
            # if self.time_horizon_change is not None and i_episode == self.time_horizon_change:
            #     self.max_steps = self.later_number_steps

            for gamma_idx, gamma in enumerate(self.gammas):
                print('Episode: {} Reward: {} Max_Reward: {}'.format(i_episode, self.check_model_improved[gamma_idx].item(), self.best_max[gamma_idx].item()))
                print('-' * 64)
                self.check_model_improved[gamma_idx] = 0
                for t in count():    
                    env = self.envs[gamma_idx]
                    state = self.states[gamma_idx]
                    
                    action = self.select_action(gamma_idx)
                    # No horizon change during training for now
                    # if (self.time_horizon_change is None) or (self.time_horizon_change is not None and i_episode < self.time_horizon_change):
                    #     action = self.select_action(gamma_idx)
                    # else:
                    #     action = self.select_action_after_H_change(gamma_idx) 
                                        
                    observation, reward, terminated, truncated, _ = env.step(action.item())
                    reward = torch.tensor([reward], device=self.device)
                    done = terminated or truncated

                    # if terminated or t == self.max_steps:
                    # if terminated:
                    #     next_state = None
                    if done:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Store the transition in memory
                    self.replay_memory[gamma_idx].push(state, action, next_state, reward)

                    # Move to the next state
                    self.states[gamma_idx] = next_state
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
                    
                    self.check_model_improved[gamma_idx] += reward 
                    # if done or t == self.max_steps:
                    if done:
                        self.episode_durations.append(t + 1)
                        self.model_reward_hist[gamma_idx].append(self.check_model_improved[gamma_idx].detach().numpy())
                        self.states[gamma_idx] = torch.from_numpy(np.array(self.envs[gamma_idx].reset(seed=self.seed)[0])).float().unsqueeze(0)
                        break

                    self._on_step() 

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
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net        
        if self.is_atari:
            z = self.policy_net(state_batch, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0).repeat(state_batch.shape[0], 1)) 
        else:
            input = torch.cat((state_batch, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0).repeat(state_batch.shape[0], 1)), dim=1)  

            assert state_batch.shape[0] == input.shape[0]
            z = self.policy_net(input) 
        
        state_action_values = z[torch.arange(z.size(0)), action_batch.squeeze()]
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_output = torch.zeros(self.BATCH_SIZE, self.action_dim, self.num_sensitivities, device=self.device) # for 1 sensitivity 
        with torch.no_grad():
            
            if self.is_atari:
                next_output[non_final_mask] = self.target_net(non_final_next_states, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0).repeat(non_final_next_states.shape[0], 1)) 
            else:
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


    # for the evaluation: predict with largest gamma
    # given a state as input
    # choose with which gamma to predict the action
    def predict(self, obs, gamma_idx):
        inp_state = torch.tensor([obs])
        input = torch.cat((torch.tensor(obs), torch.tensor([self.gammas[gamma_idx]])))
        input = input.reshape(1, -1)
        z = self.policy_net(input) 
        action_values = torch.sum((z[:, :, :-1] - z[:, :, 1:]) * self.middle_sensitivities, dim=2) 
        action = action_values.max(1).indices.view(1, 1)
        return action
    
    def select_action(self, gamma_idx):
        """
           Select action before the time horizon change
        """
        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
            math.exp(-1. * self.steps_done[gamma_idx] / self.config.EPS_DECAY)
        self.steps_done[gamma_idx] += 1

        if sample > eps_threshold:
            with torch.no_grad():
                if self.is_atari:
                    z = self.policy_net(self.states[gamma_idx], torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0)) 
                else:
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
    
    def select_action_eval(self, states, gamma_idx):
        """
           Select action before the time horizon change
        """

        states = torch.tensor(states)

        with torch.no_grad():
            if len(states.shape)==1:
                states = states.reshape(1,-1)
                
            if self.is_atari:
                z = self.policy_net(states, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0)) 
            else:
                input = torch.cat((states, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0)), dim=1)
            
                assert states.shape[0] == input.shape[0]
                
                z = self.policy_net(input.to(self.device)) 
            
            action_values = torch.sum((z[:, :, :-1] - z[:, :, 1:]) * self.middle_sensitivities, dim=2) 
            
            assert action_values.shape[0] == states.shape[0]
        
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return action_values.max(1).indices.view(1, 1)


    def select_action_after_H_change(self, states, time_horizon, gamma_idx):
        """
           Select action after the time horizon change using the inverse Laplace transform
        """
        # assuming batch_size = 1, since only will be used during evaluation => 1 state at a time !
        # need to permute and separate batch dimension if needed

        states = torch.tensor(states)

        with torch.no_grad():
            if len(states.shape)==1:
                states = states.reshape(1,-1)
                
            Q_gamma = torch.zeros((1, self.action_dim, self.num_gamma_to_tau, self.num_sensitivities))
            N = self.gammas_to_tau.shape[0]
            states_exp = states.expand(N,-1)
            
            # if self.is_atari:
            #     z = self.policy_net(states, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0))
            # else:
                  # NOTE may not be necessary 
            #     input = torch.cat((states, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0)), dim=1)
            #     assert states.shape[0] == input.shape[0]
    
            if self.is_atari:
                q_out = self.policy_net(states_exp, self.gammas_to_tau.unsqueeze(-1))
            else:
                input = torch.cat((states_exp, self.gammas_to_tau.unsqueeze(-1)), dim=1) 
                q_out = self.policy_net(input.to(self.device))
                
            Q_gamma = q_out.permute(1,0,2).unsqueeze(0)
            
            tau_space = SVD_approximation_inverse_Laplace(self.config, time_horizon=time_horizon, Q_gamma=Q_gamma)
            
            # eq. 13
            gammas_pow_tau = torch.tensor([self.gammas[gamma_idx]**tau for tau in range(time_horizon)])
            gammas_pow_tau = gammas_pow_tau.to(self.device)
            
            action_values = torch.matmul(torch.matmul(tau_space.to(self.device), self.middle_sensitivities[:-1].to(self.device)), gammas_pow_tau)
            
            # print("Action Values shape after change in time horizon", action_values.shape)
                                
        return action_values.max(1).indices.view(1, 1)
      
        
    def eval_step(self, render=True):
        """
        Evaluation using the trained target network, no training involved
        :param render: whether to visualize the evaluation or not
        """
        self.max_steps = self.initial_number_steps
        
        # TODO: Makes most sense to restrict evaluation to the largest gamma
        for gamma_idx in range(self.num_gamma):
            
            for each_ep in range(self.config.evaluate_episodes):
                
                if self.time_horizon_change is not None and each_ep == self.time_horizon_change:
                   self.max_steps = self.later_number_steps
                
                state, info = self.envs[gamma_idx].reset(seed=self.seed) 
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                print('Episode: {} Reward: {} Training_Max_Reward: {}'.format(each_ep, self.check_model_improved[0].item(),
                                                                                self.best_max[0].item()))
                print('-' * 64)
                self.check_model_improved = 0

                for t in count(): 
                    if (self.time_horizon_change is None) or (self.time_horizon_change is not None and each_ep < self.time_horizon_change):
                        action = self.select_action(gamma_idx)
                    else:
                        action = self.select_action_after_H_change(gamma_idx) 
                        
                    #action = self.select_action(gamma_idx=-1)
                    
                    # input = torch.cat((state, torch.tensor([self.gammas[gamma_idx]]).unsqueeze(0).repeat(state.shape[0], 1)), dim=1) # NOTE evaluating for the ith gamma
                    # z = self.policy_net(input)
                    observation, reward, terminated, truncated, _ = self.envs[gamma_idx].step(action.item()) 
                    reward = torch.tensor([reward], device=self.device)
                    done = terminated or truncated 

                    if render:
                        self.envs[gamma_idx].render() 
                    
                    self.check_model_improved += reward
                
                    if done or t == self.max_steps:
                        break
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)  
                        state = next_state
        print('Complete')
        
        # Calculate the runtime
        runtime = self.end_time - self.start_time
        print(f'Runtime: {runtime} seconds')
    
    def _on_step(self) -> bool:
        # only log the last ie. the largest gamma
        if self.total_steps[-1] % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            print(len(x))
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.log_dir))
                    # self.model.save(self.save_path)
                    
                    self.save_model()
        return True

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self.config, file)
        
        policy_file = os.path.join(self.log_dir, 'policy_net.pth')
        torch.save(self.policy_net.state_dict(), policy_file)

        targ_file = os.path.join(self.log_dir, 'targ_net.pth')
        torch.save(self.target_net.state_dict(), targ_file)
    
    def load_model(self, path=None):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        if path:
            log_dir = path
        else:
            log_dir = self.log_dir

        # load the config
        tmp_env = gym.make(self.game, None)
        config_loaded = config_categ[self.game](tmp_env, 'Nope')
        config_path = os.path.join(log_dir, 'config.pkl')
        config_loaded = config_loaded.load_model(config_path)
        self.config = config_loaded
        # re initialize with loaded config
        print('re-initialized')
        self.re_init()

        # load policy_net
        print('policy_net loaded')
        policy_path = os.path.join(log_dir, 'policy_net.pth')
        self.policy_net.load_state_dict(torch.load(policy_path, map_location='cpu'))
        self.policy_net.eval()

        # load target_net
        print('target_net loaded')
        target_path = os.path.join(log_dir, 'targ_net.pth')
        self.target_net.load_state_dict(torch.load(target_path, map_location='cpu'))
        self.target_net.eval()
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove the non-picklable entries.
    #     del state['envs']
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # Recreate the non-picklable entries.
    #     self.envs = [gym.make(self.game, None) for _ in range(self.config.num_gamma)]
    #     for env in self.envs:
    #         env.action_space.seed(self.config.seed)
    #     self.envs[-1] = Monitor(self.envs[-1], self.log_dir)