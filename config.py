# -*- coding:utf-8 -*-
import torch 
import os
import pickle
from gymnasium.wrappers import LazyFrames
import numpy as np

# TODO add a parameter for class initialisation for the type of game 
# and automatically extract environment parameters -> for now setup 
# for CartPole environment
class Config_Cartpole:
    def __init__(self, env, log_dir):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        self.input_dim = len(state)
    
        self.num_episodes = 500
        self.evaluate_episodes = 1
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.initial_number_steps = 500 # before change in time horizon
        self.later_number_steps = 200 # after change in time horizon
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9 # 0.9 if not pre-trained weights
        self.EPS_END = 0.0
        self.EPS_DECAY = 1000
        self.TAU = 0.05
        self.LR = 1e-3

        # reproducibility 
        self.seed = 42

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each DRL Algorithm 
        # -----------------------------------------------------------------------

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 100
        self.categorical_n_atoms = 51

        # Quantile Regression DQN parameters
        self.num_quantiles = 20 
        self.huber_loss_threshold = 1
        
        # Laplace Code parameters
        self.num_gamma_per_module = 5 # orig 3
        self.gamma_min = 0.01
        self.gamma_max = 0.99 # orig 0.99
        self.num_sensitivities = 500 # 250 used in toy MP
        self.rmin = 0 # -2 # 0 for ccartpole
        self.rmax = 2 # 0 #2 for cartpole
        self.activ_sharpness = 50 # sharpness of the activation function
        # Inverse Laplace transform parameters
        self.num_gamma_to_tau = 100 # num_gamma on which the inverse Laplace transform is computed
        self.gamma_to_tau_min = 0.01
        self.gamma_to_tau_max = 0.99 
        self.K = self.later_number_steps # temporal horizon
        self.delta_t = 1  # temporal resolution
        self.alpha_reg = 0.2 # Regularization parameter for SVD-based Discrete Linear Decoder
        self.time_horizon_change = 6 # when time horizon changes during training (episode number)

        # monitoring parameters
        self.check_freq = 100
        self.log_dir = log_dir
        self.verbose = False

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj

class Config_LunarLander:
    def __init__(self, env, log_dir):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        self.input_dim = len(state)
    
        self.num_episodes = 500
        self.evaluate_episodes = 1
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.initial_number_steps = 500 # before change in time horizon
        self.later_number_steps = 200 # after change in time horizon
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9 # 0.9 if not pre-trained weights
        self.EPS_END = 0.0
        self.EPS_DECAY = 1000
        self.TAU = 0.05
        self.LR = 1e-3

        # reproducibility 
        self.seed = 42

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each DRL Algorithm 
        # -----------------------------------------------------------------------

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 100
        self.categorical_n_atoms = 51

        # Quantile Regression DQN parameters
        self.num_quantiles = 20 
        self.huber_loss_threshold = 1
        
        # Laplace Code parameters
        self.num_gamma_per_module = 5 # orig 3
        self.gamma_min = 0.01
        self.gamma_max = 0.99 # orig 0.99
        self.num_sensitivities = 500 # 250 used in toy MP
        self.rmin = -120 # -2 # 0 for ccartpole
        self.rmax = +120 # 0 #2 for cartpole
        self.activ_sharpness = 50 # sharpness of the activation function
        # Inverse Laplace transform parameters
        self.num_gamma_to_tau = 100 # num_gamma on which the inverse Laplace transform is computed
        self.gamma_to_tau_min = 0.01
        self.gamma_to_tau_max = 0.99 
        self.K = self.later_number_steps # temporal horizon
        self.delta_t = 1  # temporal resolution
        self.alpha_reg = 0.2 # Regularization parameter for SVD-based Discrete Linear Decoder
        self.time_horizon_change = 6 # when time horizon changes during training (episode number)

        # monitoring parameters
        self.check_freq = 100
        self.log_dir = log_dir
        self.verbose = False

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj

class Config_Acrobot:
    def __init__(self, env, log_dir):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        self.input_dim = len(state)
    
        self.num_episodes = 500
        self.evaluate_episodes = 1
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.initial_number_steps = 500 # before change in time horizon
        self.later_number_steps = 200 # after change in time horizon
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9 # 0.9 if not pre-trained weights
        self.EPS_END = 0.0
        self.EPS_DECAY = 1000
        self.TAU = 0.05
        self.LR = 1e-3

        # reproducibility 
        self.seed = 42

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each DRL Algorithm 
        # -----------------------------------------------------------------------

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 100
        self.categorical_n_atoms = 51

        # Quantile Regression DQN parameters
        self.num_quantiles = 20 
        self.huber_loss_threshold = 1
        
        # Laplace Code parameters
        self.num_gamma_per_module = 5 # orig 3
        self.gamma_min = 0.01
        self.gamma_max = 0.99 # orig 0.99
        self.num_sensitivities = 500 # 250 used in toy MP
        self.rmin = -2 # -2 # 0 for ccartpole
        self.rmax = 0 # 0 #2 for cartpole
        self.activ_sharpness = 50 # sharpness of the activation function
        # Inverse Laplace transform parameters
        self.num_gamma_to_tau = 100 # num_gamma on which the inverse Laplace transform is computed
        self.gamma_to_tau_min = 0.01
        self.gamma_to_tau_max = 0.99 
        self.K = self.later_number_steps # temporal horizon
        self.delta_t = 1  # temporal resolution
        self.alpha_reg = 0.2 # Regularization parameter for SVD-based Discrete Linear Decoder
        self.time_horizon_change = 6 # when time horizon changes during training (episode number)

        # monitoring parameters
        self.check_freq = 100
        self.log_dir = log_dir
        self.verbose = False

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj
    
class Config_Breakout:
    def __init__(self, env, log_dir):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        self.input_dim = env.observation_space.shape[0]
    
        self.num_episodes = 5
        self.evaluate_episodes = 1
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.initial_number_steps = 500 # before change in time horizon
        self.later_number_steps = 200 # after change in time horizon
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9 # 0.9 if not pre-trained weights
        self.EPS_END = 0.0
        self.EPS_DECAY = 1000
        self.TAU = 0.05
        self.LR = 5e-4

        # reproducibility 
        self.seed = 42

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each DRL Algorithm 
        # -----------------------------------------------------------------------

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 100
        self.categorical_n_atoms = 51

        # Quantile Regression DQN parameters
        self.num_quantiles = 20 
        self.huber_loss_threshold = 1
        
        # Laplace Code parameters
        self.num_gamma_per_module = 5 # orig 3
        self.gamma_min = 0.01
        self.gamma_max = 0.99 # orig 0.99
        self.num_sensitivities = 500 # 250 used in toy MP
        self.rmin = 0
        self.rmax = 5
        self.activ_sharpness = 50 # sharpness of the activation function
        # Inverse Laplace transform parameters
        self.num_gamma_to_tau = 100 # num_gamma on which the inverse Laplace transform is computed
        self.gamma_to_tau_min = 0.01
        self.gamma_to_tau_max = 0.99 
        self.K = self.later_number_steps # temporal horizon
        self.delta_t = 1  # temporal resolution
        self.alpha_reg = 0.2 # Regularization parameter for SVD-based Discrete Linear Decoder
        self.time_horizon_change = 6 # when time horizon changes during training (episode number)

        # monitoring parameters
        self.check_freq = 100
        self.log_dir = log_dir
        self.verbose = False

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj

class Config_Seaquest:
    def __init__(self, env, log_dir):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        self.input_dim = env.observation_space.shape[0]
    
        self.num_episodes = 500
        self.evaluate_episodes = 1
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.initial_number_steps = 500 # before change in time horizon
        self.later_number_steps = 200 # after change in time horizon
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9 # 0.9 if not pre-trained weights
        self.EPS_END = 0.0
        self.EPS_DECAY = 1000
        self.TAU = 0.05
        self.LR = 5e-4

        # reproducibility 
        self.seed = 42

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each DRL Algorithm 
        # -----------------------------------------------------------------------

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 100
        self.categorical_n_atoms = 51

        # Quantile Regression DQN parameters
        self.num_quantiles = 20 
        self.huber_loss_threshold = 1
        
        # Laplace Code parameters
        self.num_gamma_per_module = 5 # orig 3
        self.gamma_min = 0.01
        self.gamma_max = 0.99 # orig 0.99
        self.num_sensitivities = 500 # 250 used in toy MP
        self.rmin = 0 # -2 # 0 for ccartpole
        self.rmax = 10 # 0 #2 for cartpole
        self.activ_sharpness = 50 # sharpness of the activation function
        # Inverse Laplace transform parameters
        self.num_gamma_to_tau = 100 # num_gamma on which the inverse Laplace transform is computed
        self.gamma_to_tau_min = 0.01
        self.gamma_to_tau_max = 0.99 
        self.K = self.later_number_steps # temporal horizon
        self.delta_t = 1  # temporal resolution
        self.alpha_reg = 0.2 # Regularization parameter for SVD-based Discrete Linear Decoder
        self.time_horizon_change = 6 # when time horizon changes during training (episode number)

        # monitoring parameters
        self.check_freq = 100
        self.log_dir = log_dir
        self.verbose = False

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj

class Config_Defender:
    def __init__(self, env, log_dir):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        self.input_dim = env.observation_space.shape[0]
    
        self.num_episodes = 500
        self.evaluate_episodes = 1
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.initial_number_steps = 500 # before change in time horizon
        self.later_number_steps = 200 # after change in time horizon
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9 # 0.9 if not pre-trained weights
        self.EPS_END = 0.0
        self.EPS_DECAY = 1000
        self.TAU = 0.05
        self.LR = 5e-4

        # reproducibility 
        self.seed = 42

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each DRL Algorithm 
        # -----------------------------------------------------------------------

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 100
        self.categorical_n_atoms = 51

        # Quantile Regression DQN parameters
        self.num_quantiles = 20 
        self.huber_loss_threshold = 1
        
        # Laplace Code parameters
        self.num_gamma_per_module = 5 # orig 3
        self.gamma_min = 0.01
        self.gamma_max = 0.99 # orig 0.99
        self.num_sensitivities = 500 # 250 used in toy MP
        self.rmin = 0 
        self.rmax = 10
        self.activ_sharpness = 50 # sharpness of the activation function
        # Inverse Laplace transform parameters
        self.num_gamma_to_tau = 100 # num_gamma on which the inverse Laplace transform is computed
        self.gamma_to_tau_min = 0.01
        self.gamma_to_tau_max = 0.99 
        self.K = self.later_number_steps # temporal horizon
        self.delta_t = 1  # temporal resolution
        self.alpha_reg = 0.2 # Regularization parameter for SVD-based Discrete Linear Decoder
        self.time_horizon_change = 6 # when time horizon changes during training (episode number)

        # monitoring parameters
        self.check_freq = 100
        self.log_dir = log_dir
        self.verbose = False

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj

class Config_Hero:
    def __init__(self, env, log_dir):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        self.input_dim = env.observation_space.shape[0]
    
        self.num_episodes = 500
        self.evaluate_episodes = 1
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.initial_number_steps = 500 # before change in time horizon
        self.later_number_steps = 200 # after change in time horizon
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9 # 0.9 if not pre-trained weights
        self.EPS_END = 0.0
        self.EPS_DECAY = 1000
        self.TAU = 0.05
        self.LR = 5e-4

        # reproducibility 
        self.seed = 42

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each DRL Algorithm 
        # -----------------------------------------------------------------------

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 100
        self.categorical_n_atoms = 51

        # Quantile Regression DQN parameters
        self.num_quantiles = 20 
        self.huber_loss_threshold = 1
        
        # Laplace Code parameters
        self.num_gamma_per_module = 5 # orig 3
        self.gamma_min = 0.01
        self.gamma_max = 0.99 # orig 0.99
        self.num_sensitivities = 500 # 250 used in toy MP
        self.rmin = 0 # -2 # 0 for ccartpole
        self.rmax = 10 # 0 #2 for cartpole
        self.activ_sharpness = 50 # sharpness of the activation function
        # Inverse Laplace transform parameters
        self.num_gamma_to_tau = 100 # num_gamma on which the inverse Laplace transform is computed
        self.gamma_to_tau_min = 0.01
        self.gamma_to_tau_max = 0.99 
        self.K = self.later_number_steps # temporal horizon
        self.delta_t = 1  # temporal resolution
        self.alpha_reg = 0.2 # Regularization parameter for SVD-based Discrete Linear Decoder
        self.time_horizon_change = 6 # when time horizon changes during training (episode number)

        # monitoring parameters
        self.check_freq = 100
        self.log_dir = log_dir
        self.verbose = False

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj

class Config_Tutankham:
    def __init__(self, env, log_dir):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        self.input_dim = env.observation_space.shape[0]
    
        self.num_episodes = 500
        self.evaluate_episodes = 1
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.initial_number_steps = 500 # before change in time horizon
        self.later_number_steps = 200 # after change in time horizon
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9 # 0.9 if not pre-trained weights
        self.EPS_END = 0.0
        self.EPS_DECAY = 1000
        self.TAU = 0.05
        self.LR = 5e-4

        # reproducibility 
        self.seed = 42

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each DRL Algorithm 
        # -----------------------------------------------------------------------

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 100
        self.categorical_n_atoms = 51

        # Quantile Regression DQN parameters
        self.num_quantiles = 20 
        self.huber_loss_threshold = 1
        
        # Laplace Code parameters
        self.num_gamma_per_module = 5 # orig 3
        self.gamma_min = 0.01
        self.gamma_max = 0.99 # orig 0.99
        self.num_sensitivities = 500 # 250 used in toy MP
        self.rmin = 0 
        self.rmax = 10 
        self.activ_sharpness = 50 # sharpness of the activation function
        # Inverse Laplace transform parameters
        self.num_gamma_to_tau = 100 # num_gamma on which the inverse Laplace transform is computed
        self.gamma_to_tau_min = 0.01
        self.gamma_to_tau_max = 0.99 
        self.K = self.later_number_steps # temporal horizon
        self.delta_t = 1  # temporal resolution
        self.alpha_reg = 0.2 # Regularization parameter for SVD-based Discrete Linear Decoder
        self.time_horizon_change = 6 # when time horizon changes during training (episode number)

        # monitoring parameters
        self.check_freq = 100
        self.log_dir = log_dir
        self.verbose = False

    def save_model(self):
        # Save the object to a file
        conf_file = os.path.join(self.log_dir, 'config.pkl')
        with open(conf_file, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(cls, filename):
        """
            Load an instance from a file.
            Usage: loaded_obj = MyObject.load('saved_object.pkl')
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj

        
config_categ ={
    "CartPole-v1": Config_Cartpole,
    "LunarLander-v2": Config_LunarLander,
    "Acrobot-v1": Config_Acrobot,
    "Breakout-v0": Config_Breakout,
    "Seaquest-v0": Config_Seaquest,
    "Defender-v0": Config_Defender,
    "Hero-v0": Config_Hero,
    "Tutankham-v0": Config_Tutankham, 
}