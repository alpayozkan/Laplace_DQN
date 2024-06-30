from categorical import *

import gymnasium as gym
import torch

from utils import *
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)

from hyperparams.categ_hyper import *

if __name__ == '__main__':

    # Select the right environment name
    # env_name = "CartPole-v1"
    # env_name = "LunarLander-v2"
    # env_name = "Acrobot-v1"
    
    env_name = "Breakout-v0"
    # env_name = "Seaquest-v0"
    # env_name = "Defender-v0"
    # env_name = "Hero-v0"
    # env_name = "Tutankham-v0"

    # get the optimal hyperparams for the given env
    hyperparams = hyperdict_categ[env_name] 
    
    prefix = env_name.split("/")[0]
    atari_game_list = ["Breakout-v0", "Defender-v0", "Hero-v0", "Tutankham-v0", "Seaquest-v0"]
    if env_name in atari_game_list:
        is_atari = True 

    log_dir = os.path.join("logs", "categ", env_name)
    os.makedirs(log_dir, exist_ok=True)
    
    if is_atari:
        env = gym.make(env_name, render_mode="rgb_array") # for atari environments
    else:
        env = gym.make(env_name, render_mode=None)
    
    env = Monitor(env, log_dir)
    seed = 42
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
    if is_atari:
        #env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        # env = gym.wrappers.RecordVideo(
        #     env, video_folder='./video/', episode_trigger=lambda episode_id: episode_id % 5 == 0) # only works with render_mode="human"

    state_dim = env.observation_space.shape[0]
    # required for the categorical to know the action dim
    act_dim = env.action_space.n

    n_atoms = hyperparams['n_atoms'] # args.n_atoms
    
    if is_atari:
        # using same number of hidden layers and units as in the original DQN nature paper 
        z_net = networks.DistributionalDQN_network(observation_space=env.observation_space, n_actions=act_dim, n_atoms=n_atoms)
    else:
        n_units = hyperparams['n-hidden-units'] # args.n_hidden_units
        n_layers = hyperparams['n_hidden_layers'] # args.n_hidden_layers
        z_net = networks.DistributionalNetwork(inputs=state_dim, n_actions=act_dim, n_atoms=n_atoms,
                                            n_hidden_units=n_units, n_hidden_layers=n_layers)
    
    v_min, v_max = hyperparams['support_range'] # args.support_range
    start_train_at = hyperparams['start_train_at'] # args.start_train_at
    update_net_every = hyperparams['update_net_every'] # args.update_net_every
    epsilon = hyperparams['epsilon'] # args.epsilon
    n_steps = hyperparams['n_steps'] # args.n_steps
        
    DDQN = CategoricalDQN(z_net=z_net, n_atoms=n_atoms, v_min=v_min, v_max=v_max,
                          start_train_at=start_train_at,
                          update_every=update_net_every, epsilon=epsilon, log_dir=log_dir, check_freq=100 ,verbose=1)
   
    plot = DDQN.train(env=env, n_steps=n_steps)
