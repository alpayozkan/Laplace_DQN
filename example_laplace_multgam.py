# -*- coding:utf-8 -*-
from config import config_categ

from agents.DQN import DQNAgent
from agents.LaplaceDQN_multgam_4smdp import LaplaceDQNAgentMultgam_4smdp
from agents.LaplaceDQN_multgam import LaplaceDQNAgentMultgam, LaplaceDQNAgentMultgamIntv

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)

import os

def run_DQN_example(game):
    env = gym.make(game, render_mode="human")
    
    log_dir = os.path.join("logs", "dqn", game)
    os.makedirs(log_dir, exist_ok=True)

    # we are using the same laplace config, be mindful !!
    C = config_categ[game](env, log_dir)

    dqn_agent = DQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)
    
def run_LaplaceDQN_example(game):
    atari_game_list = ["Breakout-v0", "Defender-v0", "Hero-v0", "Tutankham-v0", "Seaquest-v0"]
    if game in atari_game_list:
        is_atari = True 
    else:
        is_atari = False
        
    log_dir = os.path.join("logs", "laplace_multgamintv", game)
    os.makedirs(log_dir, exist_ok=True)
    
    if is_atari:
        env = gym.make(game, render_mode="rgb_array") # for atari environments
    else:
        env = gym.make(game, render_mode=None)
        
    seed = 42
        
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
    
    # do monitoring for the largest gamma, and the environment that corresponds to it
    # env = Monitor(env, log_dir)

    config = config_categ[game](env, log_dir)
        
    # with separate gamma networks low, medium, high
    dqn_agent = LaplaceDQNAgentMultgamIntv(game, config)

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)
    
def run_LaplaceDQN_4smdp_example(game):
    dqn_agent = LaplaceDQNAgentMultgam_4smdp(game)

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)
    
if __name__ == '__main__':
    # game = 'CartPole-v1'
    # game =  'Acrobot-v1'
    # game = "LunarLander-v2"
    game = "Breakout-v0"
    # game = "Seaquest-v0"
    # game = "Defender-v0"
    # game = "Hero-v0"
    # game = "Tutankham-v0"

    run_LaplaceDQN_example(game)

    #run_LaplaceDQN_4smdp_example(game)
    # run_DQN_example(game)

