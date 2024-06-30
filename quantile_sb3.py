from collections import OrderedDict


import gymnasium as gym

import stable_baselines3
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gymnasium.wrappers import FrameStack

# for logging and visuals
from utils import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)

# import hyperparameters
from hyperparams.qrdqn_hyper import *
import os

# Select the right environment name
# env_name = "CartPole-v1"
# env_name = "LunarLander-v2"
# env_name = "Acrobot-v1"

#env_name = "Breakout-"
env_name = "Seaquest-v0"
# env_name = "Defender-v0"
# env_name = "Hero-v0"
# env_name = "Tutankham-v0"

atari_game_list = ["Breakout-v0", "Defender-v0", "Hero-v0", "Tutankham-v0", "Seaquest-v0"]
if env_name in atari_game_list:
    is_atari = True 

log_dir = os.path.join("logs", "qdqn", env_name)
os.makedirs(log_dir, exist_ok=True)

# create the environment
if is_atari:
  env = gym.make(env_name, render_mode="rgb_array") # for atari environments
else:
  env = gym.make(env_name, render_mode=None)
  
env = Monitor(env, log_dir)
seed = 42
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
# wraping of the environment can also be done in utils.py/load_model.py
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

hyperparams = hyperdict_qrdqn[env_name]
model, env = get_model_env(env_name, env)

model.learn(total_timesteps=hyperparams['n_timesteps'], 
            log_interval=4,
            callback=callback
            )

# model.save("qrdqn_" + env_name)

del model # remove to demonstrate saving and loading

model_dir = os.path.join(log_dir, 'best_model.zip')
model = QRDQN.load(model_dir)

print('Training Done, Evaluation...:')
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
      obs, _ = env.reset()