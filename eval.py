from collections import OrderedDict


import gymnasium as gym

import stable_baselines3
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gymnasium.wrappers import FrameStack

# for logging and visuals
from utils import *
from stable_baselines3.common.monitor import Monitor

# import hyperparameters
from hyperparams.qrdqn_hyper import *
import os


# Select the right environment name
env_name = "CartPole-v1"
# env_name = "LunarLander-v2"
# env_name = "Acrobot-v1"
# env_name = "ALE/Seaquest-v0"

log_dir = os.path.join("logs", "qdqn", env_name)

# create the environment
env = gym.make(env_name, render_mode='human')

# load model and env wrt optimal hyperparams
model, env = get_model_env(env_name, env)


model_dir = os.path.join(log_dir, 'best_model.zip')
model = QRDQN.load(model_dir)

print('Evaluation...:')
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
      obs, _ = env.reset()
