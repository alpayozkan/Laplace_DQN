import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import os
from hyperparams.qrdqn_hyper import *

env_name = "CartPole-v1"
# env_name = "LunarLander-v2"
# env_name = "Acrobot-v1"
# env_name = "ALE/Seaquest-v0"


df_dir = os.path.join("logs", "qdqn", env_name,'monitor.csv')
# os.makedirs(log_dir, exist_ok=True)

df = pd.read_csv(df_dir, skiprows=1)

timesteps = hyperdict[env_name]['n_timesteps']

y = df['r'].to_numpy()
x = np.linspace(1, timesteps, len(y))

# take average of last n many episodes
n = 25
mean_kernel = np.ones(n)/n
y_conv = np.convolve(y, mean_kernel, mode='valid')

plt.plot(x[:y_conv.shape[0]], y_conv)

plt.title("qrdqn_{}".format(env_name))
plt.xlabel('Timesteps')
plt.ylabel('Average Episode Rewards')