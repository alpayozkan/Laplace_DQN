from collections import OrderedDict


hyperparams_cartpole = \
OrderedDict([('batch_size', 64),
             ('buffer_size', 100000),
             ('exploration_final_eps', 0.04),
             ('exploration_fraction', 0.16),
             ('gamma', 0.99),
             ('gradient_steps', 128),
             ('learning_rate', 0.0023),
             ('learning_starts', 1000),
             ('n_timesteps', 50000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs', 'dict(net_arch=[256, 256], n_quantiles=10)'),
             ('target_update_interval', 10),
             ('train_freq', 256),
             ('normalize', False)
             ])


# LunarLander-v2
hyperparams_lunarlander = \
OrderedDict([('batch_size', 128),
             ('buffer_size', 100000),
             ('exploration_final_eps', 0.18),
             ('exploration_fraction', 0.24),
             ('gamma', 0.995),
             ('gradient_steps', -1),
             ('learning_rate', 'lin_1.5e-3'),
             ('learning_starts', 10000),
             ('n_timesteps', 100000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs', 'dict(net_arch=[256, 256], n_quantiles=170)'),
             ('target_update_interval', 1),
             ('train_freq', 256),
             ('normalize', False)])


# acrobot-v1
hyperparams_acrobot = \
OrderedDict([('batch_size', 128),
             ('buffer_size', 50000),
             ('exploration_final_eps', 0.1),
             ('exploration_fraction', 0.12),
             ('gamma', 0.99),
             ('gradient_steps', -1),
             ('learning_rate', 0.00063),
             ('learning_starts', 0),
             ('n_timesteps', 100000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs', 'dict(net_arch=[256, 256], n_quantiles=25)'),
             ('target_update_interval', 250),
             ('train_freq', 4),
             ('normalize', False)])

### Atari games

hyperparams_breakout = \
OrderedDict([('env_wrapper',
              ['stable_baselines3.common.atari_wrappers.AtariWrapper']),
             ('batch_size', 128),
             ('buffer_size', 50000),
             ('exploration_final_eps', 0.1),
             ('exploration_fraction', 0.025),
             ('gamma', 0.99),
             ('gradient_steps', -1),
             ('learning_rate', 0.00063),
             ('learning_starts', 0),
             ('frame_stack', 4),
             ('n_timesteps', 10000000.0),
             ('optimize_memory_usage', True),
             ('policy', 'CnnPolicy'),
             ('target_update_interval', 250),
             ('train_freq', 4),
             ('normalize', False)])


hyperparams_seaquest = \
OrderedDict([('env_wrapper',
              ['stable_baselines3.common.atari_wrappers.AtariWrapper']),
             ('batch_size', 128),
             ('buffer_size', 50000),
             ('exploration_final_eps', 0.1),
             ('exploration_fraction', 0.025),
             ('gamma', 0.99),
             ('gradient_steps', -1),
             ('learning_rate', 0.00063),
             ('learning_starts', 0),
             ('frame_stack', 4),
             ('n_timesteps', 10000000.0),
             ('optimize_memory_usage', True),
             ('policy', 'CnnPolicy'),
             ('target_update_interval', 250),
             ('train_freq', 4),
             ('normalize', False)])

hyperparams_defender = \
OrderedDict([('env_wrapper',
              ['stable_baselines3.common.atari_wrappers.AtariWrapper']),
             ('batch_size', 128),
             ('buffer_size', 50000),
             ('exploration_final_eps', 0.1),
             ('exploration_fraction', 0.025),
             ('gamma', 0.99),
             ('gradient_steps', -1),
             ('learning_rate', 0.00063),
             ('learning_starts', 0),
             ('frame_stack', 4),
             ('n_timesteps', 10000000.0),
             ('optimize_memory_usage', True),
             ('policy', 'CnnPolicy'),
             ('target_update_interval', 250),
             ('train_freq', 4),
             ('normalize', False)])

hyperparams_hero = \
OrderedDict([('env_wrapper',
              ['stable_baselines3.common.atari_wrappers.AtariWrapper']),
             ('batch_size', 128),
             ('buffer_size', 50000),
             ('exploration_final_eps', 0.1),
             ('exploration_fraction', 0.025),
             ('gamma', 0.99),
             ('gradient_steps', -1),
             ('learning_rate', 0.00063),
             ('learning_starts', 0),
             ('frame_stack', 4),
             ('n_timesteps', 10000000.0),
             ('optimize_memory_usage', True),
             ('policy', 'CnnPolicy'),
             ('target_update_interval', 250),
             ('train_freq', 4),
             ('normalize', False)])

hyperparams_tutankham = \
OrderedDict([('env_wrapper',
              ['stable_baselines3.common.atari_wrappers.AtariWrapper']),
             ('batch_size', 128),
             ('buffer_size', 50000),
             ('exploration_final_eps', 0.1),
             ('exploration_fraction', 0.025),
             ('gamma', 0.99),
             ('gradient_steps', -1),
             ('learning_rate', 0.00063),
             ('learning_starts', 0),
             ('frame_stack', 4),
             ('n_timesteps', 10000000.0),
             ('optimize_memory_usage', True),
             ('policy', 'CnnPolicy'),
             ('target_update_interval', 250),
             ('train_freq', 4),
             ('normalize', False)])


hyperdict_qrdqn ={
    "CartPole-v1": hyperparams_cartpole,
    "LunarLander-v2": hyperparams_lunarlander,
    "Acrobot-v1": hyperparams_acrobot,
    "Breakout-v0": hyperparams_breakout,
    "Seaquest-v0": hyperparams_seaquest,
    "Defender-v0": hyperparams_defender,
    "Hero-v0": hyperparams_hero,
    "Tutankham-v0": hyperparams_tutankham,
}