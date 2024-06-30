from collections import OrderedDict

hyperparams_cartpole = \
OrderedDict([('batch_size', 64),
             ('n_atoms', 51),
             ('n-hidden-units', 64),
             ('n_hidden_layers', 2),
             ('support_range', [0, 2]),
             ('start_train_at', 32),
             ('update_net_every', 5),
             ('epsilon', 0.1),
             ('n_steps', 50000), # TODO changed from 20000 - check if this solves the weird plot issue
             \
             # NOTE Note used for now - would need to modify categorical_dqn file 
             # if we want to use 
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
             ('n_atoms', 51),
             ('n-hidden-units', 32),
             ('n_hidden_layers', 4),
             ('support_range', [-100, 300]),
             ('start_train_at', 10000), 
             ('update_net_every', 1),
             ('epsilon', 0.18),
             ('n_steps', 150000),
             \
             # NOTE Note used for now - would need to modify categorical_dqn file 
             # if we want to use 
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


hyperparams_acrobot = \
OrderedDict([('batch_size', 128),
             ('n_atoms', 51), # adjust
             ('n-hidden-units', 64),
             ('n_hidden_layers', 2),
             ('support_range', [0, 100]), # adjust
             ('start_train_at', 0),
             ('update_net_every', 5),
             ('epsilon', 0.1),
             ('n_steps', 100000),
             \
             # NOTE Note used for now - would need to modify categorical_dqn file 
             # if we want to use 
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
OrderedDict([('batch_size', 128),
             ('n_atoms', 51),
             ('support_range', [0, 10]), 
             ('start_train_at', 1000), 
             ('update_net_every', 5),
             ('epsilon', 0.18),
             ('n_steps', 150000)])

hyperparams_seaquest = \
OrderedDict([('batch_size', 128),
             ('n_atoms', 51),
             ('support_range', [0, 10]),
             ('start_train_at', 10000), 
             ('update_net_every', 5),
             ('epsilon', 0.18),
             ('n_steps', 150000)])

hyperparams_defender = \
OrderedDict([('batch_size', 128),
             ('n_atoms', 51),
             ('support_range', [0, 10]),
             ('start_train_at', 10000), 
             ('update_net_every', 5),
             ('epsilon', 0.18),
             ('n_steps', 150000)])

hyperparams_hero = \
OrderedDict([('batch_size', 128),
             ('n_atoms', 51),
             ('support_range', [0, 10]),
             ('start_train_at', 10000), 
             ('update_net_every', 5),
             ('epsilon', 0.18),
             ('n_steps', 150000)])

hyperparams_tutankham = \
OrderedDict([('batch_size', 128),
             ('n_atoms', 51),
             ('support_range', [0, 10]),
             ('start_train_at', 1000), 
             ('update_net_every', 5),
             ('epsilon', 0.18),
             ('n_steps', 150000)])


hyperdict_categ ={
    "CartPole-v1": hyperparams_cartpole,
    "LunarLander-v2": hyperparams_lunarlander,
    "Acrobot-v1": hyperparams_acrobot,
    "Breakout-v0": hyperparams_breakout,
    "Seaquest-v0": hyperparams_seaquest,
    "Defender-v0": hyperparams_defender,
    "Hero-v0": hyperparams_hero,
    "Tutankham-v0": hyperparams_tutankham,
}