import stable_baselines3
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gymnasium.wrappers import FrameStack

from hyperparams.qrdqn_hyper import *

def normalize_env(env):
    env = DummyVecEnv([lambda: env])  # Wrap the environment
    env = VecNormalize(env)           # Normalize observations and rewards   
    return env

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def get_model_env(env_name, env):
    # this only works for quantile => which uses sbl3 pipeline
    hyperparams = hyperdict_qrdqn[env_name]

    if env_name == 'CartPole-v1':
        if hyperparams['normalize']:
            env = normalize_env(env)

        model = QRDQN(
                    hyperparams['policy'], 
                    env,
                    batch_size=hyperparams['batch_size'],
                    buffer_size=hyperparams['buffer_size'],
                    exploration_final_eps=hyperparams['exploration_final_eps'],
                    exploration_fraction=hyperparams['exploration_fraction'],
                    gamma=hyperparams['gamma'],
                    gradient_steps=hyperparams['gradient_steps'],
                    learning_rate=hyperparams['learning_rate'],
                    learning_starts=hyperparams['learning_starts'],
                    policy_kwargs=eval(hyperparams['policy_kwargs']),
                    target_update_interval=hyperparams['target_update_interval'],
                    train_freq=hyperparams['train_freq'],
                    verbose=1
                    )
        
    elif env_name == 'LunarLander-v2':
        if hyperparams['normalize']:
            env = normalize_env(env)

        lr = eval(hyperparams['learning_rate'].split('_')[1])

        model = QRDQN(
                    hyperparams['policy'], 
                    env,
                    batch_size=hyperparams['batch_size'],
                    buffer_size=hyperparams['buffer_size'],
                    exploration_final_eps=hyperparams['exploration_final_eps'],
                    exploration_fraction=hyperparams['exploration_fraction'],
                    gamma=hyperparams['gamma'],
                    gradient_steps=hyperparams['gradient_steps'],
                    learning_rate=linear_schedule(lr),
                    learning_starts=hyperparams['learning_starts'],
                    policy_kwargs=eval(hyperparams['policy_kwargs']),
                    target_update_interval=hyperparams['target_update_interval'],
                    train_freq=hyperparams['train_freq'],
                    verbose=1
                    )
        
    elif env_name == 'Acrobot-v1':
        if hyperparams['normalize']:
            env = normalize_env(env)

        model = QRDQN(
                    hyperparams['policy'], 
                    env,
                    batch_size=hyperparams['batch_size'],
                    buffer_size=hyperparams['buffer_size'],
                    exploration_final_eps=hyperparams['exploration_final_eps'],
                    exploration_fraction=hyperparams['exploration_fraction'],
                    gamma=hyperparams['gamma'],
                    gradient_steps=hyperparams['gradient_steps'],
                    learning_rate=hyperparams['learning_rate'],
                    learning_starts=hyperparams['learning_starts'],
                    policy_kwargs=eval(hyperparams['policy_kwargs']),
                    target_update_interval=hyperparams['target_update_interval'],
                    train_freq=hyperparams['train_freq'],
                    verbose=1
                    )
        
    elif env_name == 'Seaquest-v0':
        #env_wrap = FrameStack(env, num_stack=hyperparams['frame_stack'])
        #wrapped_env = stable_baselines3.common.atari_wrappers.AtariWrapper(env)
        if hyperparams['normalize']:
            env = normalize_env(env)

        model = QRDQN(
                    hyperparams['policy'], 
                    env=env,
                    # optimize_memory_usage=hyperparams['optimize_memory_usage'],
                    batch_size=hyperparams['batch_size'],
                    buffer_size=hyperparams['buffer_size'],
                    exploration_final_eps=hyperparams['exploration_final_eps'],
                    exploration_fraction=hyperparams['exploration_fraction'],
                    gamma=hyperparams['gamma'],
                    gradient_steps=hyperparams['gradient_steps'],
                    learning_rate=hyperparams['learning_rate'],
                    learning_starts=hyperparams['learning_starts'],
                    target_update_interval=hyperparams['target_update_interval'],
                    train_freq=hyperparams['train_freq'],
                    verbose=1
                    )
        
    elif env_name == 'Defender-v0':
    #env_wrap = FrameStack(env, num_stack=hyperparams['frame_stack'])
    #wrapped_env = stable_baselines3.common.atari_wrappers.AtariWrapper(env)
        if hyperparams['normalize']:
            env = normalize_env(env)

        model = QRDQN(
                    hyperparams['policy'], 
                    env=env,
                    # optimize_memory_usage=hyperparams['optimize_memory_usage'],
                    batch_size=hyperparams['batch_size'],
                    buffer_size=hyperparams['buffer_size'],
                    exploration_final_eps=hyperparams['exploration_final_eps'],
                    exploration_fraction=hyperparams['exploration_fraction'],
                    gamma=hyperparams['gamma'],
                    gradient_steps=hyperparams['gradient_steps'],
                    learning_rate=hyperparams['learning_rate'],
                    learning_starts=hyperparams['learning_starts'],
                    target_update_interval=hyperparams['target_update_interval'],
                    train_freq=hyperparams['train_freq'],
                    verbose=1
                    )
    
    elif env_name == 'Hero-v0':
    #env_wrap = FrameStack(env, num_stack=hyperparams['frame_stack'])
    #wrapped_env = stable_baselines3.common.atari_wrappers.AtariWrapper(env)
        if hyperparams['normalize']:
            env = normalize_env(env)

        model = QRDQN(
                    hyperparams['policy'], 
                    env=env,
                    # optimize_memory_usage=hyperparams['optimize_memory_usage'],
                    batch_size=hyperparams['batch_size'],
                    buffer_size=hyperparams['buffer_size'],
                    exploration_final_eps=hyperparams['exploration_final_eps'],
                    exploration_fraction=hyperparams['exploration_fraction'],
                    gamma=hyperparams['gamma'],
                    gradient_steps=hyperparams['gradient_steps'],
                    learning_rate=hyperparams['learning_rate'],
                    learning_starts=hyperparams['learning_starts'],
                    target_update_interval=hyperparams['target_update_interval'],
                    train_freq=hyperparams['train_freq'],
                    verbose=1
                    )
        
    elif env_name == 'Tutankham-v0':
    #env_wrap = FrameStack(env, num_stack=hyperparams['frame_stack'])
    #wrapped_env = stable_baselines3.common.atari_wrappers.AtariWrapper(env)
        if hyperparams['normalize']:
            env = normalize_env(env)

        model = QRDQN(
                    hyperparams['policy'], 
                    env=env,
                    # optimize_memory_usage=hyperparams['optimize_memory_usage'],
                    batch_size=hyperparams['batch_size'],
                    buffer_size=hyperparams['buffer_size'],
                    exploration_final_eps=hyperparams['exploration_final_eps'],
                    exploration_fraction=hyperparams['exploration_fraction'],
                    gamma=hyperparams['gamma'],
                    gradient_steps=hyperparams['gradient_steps'],
                    learning_rate=hyperparams['learning_rate'],
                    learning_starts=hyperparams['learning_starts'],
                    target_update_interval=hyperparams['target_update_interval'],
                    train_freq=hyperparams['train_freq'],
                    verbose=1
                    )
    
    elif env_name == 'Breakout-v0':
    #env_wrap = FrameStack(env, num_stack=hyperparams['frame_stack'])
    #wrapped_env = stable_baselines3.common.atari_wrappers.AtariWrapper(env)
        if hyperparams['normalize']:
            env = normalize_env(env)

        model = QRDQN(
                    hyperparams['policy'], 
                    env=env,
                    # optimize_memory_usage=hyperparams['optimize_memory_usage'],
                    batch_size=hyperparams['batch_size'],
                    buffer_size=hyperparams['buffer_size'],
                    exploration_final_eps=hyperparams['exploration_final_eps'],
                    exploration_fraction=hyperparams['exploration_fraction'],
                    gamma=hyperparams['gamma'],
                    gradient_steps=hyperparams['gradient_steps'],
                    learning_rate=hyperparams['learning_rate'],
                    learning_starts=hyperparams['learning_starts'],
                    target_update_interval=hyperparams['target_update_interval'],
                    train_freq=hyperparams['train_freq'],
                    verbose=1
                    )
        
    else:
        raise ValueError("Environment Setting Not Supported")
    
    return model, env