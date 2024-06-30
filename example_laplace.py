# -*- coding:utf-8 -*-
from config import Config
from utils.reproducibility import set_seed
from agents.DQN import DQNAgent
from agents.LaplaceDQN import LaplaceDQNAgent
from envs.four_state_mdp import Simple4MDP

import gym

# We won't separate the agents from their neural network 
# architecture 

def run_DQN_example(game):
    env = gym.make(game, render_mode="human")
    C = Config(env)
    dqn_agent = DQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)
    
    
def run_LaplaceDQN_example(game):
    env = gym.make(game, render_mode="human")
    C = Config(env)
    dqn_agent = LaplaceDQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)
    
    
def run_LaplaceDQN_4state_mdp_example():
    env = Simple4MDP()
    C = Config(env)
    dqn_agent = LaplaceDQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)


if __name__ == '__main__':
    game = 'CartPole-v1'
    
    #run_LaplaceDQN_example(game)
    run_LaplaceDQN_4state_mdp_example()
    # run_DQN_example(game)
    # run_CategoricalDQN_example(game)
    # run_QuantileDQN_example(game)
    # run_ExpectileDQN_example(game)


