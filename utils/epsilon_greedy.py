# -*- coding:utf-8 -*-
import random
import math
import torch

# Better to create an abstract class of Agents with this method
def select_action(agent, state):
    
    sample = random.random()
    eps_threshold = agent.config.EPS_END + (agent.config.EPS_START - agent.config.EPS_END) * \
        math.exp(-1. * agent.steps_done / agent.config.EPS_DECAY)
    agent.steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return agent.policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[agent.env.action_space.sample()]], device=agent.device, dtype=torch.long)
