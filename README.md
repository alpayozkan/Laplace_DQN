# Laplace_DQN
Laplace DQN: Deep RL for Temporally Flexible Planning


This paper explores a Distributional Reinforcement Learning (DRL) algorithm specifically the Laplace Code, previously introduced to learn the temporal evolution of immediate rewards through a biologically plausible algorithm. In order to further scale the use of this algorithm beyond tabular settings we implemented the Laplace Code with Deep Q-Networks (DQN) and compared its performance to popular DRL algorithms like C51 and Quantile Regression DQN (QR-DQN). Importantly, the distributions learnt by the Laplace Code enable to immediately adapt the agent’s policy to be optimal for a smaller time horizon. To this end an Inverse Laplace approximation is applied to the learnt distribution. By experimenting with this transformation we uncovered the artifacts it generates and proposed methods to overcome these. With this work, we come closer to using the power of the Laplace representation in temporally dynamic real-world environments.

[Paper](https://github.com/alpayozkan/Laplace_DQN/blob/main/30_Laplace_DQN_Deep_RL_Algorit.pdf)
