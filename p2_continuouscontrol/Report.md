# DRLND_continuouscontrol
Udacity Deep Reinforcement Learning Nanodegree - Project Continuous Control

# Algorithmic Details

In contrast to the [first project of this Nanodegree](https://github.com/alxwdm/DRLND_projects/tree/master/p1_navigation/), the action space in this environment is continuous. Value-based RL approaches, such as [Deep Q-Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), only tend to work well when the action space is discrete. Thus, another class of learning algorithms is required.

Policy-based RL approaches try to learn the policy directly, i.e. without learning a Q-value-function `Q(s,a)` first and then deriving the optimal action from the Q-values of a given state `Qmax(s,a)` with a greedy policy. In contrast to value-based learning, it is easy to apply policy-based learning to continuous action spaces. For example, the action space of this environment is a range of values `[-1, 1]` for each joint of the agent. When using neural networks as function approximators, a `tanh` activation function of the output layer directly maps an input state to an output action.  

It turns out that even better results can be achieved when combining both value- and policy-based learning. This lead to the development of so called **actor-critic methods**, where the actor learns how to act (i.e. policy-based), and a critic learns how to estimate the current situation (i.e. value-based). The algorithm that I have implemented in this project is [DDPG](https://arxiv.org/abs/1509.02971). It is an off-policy approach that uses a lot of tricks from DQN, such as a replay buffer and fixed Q-targets. Other commonly used actor-critic algorithms such as A3C and A2C are on-policy and replace the replay buffer with parallel training. 

DDPG works as follows: 
* After initializing the critic `Q(s,a|th_c)` and the actor `mu(s,a|th_a)`, the agent collects experience tuples `(s, a, r, s')` from interacting with the environment. 
* When enough samples are available, a mini-batch of experiences is drawn from the replay buffer. 
* The TD-estimate `r + gamma * Q(s',a|th_c)` is used to train the critic (using fixed Q-targets).
* The sampled policy-gradient is used to train the actor, with the advantage (calculated by the critic using the actions of the actor) `r + gamma * (V(s'|th_c) - V(s|th_c))` being the baseline.
* More experiences are collected and the fixed target networks are (soft-)updated. The exploration is controlled by adding noise to the actor with a so called Ornstein-Uhlenbeck process.

# Neural Network Architecture and Hyperparameters

**TODO**

```
...
```

# Ideas for Future Work

**TODO**
