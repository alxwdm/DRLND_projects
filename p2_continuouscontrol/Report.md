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
* The sampled policy-gradient is used to train the actor, where the loss of the actor is the state-value calculated by the critic with the actions taken by the actor: `actor_loss = -critic_local(states, actions_pred).mean()`. The intuition behind this loss is that the actor takes a gradient ascent step towards higher expected return.
* More experiences are collected and the fixed target networks are (soft-)updated. The exploration is controlled by adding noise to the actor with a so called Ornstein-Uhlenbeck process.

I decided to train the version with **20 agents in parallel**. The following figure shows the training progress. It can be seen that after around 40 episodes, the average episode score over all 20 agents reaches the target value for the first time. After 60 episodes, even the worst agent's score within an episode (bottom of shaded area) is above the target. The moving average of 100 episodes reaches the target after 107 episodes, thus the environment is considered to be "solved in 7 episodes" (as defined by Udacity).

<p align="center">
<img src="https://github.com/alxwdm/DRLND_projects/blob/master/p2_continuouscontrol/pics/score.png" width="350">
</p>

# Neural Network Architecture and Hyperparameters

Both the actor and the critic use a neural network consisting two hidden layers with the same number of units and batch normalization applied after the first hidden layer. The actor's architecture is straight forward, with a `tanh` activation function on the output layer to map the outputs to the action space range `[-1, 1]`. The critic needs to output a state-value for a given action. Thus it receives the actions from the actor in its forward pass and concatenates them after the first hidden layer. 

One particular challenge of this project (and of RL in general) is keeping the learning stable while still being data-efficient. Both can be achieved by tuning the learning frequency or update-ratio, respectively. The learning is stable yet reasonably fast when performing 10 learning steps at every 20 timesteps.

Here is a complete list of the hyperparameter setting:
 
```
# -- Replay Buffer ------
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
# -- Update Settings ---
TAU = 1e-3              # for soft update of target parameters
LEARN_EVERY = 20        # learn every x timesteps
LEARN_STEP_N = 10       # learn x samples for every learning step
# -- Model --------------
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
FC_UNITS_1 = 128        # Units of first hidden layer (both actor and critic)
FC_UNITS_2 = 128        # Units of second hidden layer (both actor and critic)
# -- Noise Settings -----
EPSILON = 1.0           # noise factor 
EPSILON_DECAY = 0.999   # noise decay rate 
NOISE_SIGMA = 0.1       # sigma parameter for Ornstein-Uhlenbeck noise
NOISE_THETA = 0.15      # theta parameter for Ornstein-Uhlenbeck noise
```

# Ideas for Future Work

As the DDPG algorithm is used in Udacity's benchmark implementation, it was the most promising actor-critic agent to start with for solving the environment. It would be interesting to see whether other algorithms such as [A3C](https://arxiv.org/pdf/1602.01783.pdf) show similar performance. Also, future work could include more recent algorithms like [D4PO](https://arxiv.org/pdf/1804.08617.pdf). In addition, this DDPG implementation may be improved by using prioritized experience replay ([PER](https://arxiv.org/pdf/1511.05952.pdf)), i.e. sampling "important" experiences from the replay buffer more frequently.
