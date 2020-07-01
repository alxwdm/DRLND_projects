# DRLND_collaboration
Udacity Deep Reinforcement Learning Nanodegree - Project Collaboration and Competition

# Algorithmic Details

Agents that are interacting with each other (or with humans) is one of the most exciting real-world RL application. However, the learning process can be very difficult and unstable. That is because the environment tends to be **non-stationary** from one Agent's perspective, due to the changing policies of the other Agents. Thus, indepentently learning methods of any kind do not perform well in practice.

On the other hand, a **centralized training with de-centralized execution** that allows policies to use extra information from all Agents to ease training, seems like a promising approach. Multi-Agent Deep Deterministic Gradients (MADDPG) is such a learning approach. It is based on the actor-critic DDPG algorithm (as described [here](https://github.com/alxwdm/DRLND_projects/blob/master/p2_continuouscontrol/Report.md)), where the TD-estimate is used for the critic to learn a value function, and the deterministic policy gradient is used to update the actor. 

In the multi-agent scenario, the critic of a single agent is augmented with information from all actors and the complete (observable) state space during training. At execution time, the actors act based on their individual observation space, without "help" from the critic. This centralized critic approach helps to stabilize learning, because the enviroment appears no longer non-stationary. Further details are described in the OpenAI research paper ["Multi Agent Actor Critic for Mixed Cooperative Competitive environments"](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).

The training success of the Agent can be seen in the following diagram:

<p align="center">
<img src="https://github.com/alxwdm/DRLND_projects/blob/master/p3_collaboration/pics/score.png" width="350">
</p>

# Neural Network Architecture and Hyperparameters

I have re-used the code from [project 2](https://github.com/alxwdm/DRLND_projects/tree/master/p2_continuouscontrol) in which I have already implemented a DDPG Agent. The training code was slightly modified to account for the two agents collaborating with each other.

Here is a complete list of the hyperparameter setting:
 
```
# Hyperparameter
# -- Replay Buffer ------
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
# -- Update Settings ---
TAU = 1e-3              # for soft update of target parameters
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 5           # number of learning passes
# -- Model --------------
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
# -- Noise Settings -----
NOISE_SIGMA = 0.2       # Ornstein-Uhlenbeck noise parameter
NOISE_THETA = 0.15      # Ornstein-Uhlenbeck noise parameter
EPS_START = 5.0         # initial value for epsilon
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay
```

# Ideas for Future Work

In contrast to other projects, I did not use **batch normalization** which could improve the Agent's performance. Also **prioritized experience replay** could be an interessting improvement. However, the environment could be solved without BN and PER. This is the last project of the Nanodegree. The area of Reinforcement Learning is still under heavy research, so I expect new algorithms to be released from time to time - it would be interessting to try them out with the DRLND-projects!
