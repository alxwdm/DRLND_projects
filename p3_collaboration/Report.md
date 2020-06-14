# DRLND_collaboration
Udacity Deep Reinforcement Learning Nanodegree - Project Collaboration and Competition

# Algorithmic Details

Agents that are interacting with each other (or with humans) is one of the most exciting real-world RL application. However, the learning process can be very difficult and unstable. That is because the environment tends to be **non-stationary** from one Agent's perspective, due to the changing policies of the other Agents. Thus, indepentently learning methods of any kind do not perform well in practice.

On the other hand, a **centralized training with de-centralized execution** that allows policies to use extra information from all Agents to ease training, seems like a promising approach. Multi-Agent Deep Deterministic Gradients (MADDPG) is such a learning approach. It is based on the actor-critic DDPG algorithm (as described [here](https://github.com/alxwdm/DRLND_projects/blob/master/p2_continuouscontrol/Report.md)), where the TD-estimate is used for the critic to learn a value function, and the deterministic policy gradient is used to update the actor. 

In the multi-agent scenario, the critic of a single agent is augmented with information from all actors and the complete (observable) state space during training. At execution time, the actors act based on their individual observation space, without "help" from the critic. This centralized critic approach helps to stabilize learning, because the enviroment appears no longer non-stationary. Further details are described in the OpenAI research paper ["Multi Agent Actor Critic for Mixed Cooperative Competitive environments"](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).

# Neural Network Architecture and Hyperparameters

**TODO**

Here is a complete list of the hyperparameter setting:
 
```
TODO
```

# Ideas for Future Work

**TODO**
