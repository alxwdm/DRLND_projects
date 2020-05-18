# DRLND_navigation
Udacity Deep Reinforcement Learning Nanodegree - Project Navigation

# Project Details

**TODO ADD PICTURE**

This project is part of the Udacity Deep Reinforcement Learning Nanodegree. The task is to train an Agent to navigate in a large, square-shaped world and collect yellow bananas while avoiding blue bananas.

The Agent has the following four discrete **actions**:
* `0` - move forward.
* `1` - move backward.
* `2`- turn left.
* `3` - turn right.

The **state** that is returned by the environment for each step the Agent takes has 37 dimensions. It contains the Agent's velocity, as well as a ray-based perception of objects around the Agent's forward direction. 

The task is episodic. For each yellow banana that the Agent collects, it recieves a **reward** of `+1`. In contrast, each collected blue banana yields a reward of `-1`. All other events return no reward and the episode finishes after a maximum number of steps has been taken.

The environment is considered **solved** when the Agent gets an average return of `+13` in 100 consecutive episodes. In the provided benchmark, the training takes about 1800 episodes.

# Algorithmic Details

I will shortly discuss the algorithmic details. For a thorough understanding I recommend to read the referenced research paper or take the Udacity course.

In order to solve the task, the Agent needs to estimate the state-action value function `Q(s, a)` and choose the next action `a` when in state `s` according to a policy that maximizes the expected total reward `Q`. A deep neural network is used as a function approximator for `Q(s, a)`, which is called Deep Q-Network (DQN). The details of the DQN algorithm can be found in this [research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). There, it is described why and how tweaks such as fixed Q-targets and experience replay are necessary when training deep neural nets for RL tasks.

Further improvements to the vanilla DQN algorithm have been discovered, such as:
* Using two seperate networks to calculate the TD-target, called a [Double-DQN](https://arxiv.org/abs/1509.06461).
* Sampling the experiences with a distribution based on the TD-error, called [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).
* Changing the network architecture to a double-stream network that estimates the state value and action advantage seperately, called a [Dueling DQN](https://arxiv.org/abs/1511.06581).
* Other, more recent advances in DQNs are incorporated in the so called [Rainbow Net](https://arxiv.org/abs/1710.02298).

In this project, I have implemented a Double-DQN with Prioritized Experience Replay.

# Getting Started

**TODO**

# Instructions

**TODO**
