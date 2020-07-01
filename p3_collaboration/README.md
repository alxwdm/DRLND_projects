# DRLND_collaboration
Udacity Deep Reinforcement Learning Nanodegree - Project Collaboration and Competition

# Project Details

<p align="center">
<img src="https://github.com/alxwdm/DRLND_projects/blob/master/p3_collaboration/pics/tennis.gif" width="500">
</p>

This project is part of the Udacity Deep Reinforcement Learning Nanodegree. The task is to train two Agents to play tennis with each other.

The **action space** is continuous with two dimensions per Agent in a range `[-1, 1]` that correspond to forward/backward and jump movements. The **observation state** consists of eight variables corresponding to the position and velocity of the ball and racket. Three observation frames are stacked, resulting in a total state space of 24 dimensions.

Each time an Agent bounces the ball over the net, it recieves a positive **reward** of `+0.1`. When it drops the ball, its reward is `-0.01`. Thus, the goal of each Agent is to keep the ball in play for as long as possible.

The environment in considered **solved** when the average score over 100 consecutive episodes is greater or equal to `+0.5`. After each episode, the Agent with the highest score is taken into account. 

# Implementation Details

See [Report](https://github.com/alxwdm/DRLND_projects/tree/master/p3_collaboration/Report.md).

# Getting Started

Udacity has provided a Unity environment, based on [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). It can be downloaded here:

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* [MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Detailed instructions on how to setup a local python environment for this project can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). It is important to notice that the version of Unity ML-agents needs to be **0.4.0** in order to work with the provided environment.

Also, make sure to include the python folder from the [DRLND Repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/python) to import further dependencies.

# How to run the code

Please refer to the [Jupyter Notebook](https://github.com/alxwdm/DRLND_projects/blob/master/p3_collaboration/Tennis.ipynb) to see how to train the Agent. You need to run every cell from section 1, 2 and 4 in order to train the Agent. Section 1 and 2 are only to prepare the environment, section 4 contains the main training code.
