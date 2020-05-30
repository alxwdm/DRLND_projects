# DRLND_continuouscontrol
Udacity Deep Reinforcement Learning Nanodegree - Project Continuous Control

# Project Details

<p align="center">
<img src="https://github.com/alxwdm/DRLND_projects/blob/master/p2_continuouscontrol/pics/reacher.gif" width="500">
</p>

This project is part of the Udacity Deep Reinforcement Learning Nanodegree. The task is to train an Agent (a double-jointed arm) to follow a target location.

The **action space** is continuous with four values in range `[-1, 1]`, representing the torque on each joint. The **state** that is returned by the environment for each step the Agent takes has 33 dimensions, corresponding to position, rotation, velocity, and angular velocities of the arm.

The Agent gets a **reward** of `+0.1` for each time step that its hand is at the target location. The environment is considered **solved** when the Agent gets an average return of `+30` in 100 consecutive episodes. As an alternative, 20 Agents can be **trained in parallel**. Thus, the goal is to get an average return over all agents of `+30` in 100 consecutive episodes.

# Implementation Details

See [Report](https://github.com/alxwdm/DRLND_projects/tree/master/p2_continuouscontrol/Report.md).

# Getting Started

Udacity has provided the Unity environment. The 20 Agent version can be downloaded here:
* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* [MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* [Windows 32 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* [Windows 64 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Detailed instructions on how to setup a local python environment for this project can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). It is important to notice that the version of Unity ML-agents needs to be **0.4.0** in order to work with the provided environment.

Also, make sure to include the python folder from the [DRLND Repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/python) to import further dependencies.
