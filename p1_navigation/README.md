# DRLND_navigation
Udacity Deep Reinforcement Learning Nanodegree - Project Navigation

# Project Details

<p align="center">
<img src="https://github.com/alxwdm/DRLND_projects/blob/master/p1_navigation/pics/banana.gif" width="500">
</p>

This project is part of the Udacity Deep Reinforcement Learning Nanodegree. The task is to train an Agent to navigate in a large, square-shaped world and collect yellow bananas while avoiding blue bananas.

The Agent has the following four discrete **actions**:
* `0` - move forward.
* `1` - move backward.
* `2`- turn left.
* `3` - turn right.

The **state** that is returned by the environment for each step the Agent takes has 37 dimensions. It contains the Agent's velocity, as well as a ray-based perception of objects around the Agent's forward direction. 

The task is episodic. For each yellow banana that the Agent collects, it recieves a **reward** of `+1`. In contrast, each collected blue banana yields a reward of `-1`. All other events return no reward and the episode finishes after a maximum number of steps has been taken.

The environment is considered **solved** when the Agent gets an average return of `+13` in 100 consecutive episodes. In the provided benchmark, the training takes about 1800 episodes.

# Implementation Details

See [Report](https://github.com/alxwdm/DRLND_projects/tree/master/p1_navigation/Report.md).

# Getting Started

Udacity has built a modified version of the Unity ML-agents environment for this "banana collector" project. It can be downloaded here:
* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows 32 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows 64 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Detailed instructions on how to setup a local python environment for this project can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). It is important to notice that the version of Unity ML-agents needs to be **0.4.0** in order to work with the provided environment.

Also, make sure to include the python folder from the [DRLND Repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/python) to import further dependencies.

# Instructions

If you want to train an agent from scratch, make sure you have correctly setup the python environment as described above and run all sections in the provided Jupyter notebook. To verify the training success of the pre-trained agent, clone the repository (including checkpoint.pth) and run section 5 of the notebook.
