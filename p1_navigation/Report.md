# DRLND_navigation
Udacity Deep Reinforcement Learning Nanodegree - Project Navigation

# Algorithmic Details

I will very shortly discuss the algorithmic details of the project. For a thorough understanding I recommend to read the referenced research paper - or even take the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

In order to solve the task, the Agent needs to estimate the state-action value function `Q(s, a)` and choose the next action `a` when in state `s` according to a policy that maximizes the expected total reward `r + Qmax(s, a)`. A deep neural network is used as a function approximator for `Q(s, a)`, which is called Deep Q-Network (DQN). The details of the DQN algorithm can be found in this [research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). There, it is described why tweaks such as fixed Q-targets and experience replay are necessary when training deep neural nets for RL tasks and how to implement them.

Further improvements to the vanilla DQN algorithm have been discovered, such as:
* Using two separate networks to calculate the TD-target, called a [Double-DQN](https://arxiv.org/abs/1509.06461).
* Sampling the replay experiences with a distribution based on the TD-error, called [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (PER).
* Changing the network architecture to a double-stream network that estimates the state value and action advantage separately, called a [Dueling DQN](https://arxiv.org/abs/1511.06581).
* Other, more recent advances in DQNs are incorporated in the so called [Rainbow Net](https://arxiv.org/abs/1710.02298).

In this project, I have implemented a Double-DQN with Prioritized Experience Replay. The task is solved after about 500 episodes. Here is the corresponding learning curve:

<p align="center">
<img src="https://github.com/alxwdm/DRLND_projects/blob/master/p1_navigation/pics/score.png" width="350">
</p>

The architecture and weights of a trained agent can be found in the `checkpoint.pth` file in this repository. With the `agent.load_checkpoint()` method, the Q-Network can be loaded to verify the training success. Here are the scores of a trained agent:

<p align="center">
<img src="https://github.com/alxwdm/DRLND_projects/blob/master/p1_navigation/pics/trained.png" width="500">
</p>

# Neural Network Architecture and Hyperparameters

The DQN model architecture consists of three fully-connected hidden layers of size `[128, 128, 128]` with ReLU activation.

Here is a list of the additional Hyperparameters that I have chosen:

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
ALPHA = 0.6             # PER alpha rate
INIT_BETA = 0.4         # PER initial beta rate
BETA_INC = 1e-4         # PER beta increment per step
MIN_PRIO = 1e-6         # PER minimum priority for experience
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
USE_DDQN = True         # whether to use Double-DQN 
USE_PER = True          # whether to use prioritized experience replay
```

# Ideas for Future Work

This task can be solved easily, even with a vanilla DQN. The improvements that I have implemented so far (Double-DQN and PER) do not necessarily lead to faster learning. In future work, it would be interesting to see whether a [Dueling DQN](https://arxiv.org/abs/1511.06581) or a [Rainbow Net](https://arxiv.org/abs/1710.02298) are able to increase the learning speed.
