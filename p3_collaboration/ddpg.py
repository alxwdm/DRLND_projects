# DDPG Agent for RL with continuous action spaces
# >> See https://arxiv.org/abs/1509.02971
#
# Modified Implementation from Udacity Deep Reinforcement Learning Nanodegree
# >> See https://github.com/udacity/deep-reinforcement-learning/
# 
# Has also been used for project 2
# >> See https://github.com/alxwdm/DRLND_projects/
#
# Modified for use with 2 Agents (MADDPG)
#

import numpy as np
import random
import copy

from model import Actor, Critic
from utils import OUNoise, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """ Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # for MADDPG
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed)
        self.eps = EPS_START
        self.eps_decay = 1/(EPS_EP_END*LEARN_NUM)
        self.timestep = 0

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    

    def step(self, state, action, reward, next_state, done, agent_number):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.timestep += 1
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory and at learning interval settings
        if len(self.memory) > BATCH_SIZE and self.timestep % LEARN_EVERY == 0:
                for _ in range(LEARN_NUM):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA, agent_number)

                    
    def act(self, states, add_noise):
        """Returns actions for both agents as per current policy, given their respective states."""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            # For MADDPG: get action for each agent and concatenate them
            for agent_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_num, :] = action
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        actions = np.clip(actions, -1, 1)
        return actions

    
    def reset(self):
        self.noise.reset()

        
    def learn(self, experiences, gamma, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        # Construct next actions vector relative to the agent
        if agent_number == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
        # Compute Q targets for current states (y_i)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # For MADDPG: Construct action vector for each agent
        actions_pred = self.actor_local(states)
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # update noise decay parameter
        #self.eps -= self.eps_decay
        #self.eps = max(self.eps, EPS_FINAL)
        #self.noise.reset()

        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
    def save_checkpoint(self, agent_number, filename='checkpoint'):
        checkpoint = {'action_size': self.action_size,
                      'state_size': self.state_size,
                      'actor_state_dict': self.actor_local.state_dict(),
                      'critic_state_dict': self.critic_local.state_dict()}
        filepath = filename + '_' + str(agent_number) + '.pth'
        torch.save(checkpoint, filepath)
        print(filepath + ' succesfully saved.')
        
        
    def load_checkpoint(self, agent_number, filename='checkpoint'):
        filepath = filename + '_' + str(agent_number) + '.pth'
        checkpoint = torch.load(filepath)
        state_size = checkpoint['state_size']
        action_size = checkpoint['action_size']
        self.actor_local = Actor(state_size, action_size, seed=42).to(device)
        self.critic_local = Critic(state_size, action_size, seed=42).to(device)
        self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
        print(filepath + ' successfully loaded.')  

            
            