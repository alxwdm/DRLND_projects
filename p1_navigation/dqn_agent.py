import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from replay_buffer import PrioritizedReplayBuffer, DequeReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
ALPHA = 0.6             # PER alpha rate
INIT_BETA = 0.4         # PER initial beta rate
BETA_INC = 0.0001       # PER beta increment per step
MIN_PRIO = 1e-6         # PER minimum priority for experience
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
USE_DDQN = True         # whether to use Double-DQN 
USE_PER = True          # whether to use prioritized experience replay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, layer_spec, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.layer_spec = layer_spec
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, layer_spec).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, layer_spec).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # (Prioritized) experience replay setup
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.min_prio = MIN_PRIO
        self.alpha = ALPHA
        self.beta = INIT_BETA
        self.beta_increment = BETA_INC
        if USE_PER:
            self.memory = PrioritizedReplayBuffer(size=self.buffer_size, alpha=self.alpha)
        else:
            self.memory = DequeReplayBuffer(action_size=self.action_size, 
                                            buffer_size=self.buffer_size,
                                            batch_size=self.batch_size, seed=42)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # print info about Agent
        print('Units in the hidden layers are {}.'.format(str(layer_spec)))
        print('Using Double-DQN is \"{}\".'.format(str(USE_DDQN)))
        print('Using prioritized experience replay is \"{}\".'.format(str(USE_PER)))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get subset and learn
            if len(self.memory) > BATCH_SIZE:
                self.beta = min(1., self.beta + self.beta_increment)
                experiences = self.memory.sample(self.batch_size, beta=self.beta)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Get TD step from experiences
        states, actions, rewards, next_states, dones, weights, idxes = experiences
          
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # DOUBLE DQN: Select action based on _local, evaluate action based on _target
        if USE_DDQN:
            Q_action_select = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_action_select)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
 
        # Compute (PER-weighted) MSE loss
        if USE_PER:
            TD_error = Q_targets - Q_expected
            weighted_TD_error = weights*(TD_error**2)
            loss = torch.mean(weighted_TD_error)
            # Update priorities in Replay Buffer
            prio_updates = np.abs(TD_error.detach().squeeze(1).cpu().numpy()) + self.min_prio
            self.memory.update_priorities(idxes, prio_updates.tolist())
        else:
            loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft-update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)          
        
    def save_checkpoint(self):
        checkpoint = {'input_size': self.state_size,
                      'output_size': self.action_size,
                      'layer_spec': self.layer_spec,
                      'state_dict': self.qnetwork_local.state_dict()}
        torch.save(checkpoint, 'checkpoint.pth')
        print('Checkpoint succesfully saved.')
        
    def load_checkpoint(self, filepath='checkpoint.pth'):
        checkpoint = torch.load(filepath)
        self.qnetwork_local = QNetwork(
            checkpoint['input_size'], 
            checkpoint['output_size'], 
            checkpoint['layer_spec']).to(device)
        self.qnetwork_local.load_state_dict(checkpoint['state_dict'])
        print('Checkpoint successfully loaded.')

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
