import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PRIORITIZED_ER_e = 5e-2 # added to td for the calculus of the probability
PRIORITIZED_ER_b = 0.6
PRIORITIZED_ER_a = 0.4

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        fc1_units = 64*4
        fc2_units = 64*42
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        td = self.td(state, action, reward, next_state, done) + PRIORITIZED_ER_e
        self.memory.add(state, action, reward, next_state, done, td)
        
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
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
        states, actions, rewards, next_states, dones, probs = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        # Double DQN
        local_argmax_actions = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
        next_qvalues = self.qnetwork_target(next_states).gather(1,local_argmax_actions).detach()
        
        # Single DQN
        #next_qvalues = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        Q_targets = rewards + (gamma * next_qvalues * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_targets = Q_expected + (Q_targets - Q_expected)*(1/(len(self.memory)*torch.Tensor(probs)))**PRIORITIZED_ER_b 

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def td(self, state, action, reward, next_state, done):
        with torch.no_grad():
            action_values = self.qnetwork_local(torch.Tensor(state))
            state = (torch.from_numpy(state)).float().to(device)
            next_state = (torch.from_numpy(next_state)).float().to(device)
            q_target_next = self.qnetwork_target.forward(next_state).detach().max(0)[0]
            Q_target = reward + GAMMA*(q_target_next[0]) * (1-done)
            Q_expected = self.qnetwork_local.forward(state)[action]
            td = float(Q_target) - float(Q_expected)
        
        return abs(td)
    
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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "td"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, td):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, td)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probs = np.array([abs(e.td) for e in self.memory if e is not None], dtype=np.float)
        probs = probs**PRIORITIZED_ER_a / sum(probs**PRIORITIZED_ER_a)
        #probs = np.ones(len(self.memory))/len(self.memory)
        chosen_indexes = random.choices(range(len(self.memory)), k=self.batch_size, weights=probs)
        experiences = [self.memory[i] for i in chosen_indexes]
        probs = probs[chosen_indexes]
        experiences = random.sample(self.memory, k=self.batch_size)
        
        

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, probs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)