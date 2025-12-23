"""
Deep Q-Network Agent with GPU Support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from config import (
    DEVICE, STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE,
    LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    BATCH_SIZE, MEMORY_SIZE, TARGET_UPDATE
)


class DQN(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with GPU support and target network"""
    
    def __init__(self):
        self.device = DEVICE
        print(f"[INIT] Initializing DQN Agent on: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Networks
        self.policy_net = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(self.device)
        self.target_net = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Replay buffer
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        # Exploration parameters
        self.epsilon = EPSILON_START
        
        # Training tracking
        self.training_step = 0
        self.losses = []
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        """Update network weights using a batch from replay buffer"""
        if len(self.memory) < BATCH_SIZE:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Convert to tensors and move to GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (using target network for stability)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        print(f"[SAVE] Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"[LOAD] Model loaded from {filepath}")
    
    def get_recent_loss(self):
        """Get average loss from recent training"""
        if not self.losses:
            return 0
        recent = self.losses[-100:]
        return sum(recent) / len(recent)
