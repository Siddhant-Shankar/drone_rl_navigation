import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class Actor:
    """Policy network (Actor)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Simple neural network weights (for demonstration)
        self.w1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.w3 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)
    
    def forward(self, state):
        """Forward pass through network"""
        x = np.maximum(0, np.dot(state, self.w1) + self.b1)  # ReLU
        x = np.maximum(0, np.dot(x, self.w2) + self.b2)  # ReLU
        action = np.tanh(np.dot(x, self.w3) + self.b3)  # Tanh for bounded actions
        return action
    
    def get_action(self, state, noise=0.0):
        """Get action with optional exploration noise"""
        action = self.forward(state)
        if noise > 0:
            action += np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -1, 1)
        return action


class Critic:
    """Q-value network (Critic)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Simple neural network weights
        input_dim = state_dim + action_dim
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.w3 = np.random.randn(hidden_dim, 1) * 0.1
        self.b3 = np.zeros(1)
    
    def forward(self, state, action):
        """Forward pass through network"""
        x = np.concatenate([state, action])
        x = np.maximum(0, np.dot(x, self.w1) + self.b1)  # ReLU
        x = np.maximum(0, np.dot(x, self.w2) + self.b2)  # ReLU
        q_value = np.dot(x, self.w3) + self.b3
        return q_value


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient Agent"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self._copy_weights(self.actor, self.actor_target)
        
        # Critic networks (Twin Q-networks)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim)
        self._copy_weights(self.critic1, self.critic1_target)
        
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim)
        self._copy_weights(self.critic2, self.critic2_target)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.005  # soft update parameter
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2  # delayed policy updates
        self.batch_size = 256
        
        self.total_iterations = 0
        self.actor_loss_history = []
        self.critic_loss_history = []
    
    def _copy_weights(self, source, target):
        """Copy weights from source network to target network"""
        target.w1 = source.w1.copy()
        target.b1 = source.b1.copy()
        target.w2 = source.w2.copy()
        target.b2 = source.b2.copy()
        target.w3 = source.w3.copy()
        target.b3 = source.b3.copy()
    
    def _soft_update(self, source, target):
        """Soft update of target network"""
        target.w1 = self.tau * source.w1 + (1 - self.tau) * target.w1
        target.b1 = self.tau * source.b1 + (1 - self.tau) * target.b1
        target.w2 = self.tau * source.w2 + (1 - self.tau) * target.w2
        target.b2 = self.tau * source.b2 + (1 - self.tau) * target.b2
        target.w3 = self.tau * source.w3 + (1 - self.tau) * target.w3
        target.b3 = self.tau * source.b3 + (1 - self.tau) * target.b3
    
    def select_action(self, state, noise=0.1):
        """Select action using current policy with exploration noise"""
        return self.actor.get_action(state, noise=noise)
    
    def train(self):
        """Train the agent on a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        self.total_iterations += 1
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Simplified training (actual implementation would use gradients)
        # This is a placeholder showing the TD3 algorithm structure
        
        # Compute target Q-values
        next_actions = np.array([self.actor_target.forward(s) for s in next_states])
        # Add noise for smoothing
        noise = np.clip(np.random.normal(0, self.policy_noise, next_actions.shape),
                       -self.noise_clip, self.noise_clip)
        next_actions = np.clip(next_actions + noise, -1, 1)
        
        # Compute twin Q-values
        target_q1 = np.array([self.critic1_target.forward(s, a) for s, a in zip(next_states, next_actions)])
        target_q2 = np.array([self.critic2_target.forward(s, a) for s, a in zip(next_states, next_actions)])
        target_q = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * np.minimum(target_q1, target_q2)
        
        # Critic loss (simplified)
        current_q1 = np.array([self.critic1.forward(s, a) for s, a in zip(states, actions)])
        current_q2 = np.array([self.critic2.forward(s, a) for s, a in zip(states, actions)])
        critic_loss = np.mean((current_q1 - target_q)**2) + np.mean((current_q2 - target_q)**2)
        
        self.critic_loss_history.append(critic_loss)
        
        # Delayed policy updates
        actor_loss = 0
        if self.total_iterations % self.policy_freq == 0:
            # Actor loss (simplified)
            current_actions = np.array([self.actor.forward(s) for s in states])
            actor_q = np.array([self.critic1.forward(s, a) for s, a in zip(states, current_actions)])
            actor_loss = -np.mean(actor_q)
            
            self.actor_loss_history.append(actor_loss)
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
        
        return actor_loss, critic_loss
    
    def save_model(self, filepath):
        """Save model weights"""
        model_data = {
            'actor': {
                'w1': self.actor.w1, 'b1': self.actor.b1,
                'w2': self.actor.w2, 'b2': self.actor.b2,
                'w3': self.actor.w3, 'b3': self.actor.b3
            }
        }
        np.save(filepath, model_data)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights"""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.actor.w1 = model_data['actor']['w1']
        self.actor.b1 = model_data['actor']['b1']
        self.actor.w2 = model_data['actor']['w2']
        self.actor.b2 = model_data['actor']['b2']
        self.actor.w3 = model_data['actor']['w3']
        self.actor.b3 = model_data['actor']['b3']
        print(f"Model loaded from {filepath}")