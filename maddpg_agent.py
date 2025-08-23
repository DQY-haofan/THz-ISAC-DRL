"""
MADDPG Multi-Agent Deep Deterministic Policy Gradient Implementation
=====================================================================

This module implements the MADDPG algorithm for multi-agent reinforcement learning
in LEO-ISAC networks. It follows the Centralized Training with Decentralized 
Execution (CTDE) paradigm, using centralized critics during training while 
maintaining decentralized actors for execution.

Based on the theoretical foundation from Report 2 and integrating GA-MADDPG
enhancements from Report 6.

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union
from copy import deepcopy
import warnings

# Import custom modules
from networks import (
    ActorNetwork, CriticNetwork, RewardDecompositionNetwork,
    GATGRUEncoder
)
from replay_buffer import PrioritizedReplayBuffer, Experience


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.
    
    Generates temporally correlated noise for continuous action spaces,
    which is more suitable for physical control problems than uncorrelated noise.
    """
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2, dt: float = 1e-2):
        """
        Initialize OU noise process.
        
        Args:
            size: Dimension of the action space
            mu: Mean reversion level
            theta: Mean reversion rate
            sigma: Volatility parameter
            dt: Time step
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        """Reset noise to mean."""
        self.state = self.mu.copy()
    
    def sample(self) -> np.ndarray:
        """Generate next noise sample."""
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state += dx
        return self.state
    
    def __call__(self) -> np.ndarray:
        """Convenience method for sampling."""
        return self.sample()


class MADDPG_Agent:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation.
    
    Implements the CTDE paradigm where agents are trained with centralized
    critics that have access to global information, but execute using only
    local observations through their decentralized actors.
    
    Key features:
    - GA-MADDPG neural architectures with GAT-GRU encoders
    - Prioritized experience replay for sample efficiency
    - Difference rewards for precise credit assignment
    - Soft target updates for training stability
    """
    
    def __init__(self,
                num_agents: int,
                obs_dim: int,
                action_dim: int,
                num_neighbors: int = 4,
                neighbor_dim: int = 10,
                actor_lr: float = 1e-4,
                critic_lr: float = 1e-3,
                reward_lr: float = 1e-3,
                tau: float = 0.005,
                gamma: float = 0.99,
                buffer_capacity: int = 1000000,
                batch_size: int = 256,
                use_prioritized_replay: bool = True,
                use_difference_rewards: bool = True,
                use_gru: bool = False,  # 添加这一行
                share_encoder: bool = True,
                total_power_budget: float = 1.0,
                per_link_max: float = 0.5,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize MADDPG agent system.
        
        Args:
            num_agents: Number of agents in the system
            obs_dim: Dimension of each agent's observation
            action_dim: Dimension of each agent's action (continuous)
            num_neighbors: Maximum number of neighbors per agent
            neighbor_dim: Dimension of neighbor features
            actor_lr: Learning rate for actor networks
            critic_lr: Learning rate for critic networks
            reward_lr: Learning rate for reward decomposition network
            tau: Soft update coefficient for target networks
            gamma: Discount factor
            buffer_capacity: Experience replay buffer size
            batch_size: Training batch size
            use_prioritized_replay: Whether to use PER
            use_difference_rewards: Whether to use learned difference rewards
            use_gru: Whether to use GRU for temporal modeling  # 添加这一行
            share_encoder: Whether actors and critics share encoders
            total_power_budget: Total power constraint for projection layer
            per_link_max: Per-link power constraint
            device: Training device (cuda/cpu)
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        self.use_difference_rewards = use_difference_rewards
        self.use_gru = use_gru  # 添加这一行
        
        # Initialize networks for each agent
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        # Shared encoder for parameter efficiency (if enabled)
        shared_encoder = None
        
        for i in range(num_agents):
            # Create actor network
            actor = ActorNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_neighbors=num_neighbors,
                neighbor_dim=neighbor_dim,
                total_power_budget=total_power_budget,
                per_link_max=per_link_max,
                use_gru=use_gru,  # 添加这一行
                use_cyclical_encoding=False  # Can be configured
            ).to(self.device)
            
            # Create critic network (potentially sharing encoder)
            critic = CriticNetwork(
                num_agents=num_agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_neighbors=num_neighbors,
                neighbor_dim=neighbor_dim,
                share_encoder=share_encoder and (shared_encoder is not None),
                actor_encoder=shared_encoder
            ).to(self.device)
            
            # Store shared encoder from first agent
            if i == 0 and share_encoder:
                shared_encoder = actor.encoder if use_gru else None  # 修改这一行
            
            # Create target networks (deep copy to avoid reference issues)
            target_actor = deepcopy(actor)
            target_critic = deepcopy(critic)
            
            # Freeze target networks
            for param in target_actor.parameters():
                param.requires_grad = False
            for param in target_critic.parameters():
                param.requires_grad = False
            
            # Create optimizers
            actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
            critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
            
            # Store all components
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        # Initialize reward decomposition network (if using difference rewards)
        if self.use_difference_rewards:
            global_state_dim = num_agents * obs_dim  # Simplified global state
            self.reward_decomposer = RewardDecompositionNetwork(
                state_dim=global_state_dim,
                obs_dim=obs_dim,
                action_dim=action_dim
            ).to(self.device)
            self.reward_optimizer = optim.Adam(
                self.reward_decomposer.parameters(), lr=reward_lr
            )
        else:
            self.reward_decomposer = None
            self.reward_optimizer = None
        
        # Initialize experience replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                alpha=0.6,
                beta_start=0.4,
                beta_end=1.0,
                beta_steps=100000
            )
        else:
            from replay_buffer import UniformReplayBuffer
            self.replay_buffer = UniformReplayBuffer(capacity=buffer_capacity)
        
        # Initialize exploration noise for each agent
        self.noise_processes = [
            OrnsteinUhlenbeckNoise(size=action_dim)
            for _ in range(num_agents)
        ]
        
        # Training statistics
        self.training_steps = 0
        self.episode_count = 0
        
    def reset_noise(self):
        """Reset exploration noise for all agents."""
        for noise in self.noise_processes:
            noise.reset()
    
    def reset_hidden_states(self, batch_size: int = 1):
        """Reset GRU hidden states for all networks."""
        for actor in self.actors:
            actor.reset_hidden(batch_size)
        for critic in self.critics:
            critic.reset_hidden(batch_size)
        for target_actor in self.target_actors:
            target_actor.reset_hidden(batch_size)
        for target_critic in self.target_critics:
            target_critic.reset_hidden(batch_size)
    
    def select_actions(self, observations: Union[List[np.ndarray], np.ndarray],
                      neighbor_observations: Optional[List[np.ndarray]] = None,
                      add_noise: bool = True,
                      noise_scale: float = 1.0) -> np.ndarray:
        """
        Select actions for all agents based on current observations.
        
        Args:
            observations: List or array of observations for each agent
            neighbor_observations: Optional neighbor observations for GAT
            add_noise: Whether to add exploration noise
            noise_scale: Scale factor for exploration noise
            
        Returns:
            Array of actions for all agents (num_agents, action_dim)
        """
        actions = []
        
        # Convert observations to tensors
        if isinstance(observations, list):
            obs_tensors = [torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                          for obs in observations]
        else:
            obs_tensors = [torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                          for i in range(self.num_agents)]
        
        # Process neighbor observations if provided
        neighbor_tensors = None
        if neighbor_observations is not None:
            neighbor_tensors = [
                torch.FloatTensor(n_obs).unsqueeze(0).to(self.device)
                if n_obs is not None else None
                for n_obs in neighbor_observations
            ]
        
        # Get action from each agent's actor
        for i in range(self.num_agents):
            self.actors[i].eval()
            with torch.no_grad():
                obs = obs_tensors[i]
                neighbor_obs = neighbor_tensors[i] if neighbor_tensors else None
                
                # Get deterministic action from actor
                action = self.actors[i](obs, neighbor_obs)
                action = action.cpu().numpy().squeeze()
                
                # Add exploration noise if in training mode
                if add_noise:
                    noise = self.noise_processes[i].sample() * noise_scale
                    action = action + noise
                    
                    # Ensure actions remain in valid range after noise
                    # The projection layer handles this, but we clip for safety
                    action = np.clip(action, 0, self.actors[i].projection.per_link_max)
                
                actions.append(action)
            
            self.actors[i].train()
        
        return np.array(actions)
    
    def store_experience(self, observations: np.ndarray,
                        actions: np.ndarray,
                        rewards: np.ndarray,
                        next_observations: np.ndarray,
                        dones: np.ndarray,
                        info: Optional[Dict] = None):
        """
        Store experience in replay buffer.
        
        Args:
            observations: Current observations for all agents
            actions: Actions taken by all agents
            rewards: Rewards received by all agents
            next_observations: Next observations for all agents
            dones: Episode termination flags
            info: Optional additional information
        """
        experience = Experience(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            info=info
        )
        self.replay_buffer.add(experience)


    def learn(self, update_actor: bool = True) -> Dict[str, float]:
        """
        Perform learning update with gradient clipping, GPU support, and proper batch size handling.
        
        This version ensures hidden states are properly reset for the batch size
        before processing experiences from the replay buffer.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        self.training_steps += 1
        metrics = {}
        
        # Sample batch from replay buffer
        if hasattr(self.replay_buffer, 'sample'):
            experiences, is_weights, indices = self.replay_buffer.sample(self.batch_size)
        else:
            raise ValueError("Replay buffer must have sample method")
        
        # Convert experiences to tensors and move to GPU
        obs_batch = torch.FloatTensor(
            np.array([e.observations for e in experiences])
        ).to(self.device)  # Shape: (batch_size, num_agents, obs_dim)
        
        actions_batch = torch.FloatTensor(
            np.array([e.actions for e in experiences])
        ).to(self.device)  # Shape: (batch_size, num_agents, action_dim)
        
        rewards_batch = torch.FloatTensor(
            np.array([e.rewards for e in experiences])
        ).to(self.device)  # Shape: (batch_size, num_agents)
        
        next_obs_batch = torch.FloatTensor(
            np.array([e.next_observations for e in experiences])
        ).to(self.device)  # Shape: (batch_size, num_agents, obs_dim)
        
        dones_batch = torch.FloatTensor(
            np.array([e.dones for e in experiences])
        ).to(self.device)  # Shape: (batch_size, num_agents)
        
        is_weights = torch.FloatTensor(is_weights).to(self.device)
        
        # Get actual batch size (might be different from self.batch_size at the end of buffer)
        actual_batch_size = obs_batch.size(0)
        
        # Reset hidden states for all networks with correct batch size
        # This is crucial for GRU-based networks
        for actor in self.actors:
            actor.reset_hidden(actual_batch_size)
        for target_actor in self.target_actors:
            target_actor.reset_hidden(actual_batch_size)
        for critic in self.critics:
            critic.reset_hidden(actual_batch_size)
        for target_critic in self.target_critics:
            target_critic.reset_hidden(actual_batch_size)
        
        # Separate observations and actions for each agent
        agent_obs = [obs_batch[:, i] for i in range(self.num_agents)]
        agent_actions = [actions_batch[:, i] for i in range(self.num_agents)]
        agent_next_obs = [next_obs_batch[:, i] for i in range(self.num_agents)]
        agent_rewards = [rewards_batch[:, i] for i in range(self.num_agents)]
        agent_dones = [dones_batch[:, i] for i in range(self.num_agents)]
        
        # Update critics with gradient clipping
        critic_losses = []
        td_errors = []
        
        for i in range(self.num_agents):
            # Calculate target actions using target actors
            target_actions = []
            for j in range(self.num_agents):
                self.target_actors[j].eval()
                with torch.no_grad():
                    # Ensure target actor has correct batch size
                    if hasattr(self.target_actors[j], 'reset_hidden'):
                        self.target_actors[j].reset_hidden(actual_batch_size)
                    target_action = self.target_actors[j](agent_next_obs[j])
                target_actions.append(target_action)
            
            # Flatten actions for critic input
            target_actions_flat = torch.cat(target_actions, dim=1)
            
            # Calculate target Q-value
            self.target_critics[i].eval()
            with torch.no_grad():
                # Ensure target critic has correct batch size
                if hasattr(self.target_critics[i], 'reset_hidden'):
                    self.target_critics[i].reset_hidden(actual_batch_size)
                target_q = self.target_critics[i](agent_next_obs, target_actions_flat)
                target_q = target_q.squeeze()
            
            # Calculate target value with reward
            if self.use_difference_rewards and self.reward_decomposer is not None:
                # Compute difference reward
                global_state = obs_batch.view(actual_batch_size, -1)
                agent_obs_i = agent_obs[i]
                agent_action_i = agent_actions[i]
                
                with torch.no_grad():
                    default_action = torch.zeros_like(agent_action_i)
                    diff_reward = self.reward_decomposer.compute_difference_reward(
                        global_state, agent_obs_i, agent_action_i, default_action
                    ).squeeze()
                
                enhanced_reward = agent_rewards[i] + 0.1 * diff_reward
            else:
                enhanced_reward = agent_rewards[i]
            
            y = enhanced_reward + self.gamma * target_q * (1 - agent_dones[i])
            
            # Calculate current Q-value
            # Ensure critic has correct batch size
            if hasattr(self.critics[i], 'reset_hidden'):
                self.critics[i].reset_hidden(actual_batch_size)
            
            actions_flat = torch.cat(agent_actions, dim=1)
            current_q = self.critics[i](agent_obs, actions_flat).squeeze()
            
            # Calculate TD-error for PER
            td_error = y - current_q
            td_errors.append(td_error.detach().cpu().numpy())
            
            # Calculate critic loss with importance sampling weights
            critic_loss = (is_weights * (td_error ** 2)).mean()
            critic_losses.append(critic_loss.item())
            
            # Update critic with gradient clipping
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            # GRADIENT CLIPPING FOR STABILITY
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), max_norm=1.0)
            self.critic_optimizers[i].step()
        
        # Update priorities in replay buffer
        if hasattr(self.replay_buffer, 'update_priorities'):
            max_td_errors = np.abs(np.max(td_errors, axis=0))
            self.replay_buffer.update_priorities(indices, max_td_errors)
        
        # Update actors with gradient clipping
        actor_losses = []
        if update_actor and self.training_steps % 2 == 0:  # Update every 2 steps
            for i in range(self.num_agents):
                # Freeze critic parameters during actor update
                for param in self.critics[i].parameters():
                    param.requires_grad = False
                
                # Reset actor hidden state for batch
                if hasattr(self.actors[i], 'reset_hidden'):
                    self.actors[i].reset_hidden(actual_batch_size)
                
                # Get actions from current actor
                actor_actions = []
                for j in range(self.num_agents):
                    if j == i:
                        # Ensure actor has correct batch size
                        if hasattr(self.actors[j], 'reset_hidden'):
                            self.actors[j].reset_hidden(actual_batch_size)
                        actor_action = self.actors[j](agent_obs[j])
                    else:
                        actor_action = agent_actions[j].detach()
                    actor_actions.append(actor_action)
                
                actions_for_critic = torch.cat(actor_actions, dim=1)
                
                # Reset critic hidden state for batch
                if hasattr(self.critics[i], 'reset_hidden'):
                    self.critics[i].reset_hidden(actual_batch_size)
                
                # Calculate actor loss
                actor_loss = -self.critics[i](agent_obs, actions_for_critic).mean()
                actor_losses.append(actor_loss.item())
                
                # Update actor with gradient clipping
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                # GRADIENT CLIPPING FOR STABILITY
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=0.5)
                self.actor_optimizers[i].step()
                
                # Unfreeze critic parameters
                for param in self.critics[i].parameters():
                    param.requires_grad = True
        
        # Update reward decomposition network if used
        if self.use_difference_rewards and self.reward_decomposer is not None:
            global_state = obs_batch.view(actual_batch_size, -1)
            
            reward_loss = 0
            for i in range(self.num_agents):
                predicted_reward = self.reward_decomposer(
                    global_state, agent_obs[i], agent_actions[i]
                ).squeeze()
                
                team_reward = rewards_batch.mean(dim=1)
                reward_loss += F.mse_loss(predicted_reward, team_reward)
            
            reward_loss = reward_loss / self.num_agents
            
            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            # Gradient clipping for reward network
            torch.nn.utils.clip_grad_norm_(self.reward_decomposer.parameters(), max_norm=0.5)
            self.reward_optimizer.step()
            
            metrics['reward_loss'] = reward_loss.item()
        
        # Soft update target networks
        self._soft_update_targets()
        
        # Clear hidden state caches to free memory (optional)
        for critic in self.critics:
            if hasattr(critic, 'clear_hidden_cache'):
                critic.clear_hidden_cache()
        for target_critic in self.target_critics:
            if hasattr(target_critic, 'clear_hidden_cache'):
                target_critic.clear_hidden_cache()
        
        # Collect metrics
        metrics.update({
            'critic_loss_mean': np.mean(critic_losses),
            'actor_loss_mean': np.mean(actor_losses) if actor_losses else 0,
            'td_error_mean': np.mean(np.abs(td_errors)),
            'training_steps': self.training_steps
        })
        
        return metrics
        
    def _soft_update_targets(self):
        """
        Perform soft update of target networks using Polyak averaging.
        
        θ' ← τ * θ + (1 - τ) * θ'
        """
        for i in range(self.num_agents):
            # Update target actor
            for target_param, param in zip(
                self.target_actors[i].parameters(),
                self.actors[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            # Update target critic
            for target_param, param in zip(
                self.target_critics[i].parameters(),
                self.critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
    
    def save_models(self, path: str):
        """
        Save all models to disk.
        
        Args:
            path: Directory path to save models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), 
                      f"{path}/actor_{i}.pth")
            torch.save(self.critics[i].state_dict(), 
                      f"{path}/critic_{i}.pth")
            torch.save(self.target_actors[i].state_dict(), 
                      f"{path}/target_actor_{i}.pth")
            torch.save(self.target_critics[i].state_dict(), 
                      f"{path}/target_critic_{i}.pth")
        
        if self.reward_decomposer is not None:
            torch.save(self.reward_decomposer.state_dict(),
                      f"{path}/reward_decomposer.pth")
        
        print(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """
        Load models from disk.
        
        Args:
            path: Directory path containing saved models
        """
        import os
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(
                torch.load(f"{path}/actor_{i}.pth", map_location=self.device)
            )
            self.critics[i].load_state_dict(
                torch.load(f"{path}/critic_{i}.pth", map_location=self.device)
            )
            self.target_actors[i].load_state_dict(
                torch.load(f"{path}/target_actor_{i}.pth", map_location=self.device)
            )
            self.target_critics[i].load_state_dict(
                torch.load(f"{path}/target_critic_{i}.pth", map_location=self.device)
            )
        
        if self.reward_decomposer is not None and \
           os.path.exists(f"{path}/reward_decomposer.pth"):
            self.reward_decomposer.load_state_dict(
                torch.load(f"{path}/reward_decomposer.pth", map_location=self.device)
            )
        
        print(f"Models loaded from {path}")
    
    def set_eval_mode(self):
        """Set all networks to evaluation mode."""
        for i in range(self.num_agents):
            self.actors[i].eval()
            self.critics[i].eval()
    
    def set_train_mode(self):
        """Set all networks to training mode."""
        for i in range(self.num_agents):
            self.actors[i].train()
            self.critics[i].train()


# ============================================================================
# Unit Tests
# ============================================================================

def test_ou_noise():
    """Test Ornstein-Uhlenbeck noise process."""
    print("Testing OU Noise Process...")
    
    noise = OrnsteinUhlenbeckNoise(size=4, sigma=0.2)
    
    # Generate samples
    samples = []
    for _ in range(100):
        samples.append(noise.sample())
    
    samples = np.array(samples)
    
    print(f"  Mean: {samples.mean(axis=0)}")
    print(f"  Std: {samples.std(axis=0)}")
    print(f"  Temporal correlation: Verified (OU process)")
    
    # Check temporal correlation
    correlation = np.corrcoef(samples[:-1].flatten(), samples[1:].flatten())[0, 1]
    print(f"  Autocorrelation: {correlation:.3f}")
    
    assert correlation > 0.5, "OU noise should have temporal correlation"
    print("✓ OU Noise test passed")


def test_maddpg_initialization():
    """Test MADDPG agent initialization."""
    print("\nTesting MADDPG Initialization...")
    
    agent = MADDPG_Agent(
        num_agents=3,
        obs_dim=20,
        action_dim=4,
        batch_size=32,
        buffer_capacity=1000
    )
    
    print(f"  Number of agents: {agent.num_agents}")
    print(f"  Actors created: {len(agent.actors)}")
    print(f"  Critics created: {len(agent.critics)}")
    print(f"  Target networks created: {len(agent.target_actors) + len(agent.target_critics)}")
    
    assert len(agent.actors) == 3, "Should have 3 actors"
    assert len(agent.critics) == 3, "Should have 3 critics"
    
    print("✓ MADDPG initialization test passed")


def test_action_selection():
    """Test action selection with and without noise."""
    print("\nTesting Action Selection...")
    
    num_agents = 2
    obs_dim = 10
    action_dim = 4
    
    agent = MADDPG_Agent(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim
    )
    
    # Create dummy observations
    observations = [np.random.randn(obs_dim) for _ in range(num_agents)]
    
    # Test without noise
    agent.reset_hidden_states()
    actions_no_noise = agent.select_actions(observations, add_noise=False)
    
    # Test with noise
    agent.reset_hidden_states()
    actions_with_noise = agent.select_actions(observations, add_noise=True)
    
    print(f"  Actions shape: {actions_no_noise.shape}")
    print(f"  Actions without noise: {actions_no_noise}")
    print(f"  Actions with noise: {actions_with_noise}")
    print(f"  Noise effect: {np.linalg.norm(actions_with_noise - actions_no_noise):.4f}")
    
    assert actions_no_noise.shape == (num_agents, action_dim)
    assert not np.array_equal(actions_no_noise, actions_with_noise), \
        "Actions should differ with noise"
    
    print("✓ Action selection test passed")


def test_experience_storage():
    """Test experience storage in replay buffer."""
    print("\nTesting Experience Storage...")
    
    agent = MADDPG_Agent(
        num_agents=2,
        obs_dim=10,
        action_dim=4,
        buffer_capacity=100
    )
    
    # Create dummy experience
    observations = np.random.randn(2, 10)
    actions = np.random.randn(2, 4)
    rewards = np.random.randn(2)
    next_observations = np.random.randn(2, 10)
    dones = np.array([False, False])
    
    # Store experience
    agent.store_experience(observations, actions, rewards, next_observations, dones)
    
    print(f"  Buffer size after storage: {len(agent.replay_buffer)}")
    
    # Store more experiences
    for _ in range(50):
        agent.store_experience(
            np.random.randn(2, 10),
            np.random.randn(2, 4),
            np.random.randn(2),
            np.random.randn(2, 10),
            np.array([False, False])
        )
    
    print(f"  Buffer size after 51 stores: {len(agent.replay_buffer)}")
    
    assert len(agent.replay_buffer) == 51
    print("✓ Experience storage test passed")


def test_learning_update():
    """Test learning update process."""
    print("\nTesting Learning Update...")
    
    agent = MADDPG_Agent(
        num_agents=2,
        obs_dim=10,
        action_dim=4,
        batch_size=16,
        buffer_capacity=100,
        use_difference_rewards=True
    )
    
    # Fill replay buffer
    for _ in range(50):
        agent.store_experience(
            np.random.randn(2, 10),
            np.random.randn(2, 4),
            np.random.randn(2),
            np.random.randn(2, 10),
            np.random.rand(2) > 0.9  # Random dones
        )
    
    # Perform learning update
    agent.reset_hidden_states(batch_size=16)
    metrics = agent.learn()
    
    print("  Learning metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")
    
    assert 'critic_loss_mean' in metrics
    assert 'td_error_mean' in metrics
    assert metrics['training_steps'] == 1
    
    print("✓ Learning update test passed")


def test_soft_update():
    """Test soft target network updates."""
    print("\nTesting Soft Target Updates...")
    
    agent = MADDPG_Agent(
        num_agents=1,
        obs_dim=10,
        action_dim=4,
        tau=0.1  # Large tau for visible change
    )
    
    # Get initial target network parameters
    initial_params = list(agent.target_actors[0].parameters())[0].data.clone()
    
    # Modify main network parameters
    for param in agent.actors[0].parameters():
        param.data += 1.0
    
    # Perform soft update
    agent._soft_update_targets()
    
    # Check target network parameters changed
    updated_params = list(agent.target_actors[0].parameters())[0].data
    
    difference = torch.norm(updated_params - initial_params).item()
    print(f"  Parameter difference after soft update: {difference:.4f}")
    
    assert difference > 0, "Target parameters should change"
    assert difference < 1, "Change should be gradual (tau=0.1)"
    
    print("✓ Soft update test passed")


def test_save_load_models():
    """Test model saving and loading."""
    print("\nTesting Model Save/Load...")
    
    import tempfile
    import shutil
    
    agent = MADDPG_Agent(
        num_agents=2,
        obs_dim=10,
        action_dim=4
    )
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save models
        agent.save_models(temp_dir)
        
        # Create new agent and load models
        new_agent = MADDPG_Agent(
            num_agents=2,
            obs_dim=10,
            action_dim=4
        )
        new_agent.load_models(temp_dir)
        
        # Compare parameters
        for i in range(2):
            original_params = list(agent.actors[i].parameters())[0].data
            loaded_params = list(new_agent.actors[i].parameters())[0].data
            
            assert torch.allclose(original_params, loaded_params), \
                f"Actor {i} parameters mismatch"
        
        print("  Models saved and loaded successfully")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
    
    print("✓ Save/Load test passed")


if __name__ == "__main__":
    """Run all unit tests."""
    print("=" * 60)
    print("MADDPG Agent Tests")
    print("=" * 60)
    
    test_ou_noise()
    test_maddpg_initialization()
    test_action_selection()
    test_experience_storage()
    test_learning_update()
    test_soft_update()
    test_save_load_models()
    
    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    # Demonstration
    print("\n" + "=" * 60)
    print("MADDPG Training Demonstration")
    print("=" * 60)
    
    # Create a simple MADDPG system
    demo_agent = MADDPG_Agent(
        num_agents=3,
        obs_dim=20,
        action_dim=4,
        batch_size=32,
        use_prioritized_replay=True,
        use_difference_rewards=True
    )
    
    print("\nSimulating training loop...")
    
    # Simulate training for a few steps
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        demo_agent.reset_noise()
        demo_agent.reset_hidden_states()
        
        # Simulate episode
        for step in range(10):
            # Generate dummy observations
            obs = [np.random.randn(20) for _ in range(3)]
            
            # Select actions
            actions = demo_agent.select_actions(obs, noise_scale=0.1)
            
            # Simulate environment step (dummy)
            next_obs = [np.random.randn(20) for _ in range(3)]
            rewards = np.random.randn(3)
            dones = np.array([False, False, False])
            
            # Store experience
            demo_agent.store_experience(
                np.array(obs), actions, rewards, next_obs, dones
            )
        
        # Learn from experiences
        if len(demo_agent.replay_buffer) >= demo_agent.batch_size:
            metrics = demo_agent.learn()
            print(f"  Critic loss: {metrics.get('critic_loss_mean', 0):.4f}")
            print(f"  TD error: {metrics.get('td_error_mean', 0):.4f}")
    
    print("\n✓ MADDPG demonstration complete!")