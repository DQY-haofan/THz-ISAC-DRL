"""
Prioritized Experience Replay Buffer for GA-MADDPG
====================================================

This module implements a Prioritized Experience Replay (PER) buffer that samples
experiences based on their TD-error priorities. This focuses learning on the most
informative transitions, significantly improving sample efficiency and convergence
speed in multi-agent reinforcement learning.

Based on Chapter 7 of the GA-MADDPG framework, implementing the PER mechanism
with importance sampling weights to correct for non-uniform sampling bias.

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional, Any, NamedTuple
from collections import deque
import warnings


class Experience(NamedTuple):
    """
    Container for a single transition experience.
    
    Attributes:
        observations: Joint observations from all agents
        actions: Joint actions taken by all agents
        rewards: Rewards received by all agents
        next_observations: Next joint observations
        dones: Episode termination flags for all agents
        info: Additional information (optional)
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    info: Optional[Dict] = None


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    A binary tree where parent nodes store the sum of their children's values.
    This enables O(log n) sampling based on priorities and O(log n) priority updates.
    
    The tree is stored as a flat array where:
    - Leaf nodes (storing priorities) are at indices [capacity-1, 2*capacity-1)
    - Parent nodes are at indices [0, capacity-1)
    - For node i: left child is 2*i+1, right child is 2*i+2
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the sum tree.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        
        # Tree array: [parent_nodes | leaf_nodes]
        self.tree = np.zeros(self.tree_size, dtype=np.float32)
        
        # Data array to store actual experiences
        self.data = np.zeros(capacity, dtype=object)
        
        # Current write position
        self.write_idx = 0
        self.n_entries = 0
        
    def _propagate(self, idx: int, change: float):
        """
        Propagate priority change up the tree.
        
        Args:
            idx: Leaf index where change occurred
            change: Change in priority value
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Find the leaf index for a given cumulative sum.
        
        Args:
            idx: Current node index (start from root=0)
            s: Cumulative sum to search for
            
        Returns:
            Leaf index corresponding to the sampled priority
        """
        left_child = 2 * idx + 1
        right_child = left_child + 1
        
        # If we're at a leaf node
        if left_child >= self.tree_size:
            return idx
        
        # Navigate down the tree based on cumulative sum
        left_sum = self.tree[left_child]
        
        if s <= left_sum:
            return self._retrieve(left_child, s)
        else:
            return self._retrieve(right_child, s - left_sum)
    
    def total(self) -> float:
        """Get total sum of all priorities (root value)."""
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """
        Add new experience with given priority.
        
        Args:
            priority: Priority value (typically TD-error)
            data: Experience data to store
        """
        # Find leaf index for current write position
        tree_idx = self.write_idx + self.capacity - 1
        
        # Store data
        self.data[self.write_idx] = data
        
        # Update tree with new priority
        self.update(tree_idx, priority)
        
        # Move write position
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, tree_idx: int, priority: float):
        """
        Update priority of existing experience.
        
        Args:
            tree_idx: Tree index of the experience
            priority: New priority value
        """
        # Calculate change in priority
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up the tree
        self._propagate(tree_idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Sample an experience based on cumulative sum.
        
        Args:
            s: Cumulative sum value for sampling
            
        Returns:
            Tuple of (tree_index, priority, data)
        """
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        
        return tree_idx, self.tree[tree_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for Multi-Agent Reinforcement Learning.
    
    Implements the PER mechanism from "Prioritized Experience Replay" (Schaul et al., 2016)
    adapted for multi-agent settings. Samples experiences proportional to their TD-error
    magnitudes and applies importance sampling weights to correct for bias.
    
    Key features:
    - Efficient O(log n) sampling using sum tree
    - Automatic priority initialization for new experiences
    - Importance sampling weight computation
    - Support for priority annealing schedules
    """
    
    def __init__(self, capacity: int = 1000000,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_end: float = 1.0,
                 beta_steps: int = 100000,
                 epsilon: float = 1e-6):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (α=0 is uniform sampling)
            beta_start: Initial importance sampling exponent
            beta_end: Final importance sampling exponent
            beta_steps: Number of steps to anneal beta
            epsilon: Small constant to prevent zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        
        # Initialize sum tree for priorities
        self.tree = SumTree(capacity)
        
        # Track maximum priority for new experiences
        self.max_priority = 1.0
        
        # Beta annealing schedule
        self.beta_current = beta_start
        self.beta_step_size = (beta_end - beta_start) / beta_steps
        self.steps = 0
        
        # Statistics
        self.total_added = 0
        self.total_sampled = 0
        
    def add(self, experience: Experience):
        """
        Add new experience to the buffer with maximum priority.
        
        New experiences are given the current maximum priority to ensure
        they are sampled at least once before their priority is adjusted
        based on their actual TD-error.
        
        Args:
            experience: Experience tuple to add
        """
        # Give new experience maximum priority
        priority = self.max_priority ** self.alpha
        
        # Add to sum tree
        self.tree.add(priority, experience)
        
        self.total_added += 1
        
    def add_batch(self, observations: np.ndarray,
                  actions: np.ndarray,
                  rewards: np.ndarray,
                  next_observations: np.ndarray,
                  dones: np.ndarray,
                  info: Optional[Dict] = None):
        """
        Add a batch of experiences (convenience method).
        
        Args:
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            next_observations: Batch of next observations
            dones: Batch of done flags
            info: Optional additional information
        """
        batch_size = observations.shape[0]
        
        for i in range(batch_size):
            exp = Experience(
                observations=observations[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_observations[i],
                dones=dones[i],
                info=info[i] if info else None
            )
            self.add(exp)
    
    def sample(self, batch_size: int, 
               beta: Optional[float] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on priorities.
        
        Implements prioritized sampling where probability of sampling experience i is:
        P(i) = p_i^α / Σp_k^α
        
        Also computes importance sampling weights to correct for bias:
        w_i = (1/(N*P(i)))^β
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (uses current beta if None)
            
        Returns:
            Tuple of:
            - List of sampled experiences
            - Array of importance sampling weights (normalized)
            - Array of tree indices for priority updates
        """
        if self.tree.n_entries == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Use current beta if not specified
        if beta is None:
            beta = self._get_beta()
        
        experiences = []
        weights = np.zeros(batch_size, dtype=np.float32)
        indices = np.zeros(batch_size, dtype=np.int32)
        
        # Compute priority segment for stratified sampling
        priority_segment = self.tree.total() / batch_size
        
        # Compute max weight for normalization
        min_prob = self._get_min_probability()
        max_weight = (min_prob * self.tree.n_entries) ** (-beta)
        
        for i in range(batch_size):
            # Stratified sampling: sample uniformly from each segment
            segment_start = priority_segment * i
            segment_end = priority_segment * (i + 1)
            
            # Sample cumulative sum uniformly from segment
            cumsum = np.random.uniform(segment_start, segment_end)
            
            # Get experience from sum tree
            tree_idx, priority, experience = self.tree.get(cumsum)
            
            # Calculate sampling probability
            sampling_prob = priority / self.tree.total()
            
            # Calculate importance sampling weight
            weight = (sampling_prob * self.tree.n_entries) ** (-beta)
            
            # Normalize by max weight
            weights[i] = weight / max_weight
            
            experiences.append(experience)
            indices[i] = tree_idx
        
        self.total_sampled += batch_size
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors from training.
        
        Priority is set to: p_i = |TD_error_i| + ε
        
        Args:
            indices: Tree indices of experiences to update
            td_errors: TD-errors computed during training
        """
        # Ensure inputs are numpy arrays
        indices = np.asarray(indices)
        td_errors = np.asarray(td_errors)
        
        for idx, td_error in zip(indices, td_errors):
            # Compute new priority
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            
            # Update in sum tree
            self.tree.update(idx, priority)
            
            # Track maximum priority
            self.max_priority = max(self.max_priority, np.abs(td_error) + self.epsilon)
    
    def _get_beta(self) -> float:
        """
        Get current beta value with annealing.
        
        Beta is annealed from beta_start to beta_end over beta_steps
        to gradually reduce importance sampling correction bias.
        
        Returns:
            Current beta value
        """
        # Update beta with annealing
        self.beta_current = min(
            self.beta_end,
            self.beta_start + self.steps * self.beta_step_size
        )
        self.steps += 1
        
        return self.beta_current
    
    def _get_min_probability(self) -> float:
        """
        Get minimum possible sampling probability.
        
        Used for computing maximum importance sampling weight.
        
        Returns:
            Minimum sampling probability
        """
        min_priority = self.epsilon ** self.alpha
        return min_priority / self.tree.total() if self.tree.total() > 0 else 1.0
    
    def __len__(self) -> int:
        """Get current number of experiences in buffer."""
        return self.tree.n_entries
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self) >= batch_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics for monitoring.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'size': len(self),
            'capacity': self.capacity,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'max_priority': self.max_priority,
            'current_beta': self.beta_current,
            'alpha': self.alpha,
            'tree_total': self.tree.total()
        }
    
    def clear(self):
        """Clear all experiences from the buffer."""
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.beta_current = self.beta_start
        self.steps = 0
        self.total_added = 0
        self.total_sampled = 0


class UniformReplayBuffer:
    """
    Standard uniform replay buffer for comparison baseline.
    
    Samples experiences uniformly at random without prioritization.
    Provided for comparison with prioritized replay.
    """
    
    def __init__(self, capacity: int = 1000000):
        """
        Initialize uniform replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.total_added = 0
        self.total_sampled = 0
    
    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
        self.total_added += 1
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, None]:
        """
        Sample batch uniformly at random.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, uniform weights, None)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples: {len(self.buffer)} < {batch_size}")
        
        experiences = random.sample(self.buffer, batch_size)
        
        # Uniform weights (all equal to 1.0)
        weights = np.ones(batch_size, dtype=np.float32)
        
        self.total_sampled += batch_size
        
        return experiences, weights, None
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self) >= batch_size
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.total_added = 0
        self.total_sampled = 0


# ============================================================================
# Unit Tests
# ============================================================================

def test_sum_tree():
    """Test sum tree data structure."""
    print("Testing Sum Tree...")
    
    capacity = 8
    tree = SumTree(capacity)
    
    # Add some priorities
    priorities = [3.0, 1.0, 4.0, 2.0, 5.0]
    for i, p in enumerate(priorities):
        tree.add(p, f"data_{i}")
    
    print(f"  Added {len(priorities)} items")
    print(f"  Total sum: {tree.total():.2f}")
    
    # Test sampling
    sampled_indices = []
    for _ in range(100):
        s = np.random.uniform(0, tree.total())
        idx, priority, data = tree.get(s)
        sampled_indices.append(int(data.split('_')[1]))
    
    # Check sampling distribution (should be proportional to priorities)
    unique, counts = np.unique(sampled_indices, return_counts=True)
    print(f"  Sampling distribution: {dict(zip(unique, counts))}")
    
    # Test update
    tree.update(capacity - 1, 10.0)  # Update first item's priority
    print(f"  Total after update: {tree.total():.2f}")
    
    assert abs(tree.total() - (10.0 + sum(priorities[1:]))) < 1e-6
    print("✓ Sum Tree test passed")


def test_prioritized_replay_buffer():
    """Test prioritized replay buffer."""
    print("\nTesting Prioritized Replay Buffer...")
    
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_start=0.4)
    
    # Add some experiences
    for i in range(50):
        exp = Experience(
            observations=np.array([i]),
            actions=np.array([i]),
            rewards=np.array([i]),
            next_observations=np.array([i+1]),
            dones=np.array([False])
        )
        buffer.add(exp)
    
    print(f"  Buffer size: {len(buffer)}")
    
    # Sample a batch
    batch_size = 10
    experiences, weights, indices = buffer.sample(batch_size)
    
    print(f"  Sampled {len(experiences)} experiences")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Weights normalized: {np.allclose(weights.max(), 1.0)}")
    
    # Update priorities with random TD errors
    td_errors = np.random.randn(batch_size)
    buffer.update_priorities(indices, td_errors)
    
    print(f"  Updated priorities with TD errors")
    print(f"  Max priority: {buffer.max_priority:.4f}")
    
    # Test beta annealing
    initial_beta = buffer.beta_current
    for _ in range(100):
        buffer._get_beta()
    
    print(f"  Beta annealing: {initial_beta:.4f} -> {buffer.beta_current:.4f}")
    
    assert len(buffer) == 50
    assert weights.max() <= 1.0 + 1e-6
    assert buffer.beta_current > initial_beta
    
    print("✓ Prioritized Replay Buffer test passed")


def test_stratified_sampling():
    """Test that stratified sampling covers the priority range."""
    print("\nTesting Stratified Sampling...")
    
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
    
    # Add experiences with distinct priorities
    for i in range(20):
        exp = Experience(
            observations=np.array([i]),
            actions=np.array([i]),
            rewards=np.array([i]),
            next_observations=np.array([i+1]),
            dones=np.array([False])
        )
        buffer.add(exp)
    
    # Set different priorities
    for i in range(20):
        tree_idx = i + buffer.tree.capacity - 1
        priority = (i + 1) ** 1.0  # Linear priorities
        buffer.tree.update(tree_idx, priority)
    
    # Sample and check coverage
    batch_size = 10
    sampled_obs = []
    
    for _ in range(10):
        experiences, _, _ = buffer.sample(batch_size)
        sampled_obs.extend([exp.observations[0] for exp in experiences])
    
    unique_samples = len(set(sampled_obs))
    print(f"  Unique samples from 20 items: {unique_samples}")
    print(f"  High priority items sampled more: {sampled_obs.count(19) > sampled_obs.count(0)}")
    
    assert unique_samples >= 10  # Should sample diverse experiences
    print("✓ Stratified Sampling test passed")


def test_importance_sampling_weights():
    """Test importance sampling weight computation."""
    print("\nTesting Importance Sampling Weights...")
    
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_start=0.4)
    
    # Add experiences
    for i in range(50):
        exp = Experience(
            observations=np.array([i]),
            actions=np.array([i]),
            rewards=np.array([0.0]),
            next_observations=np.array([i+1]),
            dones=np.array([False])
        )
        buffer.add(exp)
    
    # Set varied priorities
    for i in range(50):
        tree_idx = i + buffer.tree.capacity - 1
        priority = np.random.exponential(1.0)  # Exponential distribution
        buffer.tree.update(tree_idx, priority)
    
    # Sample with different beta values
    beta_values = [0.0, 0.5, 1.0]
    
    for beta in beta_values:
        _, weights, _ = buffer.sample(20, beta=beta)
        
        print(f"  Beta={beta:.1f}: weight_std={weights.std():.4f}, "
              f"weight_range=[{weights.min():.4f}, {weights.max():.4f}]")
        
        if beta == 0.0:
            # No IS correction, all weights should be 1
            assert np.allclose(weights, 1.0), "Beta=0 should give uniform weights"
        else:
            # Higher beta should give more varied weights
            assert weights.std() > 0, "Weights should vary with beta > 0"
    
    print("✓ Importance Sampling Weights test passed")


def test_comparison_uniform_vs_prioritized():
    """Compare uniform and prioritized replay buffers."""
    print("\nComparing Uniform vs Prioritized Buffers...")
    
    uniform_buffer = UniformReplayBuffer(capacity=1000)
    prioritized_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # Add same experiences to both
    n_experiences = 100
    important_indices = [10, 30, 50, 70, 90]  # Indices of "important" experiences
    
    for i in range(n_experiences):
        reward = 10.0 if i in important_indices else 1.0
        exp = Experience(
            observations=np.array([i]),
            actions=np.array([i]),
            rewards=np.array([reward]),
            next_observations=np.array([i+1]),
            dones=np.array([False])
        )
        uniform_buffer.add(exp)
        prioritized_buffer.add(exp)
    
    # Set higher priorities for important experiences
    for i in range(n_experiences):
        tree_idx = i + prioritized_buffer.tree.capacity - 1
        priority = 10.0 if i in important_indices else 1.0
        prioritized_buffer.tree.update(tree_idx, priority)
    
    # Sample from both and count important experiences
    n_samples = 1000
    batch_size = 10
    
    uniform_important_count = 0
    prioritized_important_count = 0
    
    for _ in range(n_samples // batch_size):
        # Uniform sampling
        uniform_batch, _, _ = uniform_buffer.sample(batch_size)
        for exp in uniform_batch:
            if exp.observations[0] in important_indices:
                uniform_important_count += 1
        
        # Prioritized sampling
        prioritized_batch, _, _ = prioritized_buffer.sample(batch_size)
        for exp in prioritized_batch:
            if exp.observations[0] in important_indices:
                prioritized_important_count += 1
    
    uniform_ratio = uniform_important_count / n_samples
    prioritized_ratio = prioritized_important_count / n_samples
    
    print(f"  Important experience sampling rate:")
    print(f"    Uniform: {uniform_ratio:.3f}")
    print(f"    Prioritized: {prioritized_ratio:.3f}")
    print(f"    Improvement: {prioritized_ratio/uniform_ratio:.2f}x")
    
    assert prioritized_ratio > uniform_ratio * 1.5, \
        "Prioritized should sample important experiences more frequently"
    
    print("✓ Buffer comparison test passed")


if __name__ == "__main__":
    """Run all unit tests."""
    print("=" * 60)
    print("Prioritized Experience Replay Buffer Tests")
    print("=" * 60)
    
    test_sum_tree()
    test_prioritized_replay_buffer()
    test_stratified_sampling()
    test_importance_sampling_weights()
    test_comparison_uniform_vs_prioritized()
    
    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    # Performance demonstration
    print("\n" + "=" * 60)
    print("Performance Demonstration")
    print("=" * 60)
    
    import time
    
    buffer = PrioritizedReplayBuffer(capacity=100000)
    
    # Benchmark adding experiences
    start = time.time()
    for i in range(10000):
        exp = Experience(
            observations=np.random.randn(20),
            actions=np.random.randn(4),
            rewards=np.random.randn(1),
            next_observations=np.random.randn(20),
            dones=np.array([False])
        )
        buffer.add(exp)
    add_time = time.time() - start
    
    print(f"\nAdding 10,000 experiences: {add_time:.3f} seconds")
    print(f"Rate: {10000/add_time:.0f} experiences/second")
    
    # Benchmark sampling
    start = time.time()
    for _ in range(100):
        experiences, weights, indices = buffer.sample(256)
    sample_time = time.time() - start
    
    print(f"\nSampling 100 batches (size=256): {sample_time:.3f} seconds")
    print(f"Rate: {100*256/sample_time:.0f} samples/second")
    
    # Benchmark priority updates
    start = time.time()
    for _ in range(100):
        td_errors = np.random.randn(256)
        buffer.update_priorities(indices, td_errors)
    update_time = time.time() - start
    
    print(f"\nUpdating 100 batches of priorities: {update_time:.3f} seconds")
    print(f"Rate: {100*256/update_time:.0f} updates/second")
    
    print("\n✓ Performance demonstration complete!")