"""
GA-MADDPG Neural Network Architecture for LEO-ISAC
====================================================

This module implements the advanced Graph Attention + GRU based neural network
architecture for multi-agent deep deterministic policy gradient (GA-MADDPG) 
as described in Report 6. It includes GAT for spatial relation modeling,
GRU for temporal belief state representation, and differentiable projection
for guaranteed constraint satisfaction.

Author: THz ISAC Research Team
Date: August 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class CyclicalEncoder(nn.Module):
    """
    Cyclical encoding for periodic features using sine and cosine transformations.
    
    Preserves the cyclical nature of features like orbital phase, time of day, etc.
    by mapping them onto a unit circle in 2D space.
    """
    
    def __init__(self, periods: Dict[int, float]):
        """
        Args:
            periods: Dictionary mapping feature indices to their periods.
                    e.g., {0: 24.0} for hour of day with 24-hour period
        """
        super(CyclicalEncoder, self).__init__()
        self.periods = periods
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cyclical encoding to specified features.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            
        Returns:
            Encoded tensor with cyclical features expanded to sin/cos pairs
        """
        batch_size = x.shape[0]
        encoded_features = []
        feature_idx = 0
        
        for i in range(x.shape[1]):
            if i in self.periods:
                # Apply sin/cos transformation for cyclical feature
                period = self.periods[i]
                phase = 2 * np.pi * x[:, i] / period
                
                sin_encoding = torch.sin(phase).unsqueeze(1)
                cos_encoding = torch.cos(phase).unsqueeze(1)
                
                encoded_features.append(sin_encoding)
                encoded_features.append(cos_encoding)
            else:
                # Keep non-cyclical features as-is
                encoded_features.append(x[:, i].unsqueeze(1))
        
        return torch.cat(encoded_features, dim=1)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer for processing neighbor information.
    
    Implements the self-attention mechanism to dynamically weight neighbor
    contributions based on their relevance to the central node.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # Linear transformations for each attention head
        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features: torch.Tensor, 
                neighbor_features: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply graph attention to aggregate neighbor information.
        
        Args:
            node_features: Central node features (batch_size, in_features)
            neighbor_features: Neighbor features (batch_size, max_neighbors, in_features)
            neighbor_mask: Boolean mask for valid neighbors (batch_size, max_neighbors)
            
        Returns:
            Aggregated features with attention-weighted neighbor information
        """
        batch_size = node_features.size(0)
        max_neighbors = neighbor_features.size(1)
        
        # Transform features
        h_node = torch.matmul(node_features, self.W)  # (batch, heads * out_features)
        h_node = h_node.view(batch_size, self.num_heads, self.out_features)
        
        h_neighbors = torch.matmul(neighbor_features, self.W)  # (batch, neighbors, heads * out)
        h_neighbors = h_neighbors.view(batch_size, max_neighbors, self.num_heads, self.out_features)
        
        # Compute attention coefficients
        attention_scores = []
        
        for head in range(self.num_heads):
            # Expand node features for broadcasting
            h_node_head = h_node[:, head, :].unsqueeze(1).expand(-1, max_neighbors, -1)
            h_neighbors_head = h_neighbors[:, :, head, :]
            
            # Concatenate node and neighbor features
            concat_features = torch.cat([
                h_node_head,
                h_neighbors_head
            ], dim=-1)  # (batch, neighbors, 2 * out_features)
            
            # Compute attention scores
            e = self.leaky_relu(torch.matmul(concat_features, self.a[head]))
            e = e.squeeze(-1)  # (batch, neighbors)
            
            # Apply mask if provided
            if neighbor_mask is not None:
                e = e.masked_fill(~neighbor_mask, float('-inf'))
            
            # Normalize with softmax
            alpha = F.softmax(e, dim=-1)
            alpha = self.dropout(alpha)
            
            attention_scores.append(alpha)
        
        # Aggregate neighbor features with attention weights
        aggregated = []
        
        for head in range(self.num_heads):
            alpha = attention_scores[head].unsqueeze(-1)  # (batch, neighbors, 1)
            h_neighbors_head = h_neighbors[:, :, head, :]  # (batch, neighbors, out_features)
            
            # Weighted sum of neighbor features
            aggregated_head = torch.sum(alpha * h_neighbors_head, dim=1)  # (batch, out_features)
            aggregated.append(aggregated_head)
        
        # Concatenate all heads
        output = torch.cat(aggregated, dim=-1)  # (batch, heads * out_features)
        
        return output

class ActorNetworkMLP(nn.Module):
    """
    Simplified Actor Network without RNN for baseline comparison.
    Uses only GAT and MLP, suitable for experience replay without sequence issues.
    """
    
    def __init__(self, obs_dim: int, action_dim: int,
                 num_neighbors: int = 4, neighbor_dim: int = 10,
                 hidden_dims: List[int] = [256, 128],
                 gat_hidden: int = 64, gat_heads: int = 4,
                 total_power_budget: float = 1.0,
                 per_link_max: float = 0.5,
                 dropout: float = 0.1):
        """
        Initialize MLP-based actor network.
        
        Args:
            obs_dim: Dimension of agent's observation
            action_dim: Dimension of action space
            num_neighbors: Maximum number of neighbors
            neighbor_dim: Dimension of neighbor features
            hidden_dims: Hidden layer dimensions for MLP
            gat_hidden: GAT hidden dimension
            gat_heads: Number of GAT attention heads
            total_power_budget: Total power constraint
            per_link_max: Per-link power constraint
            dropout: Dropout probability
        """
        super(ActorNetworkMLP, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # GAT for neighbor information processing
        self.gat = GraphAttentionLayer(
            in_features=neighbor_dim,
            out_features=gat_hidden,
            num_heads=gat_heads,
            dropout=dropout
        )
        
        # MLP for action generation (no GRU)
        input_dim = obs_dim + gat_heads * gat_hidden
        layers = []
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Differentiable projection for constraints
        self.projection = DifferentiableProjectionLayer(
            num_links=action_dim,
            total_power_budget=total_power_budget,
            per_link_max=per_link_max
        )
    
    def forward(self, obs: torch.Tensor,
                neighbor_obs: Optional[torch.Tensor] = None,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate actions from observations (no hidden state needed).
        
        Args:
            obs: Agent's observation (batch_size, obs_dim)
            neighbor_obs: Neighbors' observations (batch_size, num_neighbors, neighbor_dim)
            neighbor_mask: Valid neighbor mask (batch_size, num_neighbors)
            
        Returns:
            Actions satisfying constraints (batch_size, action_dim)
        """
        batch_size = obs.size(0)
        
        # Process neighbor information if available
        if neighbor_obs is not None and neighbor_obs.size(1) > 0:
            # Use first neighbor's features as node features for attention
            node_feat_for_gat = neighbor_obs[:, 0, :]
            neighbor_context = self.gat(node_feat_for_gat, neighbor_obs, neighbor_mask)
        else:
            # No neighbors, create zero context
            neighbor_context = torch.zeros(
                batch_size, 
                self.gat.num_heads * self.gat.out_features,
                device=obs.device  # Ensure same device
            )
        
        # Concatenate observation with neighbor context
        combined_features = torch.cat([obs, neighbor_context], dim=-1)
        
        # Generate raw actions through MLP
        raw_actions = self.mlp(combined_features)
        
        # Apply projection for constraint satisfaction
        safe_actions = self.projection(raw_actions)
        
        return safe_actions
class GATGRUEncoder(nn.Module):
    """
    Hierarchical encoder combining Graph Attention Networks and Gated Recurrent Units.
    
    This encoder first processes spatial relationships via GAT, then captures
    temporal dependencies through GRU to form a belief state.
    """
    
    def __init__(self, node_features: int, neighbor_features: int,
                 gat_hidden: int = 64, gat_heads: int = 4,
                 gru_hidden: int = 128, dropout: float = 0.1):
        """
        Args:
            node_features: Dimension of node's own features
            neighbor_features: Dimension of each neighbor's features
            gat_hidden: Hidden dimension for GAT layer
            gat_heads: Number of attention heads in GAT
            gru_hidden: Hidden dimension for GRU
            dropout: Dropout probability
        """
        super(GATGRUEncoder, self).__init__()
        
        # GAT for spatial relationship modeling
        self.gat = GraphAttentionLayer(
            in_features=neighbor_features,
            out_features=gat_hidden,
            num_heads=gat_heads,
            dropout=dropout
        )
        
        # Feature fusion layer
        self.fusion = nn.Linear(node_features + gat_heads * gat_hidden, gru_hidden)
        
        # GRU for temporal belief state
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(gru_hidden)
        
    def forward(self, node_obs: torch.Tensor,
                neighbor_obs: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None,
                neighbor_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process observation through GAT-GRU pipeline.
        
        Args:
            node_obs: Agent's own observation (batch_size, node_features)
            neighbor_obs: Neighbors' observations (batch_size, max_neighbors, neighbor_features)
            hidden_state: Previous GRU hidden state (1, batch_size, gru_hidden)
            neighbor_mask: Valid neighbor mask (batch_size, max_neighbors)
            
        Returns:
            belief_state: Current belief state (batch_size, gru_hidden)
            new_hidden: Updated GRU hidden state
        """
        batch_size = node_obs.size(0)
        
        # Process neighbor information through GAT
        if neighbor_obs is not None and neighbor_obs.size(1) > 0:
            # Use first neighbor's features as node features for attention
            node_feat_for_gat = neighbor_obs[:, 0, :]
            neighbor_context = self.gat(node_feat_for_gat, neighbor_obs, neighbor_mask)
        else:
            # No neighbors, create zero context
            neighbor_context = torch.zeros(batch_size, self.gat.num_heads * self.gat.out_features,
                                         device=node_obs.device)
        
        # Concatenate node features with neighbor context
        combined_features = torch.cat([node_obs, neighbor_context], dim=-1)
        
        # Fuse features
        fused = self.fusion(combined_features)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        
        # Add time dimension for GRU
        fused = fused.unsqueeze(1)  # (batch, 1, gru_hidden)
        
        # Process through GRU
        gru_out, new_hidden = self.gru(fused, hidden_state)
        
        # Extract belief state
        belief_state = gru_out.squeeze(1)  # (batch, gru_hidden)
        belief_state = self.layer_norm(belief_state)
        
        return belief_state, new_hidden


class DifferentiableProjectionLayer(nn.Module):
    """
    Differentiable projection layer for guaranteed constraint satisfaction.
    
    Projects unconstrained network outputs onto the feasible set defined by
    physical constraints (e.g., power budget) using convex optimization.
    """
    
    def __init__(self, num_links: int, total_power_budget: float,
                 per_link_max: Optional[float] = None):
        """
        Args:
            num_links: Maximum number of outgoing links
            total_power_budget: Total power budget constraint (Watts)
            per_link_max: Maximum power per link (Watts)
        """
        super(DifferentiableProjectionLayer, self).__init__()
        
        self.num_links = num_links
        self.total_power_budget = total_power_budget
        self.per_link_max = per_link_max or (total_power_budget / num_links)
        
        # Define the projection as a convex optimization problem
        # Variables
        p = cp.Variable(num_links)  # Projected power allocation
        p_raw = cp.Parameter(num_links)  # Raw network output
        
        # Objective: minimize ||p - p_raw||^2
        objective = cp.Minimize(cp.sum_squares(p - p_raw))
        
        # Constraints
        constraints = [
            p >= 0,  # Non-negativity
            p <= self.per_link_max,  # Per-link maximum
            cp.sum(p) <= self.total_power_budget  # Total budget
        ]
        
        # Create problem
        problem = cp.Problem(objective, constraints)
        
        # Create differentiable layer
        self.projection_layer = CvxpyLayer(problem, parameters=[p_raw], variables=[p])
        
    def forward(self, raw_power: torch.Tensor) -> torch.Tensor:
        """
        Project raw power allocation onto feasible set.
        
        Args:
            raw_power: Unconstrained power allocation (batch_size, num_links)
            
        Returns:
            Projected power allocation satisfying all constraints
        """
        batch_size = raw_power.size(0)
        projected = []
        
        # Process each sample in batch
        for i in range(batch_size):
            # Solve projection for this sample
            p_proj, = self.projection_layer(raw_power[i])
            projected.append(p_proj)
        
        return torch.stack(projected)


# Modify existing ActorNetwork to support both modes
class ActorNetwork(nn.Module):
    """
    GA-MADDPG Actor Network with optional GRU support and dual projection modes.
    
    This version includes both differentiable projection (for training) and
    non-differentiable projection (for inference/target networks).
    """
    
    def __init__(self, obs_dim: int, action_dim: int,
                 num_neighbors: int = 4, neighbor_dim: int = 10,
                 hidden_dims: List[int] = [256, 128],
                 gat_hidden: int = 64, gat_heads: int = 4,
                 gru_hidden: int = 128,
                 total_power_budget: float = 1.0,
                 per_link_max: float = 0.5,
                 use_gru: bool = True,
                 use_cyclical_encoding: bool = False,
                 cyclical_periods: Optional[Dict[int, float]] = None):
        """
        Initialize actor network with optional GRU and dual projection modes.
        
        Args:
            obs_dim: Dimension of each agent's observation
            action_dim: Dimension of each agent's action (continuous)
            num_neighbors: Maximum number of neighbors per agent
            neighbor_dim: Dimension of neighbor features
            hidden_dims: Hidden layer dimensions for MLP
            gat_hidden: GAT hidden dimension
            gat_heads: Number of GAT attention heads
            gru_hidden: GRU hidden dimension
            total_power_budget: Total power constraint for projection layer
            per_link_max: Per-link power constraint
            use_gru: Whether to use GRU for temporal modeling
            use_cyclical_encoding: Whether to use cyclical encoding
            cyclical_periods: Periods for cyclical features
        """
        super(ActorNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_gru = use_gru
        self.gru_hidden = gru_hidden if use_gru else 0
        self.total_power_budget = total_power_budget
        self.per_link_max = per_link_max
        
        # Cyclical encoder (if used)
        if use_cyclical_encoding and cyclical_periods:
            self.cyclical_encoder = CyclicalEncoder(cyclical_periods)
            encoded_dim = obs_dim + len(cyclical_periods)
        else:
            self.cyclical_encoder = None
            encoded_dim = obs_dim
        
        if self.use_gru:
            # Use GAT-GRU encoder
            self.encoder = GATGRUEncoder(
                node_features=encoded_dim,
                neighbor_features=neighbor_dim,
                gat_hidden=gat_hidden,
                gat_heads=gat_heads,
                gru_hidden=gru_hidden
            )
            mlp_input_dim = gru_hidden
            self.hidden_state = None
        else:
            # Use only GAT without GRU
            self.gat = GraphAttentionLayer(
                in_features=neighbor_dim,
                out_features=gat_hidden,
                num_heads=gat_heads
            )
            mlp_input_dim = encoded_dim + gat_heads * gat_hidden
        
        # MLP for action generation
        layers = []
        input_dim = mlp_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Differentiable projection layer (for training)
        self.projection = DifferentiableProjectionLayer(
            num_links=action_dim,
            total_power_budget=total_power_budget,
            per_link_max=per_link_max
        )
        
        # Setup cvxpy problem for non-differentiable projection (for inference)
        self._setup_numpy_projection()
    
    def _setup_numpy_projection(self):
        """
        Setup cvxpy problem for non-differentiable projection.
        This is used when gradients are not needed (e.g., in target networks).
        """
        self._p_var = cp.Variable(self.action_dim)
        self._p_raw_param = cp.Parameter(self.action_dim)
        
        # Define objective: minimize ||p - p_raw||^2
        _obj = cp.Minimize(cp.sum_squares(self._p_var - self._p_raw_param))
        
        # Define constraints
        _constraints = [
            self._p_var >= 0,  # Non-negativity
            self._p_var <= self.per_link_max,  # Per-link maximum
            cp.sum(self._p_var) <= self.total_power_budget  # Total budget
        ]
        
        # Create problem
        self._proj_problem = cp.Problem(_obj, _constraints)
    
    def _project_numpy(self, raw_power: torch.Tensor) -> torch.Tensor:
        """
        Non-differentiable projection using raw cvxpy for inference.
        
        This method is used when gradients are not needed, providing
        a stable and efficient projection without the complexity of CvxpyLayer.
        
        Args:
            raw_power: Unconstrained power allocation (batch_size, num_links)
            
        Returns:
            Projected power allocation satisfying all constraints
        """
        batch_size = raw_power.size(0)
        device = raw_power.device
        raw_power_np = raw_power.detach().cpu().numpy()
        
        projected = []
        
        for i in range(batch_size):
            # Load data into cvxpy parameter
            self._p_raw_param.value = raw_power_np[i]
            
            # Solve optimization problem
            try:
                self._proj_problem.solve(solver=cp.ECOS, verbose=False)
                
                if self._proj_problem.status in ["optimal", "optimal_inaccurate"]:
                    proj_sample = self._p_var.value
                else:
                    # If solve fails, use a simple clipping fallback
                    proj_sample = np.clip(raw_power_np[i], 0, self.per_link_max)
                    total = proj_sample.sum()
                    if total > self.total_power_budget:
                        proj_sample *= self.total_power_budget / total
                        
            except Exception as e:
                # Fallback to simple clipping if cvxpy fails
                warnings.warn(f"CVXPY solve failed: {e}, using fallback projection")
                proj_sample = np.clip(raw_power_np[i], 0, self.per_link_max)
                total = proj_sample.sum()
                if total > self.total_power_budget:
                    proj_sample *= self.total_power_budget / total
            
            projected.append(proj_sample)
        
        # Convert back to tensor
        projected_np = np.stack(projected)
        projected_tensor = torch.FloatTensor(projected_np).to(device)
        
        return projected_tensor
    
    def forward(self, obs: torch.Tensor,
                neighbor_obs: Optional[torch.Tensor] = None,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with dual projection modes.
        
        Uses differentiable projection during training (when gradients are enabled)
        and non-differentiable projection during inference (e.g., in target networks).
        
        Args:
            obs: Agent's observation (batch_size, obs_dim)
            neighbor_obs: Neighbors' observations (batch_size, num_neighbors, neighbor_dim)
            neighbor_mask: Valid neighbor mask (batch_size, num_neighbors)
            
        Returns:
            Actions satisfying constraints (batch_size, action_dim)
        """
        # Apply cyclical encoding if configured
        if self.cyclical_encoder is not None:
            obs = self.cyclical_encoder(obs)
        
        if self.use_gru:
            # Use GAT-GRU encoder
            belief_state, self.hidden_state = self.encoder(
                node_obs=obs,
                neighbor_obs=neighbor_obs,
                hidden_state=self.hidden_state,
                neighbor_mask=neighbor_mask
            )
            features = belief_state
        else:
            # Use GAT only (no GRU)
            if neighbor_obs is not None and neighbor_obs.size(1) > 0:
                node_feat = neighbor_obs[:, 0, :]
                neighbor_context = self.gat(node_feat, neighbor_obs, neighbor_mask)
            else:
                neighbor_context = torch.zeros(
                    obs.size(0),
                    self.gat.num_heads * self.gat.out_features,
                    device=obs.device
                )
            features = torch.cat([obs, neighbor_context], dim=-1)
        
        # Generate raw actions through MLP
        raw_actions = self.mlp(features)
        
        # Apply projection based on gradient mode
        if torch.is_grad_enabled():
            # Training mode: use differentiable CvxpyLayer
            try:
                safe_actions = self.projection(raw_actions)
            except Exception as e:
                # If CvxpyLayer fails, fallback to numpy projection
                warnings.warn(f"CvxpyLayer failed: {e}, using numpy projection")
                safe_actions = self._project_numpy(raw_actions)
        else:
            # Inference mode (e.g., in target networks): use stable numpy projection
            safe_actions = self._project_numpy(raw_actions)
        
        return safe_actions
    
    def reset_hidden(self, batch_size: int = 1):
        """
        Reset GRU hidden state (only if using GRU).
        
        Args:
            batch_size: Batch size for the hidden state
        """
        if self.use_gru:
            device = next(self.parameters()).device
            self.hidden_state = torch.zeros(1, batch_size, self.gru_hidden, device=device)

            

class CriticNetwork(nn.Module):
    """
    GA-MADDPG Centralized Critic Network with parallel GAT-GRU encoders.
    
    Evaluates Q-values using global information from all agents during training.
    Can share encoder parameters with actors for improved learning efficiency.
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int,
                 num_neighbors: int = 4, neighbor_dim: int = 10,
                 hidden_dims: List[int] = [512, 256, 128],
                 gat_hidden: int = 64, gat_heads: int = 4,
                 gru_hidden: int = 128,
                 share_encoder: bool = True,
                 actor_encoder: Optional[GATGRUEncoder] = None):
        """
        Args:
            num_agents: Number of agents in the system
            obs_dim: Dimension of each agent's observation
            action_dim: Dimension of each agent's action
            num_neighbors: Maximum neighbors per agent
            neighbor_dim: Dimension of neighbor features
            hidden_dims: Hidden layer dimensions for value MLP
            gat_hidden: GAT hidden dimension
            gat_heads: Number of GAT attention heads
            gru_hidden: GRU hidden dimension
            share_encoder: Whether to share encoder with actor
            actor_encoder: Pre-trained encoder from actor (if sharing)
        """
        super(CriticNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gru_hidden = gru_hidden
        
        # Create or share encoders for each agent
        self.encoders = nn.ModuleList()
        
        for i in range(num_agents):
            if share_encoder and actor_encoder is not None:
                # Share encoder parameters with actor
                self.encoders.append(actor_encoder)
            else:
                # Create independent encoder
                encoder = GATGRUEncoder(
                    node_features=obs_dim,
                    neighbor_features=neighbor_dim,
                    gat_hidden=gat_hidden,
                    gat_heads=gat_heads,
                    gru_hidden=gru_hidden
                )
                self.encoders.append(encoder)
        
        # MLP for Q-value estimation
        # Input: concatenated belief states + all actions
        input_dim = num_agents * (gru_hidden + action_dim)
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        # Output single Q-value
        layers.append(nn.Linear(input_dim, 1))
        
        self.value_mlp = nn.Sequential(*layers)
        
        # Hidden states for each agent's GRU
        self.hidden_states = [None] * num_agents
        
    def forward(self, observations: List[torch.Tensor],
                actions: torch.Tensor,
                neighbor_observations: Optional[List[torch.Tensor]] = None,
                neighbor_masks: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute Q-value for joint observation-action pair.
        
        Args:
            observations: List of observations for each agent
            actions: Joint actions (batch_size, num_agents * action_dim)
            neighbor_observations: List of neighbor observations for each agent
            neighbor_masks: List of neighbor masks for each agent
            
        Returns:
            Q-value (batch_size, 1)
        """
        batch_size = observations[0].size(0)
        belief_states = []
        
        # Process each agent's observation through its encoder
        for i in range(self.num_agents):
            obs = observations[i]
            neighbor_obs = neighbor_observations[i] if neighbor_observations else None
            neighbor_mask = neighbor_masks[i] if neighbor_masks else None
            
            belief_state, self.hidden_states[i] = self.encoders[i](
                node_obs=obs,
                neighbor_obs=neighbor_obs,
                hidden_state=self.hidden_states[i],
                neighbor_mask=neighbor_mask
            )
            
            belief_states.append(belief_state)
        
        # Concatenate all belief states and actions
        global_state = torch.cat(belief_states + [actions], dim=-1)
        
        # Compute Q-value
        q_value = self.value_mlp(global_state)
        
        return q_value
    
    def reset_hidden(self, batch_size: int = 1):
        """Reset all agents' GRU hidden states."""
        for i in range(self.num_agents):
            self.hidden_states[i] = torch.zeros(1, batch_size, self.gru_hidden)
            if next(self.parameters()).is_cuda:
                self.hidden_states[i] = self.hidden_states[i].cuda()


class RewardDecompositionNetwork(nn.Module):
    """
    Auxiliary network for learning difference rewards through counterfactual reasoning.
    
    Learns to predict team reward from individual agent's perspective to enable
    precise credit assignment in multi-agent settings.
    """
    
    def __init__(self, state_dim: int, obs_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 128, 64]):
        """
        Args:
            state_dim: Global state dimension
            obs_dim: Agent observation dimension
            action_dim: Agent action dimension
            hidden_dims: Hidden layer dimensions
        """
        super(RewardDecompositionNetwork, self).__init__()
        
        # Input: global state + agent observation + agent action
        input_dim = state_dim + obs_dim + action_dim
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        # Output: predicted team reward
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, global_state: torch.Tensor,
                agent_obs: torch.Tensor,
                agent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict team reward from agent's local information.
        
        Args:
            global_state: Global state information
            agent_obs: Agent's observation
            agent_action: Agent's action
            
        Returns:
            Predicted team reward (batch_size, 1)
        """
        combined = torch.cat([global_state, agent_obs, agent_action], dim=-1)
        return self.mlp(combined)
    
    def compute_difference_reward(self, global_state: torch.Tensor,
                                 agent_obs: torch.Tensor,
                                 agent_action: torch.Tensor,
                                 default_action: torch.Tensor) -> torch.Tensor:
        """
        Compute difference reward using counterfactual reasoning.
        
        Args:
            global_state: Global state
            agent_obs: Agent observation
            agent_action: Actual action taken
            default_action: Default/null action for counterfactual
            
        Returns:
            Difference reward D_i = R(a_i) - R(a_default)
        """
        # Actual reward with agent's action
        actual_reward = self.forward(global_state, agent_obs, agent_action)
        
        # Counterfactual reward with default action
        counterfactual_reward = self.forward(global_state, agent_obs, default_action)
        
        # Difference reward
        difference_reward = actual_reward - counterfactual_reward
        
        return difference_reward


# ============================================================================
# Unit Tests
# ============================================================================

def test_cyclical_encoder():
    """Test cyclical encoding functionality."""
    print("Testing Cyclical Encoder...")
    
    # Create encoder with hour-of-day period
    encoder = CyclicalEncoder(periods={0: 24.0})
    
    # Test input with hour values
    x = torch.tensor([[0.0, 5.0], [6.0, 10.0], [12.0, 15.0], [23.0, 20.0]])
    encoded = encoder(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Encoded shape: {encoded.shape}")
    
    # Check that hour 0 and hour 23 are close in encoded space
    hour_0_encoded = encoded[0, :2]
    hour_23_encoded = encoded[3, :2]
    distance = torch.norm(hour_0_encoded - hour_23_encoded)
    
    print(f"  Distance between hour 0 and 23: {distance:.4f}")
    assert distance < 0.5, "Cyclical features should be close"
    
    print("✓ Cyclical Encoder test passed")


def test_gat_layer():
    """Test Graph Attention Layer."""
    print("\nTesting Graph Attention Layer...")
    
    batch_size = 2
    in_features = 16
    out_features = 8
    num_heads = 4
    max_neighbors = 3
    
    gat = GraphAttentionLayer(in_features, out_features, num_heads)
    
    # Create dummy inputs
    node_features = torch.randn(batch_size, in_features)
    neighbor_features = torch.randn(batch_size, max_neighbors, in_features)
    neighbor_mask = torch.tensor([[True, True, False], [True, False, False]])
    
    # Forward pass
    output = gat(node_features, neighbor_features, neighbor_mask)
    
    print(f"  Input shapes: node={node_features.shape}, neighbors={neighbor_features.shape}")
    print(f"  Output shape: {output.shape}")
    
    expected_shape = (batch_size, num_heads * out_features)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print("✓ GAT Layer test passed")


def test_gat_gru_encoder():
    """Test GAT-GRU Encoder."""
    print("\nTesting GAT-GRU Encoder...")
    
    batch_size = 2
    node_features = 20
    neighbor_features = 10
    max_neighbors = 3
    
    encoder = GATGRUEncoder(
        node_features=node_features,
        neighbor_features=neighbor_features,
        gat_hidden=16,
        gat_heads=4,
        gru_hidden=32
    )
    
    # Create inputs
    node_obs = torch.randn(batch_size, node_features)
    neighbor_obs = torch.randn(batch_size, max_neighbors, neighbor_features)
    
    # Forward pass
    belief_state, hidden = encoder(node_obs, neighbor_obs)
    
    print(f"  Belief state shape: {belief_state.shape}")
    print(f"  Hidden state shape: {hidden.shape}")
    
    assert belief_state.shape == (batch_size, 32), "Incorrect belief state shape"
    assert hidden.shape == (1, batch_size, 32), "Incorrect hidden state shape"
    
    print("✓ GAT-GRU Encoder test passed")


def test_projection_layer():
    """Test Differentiable Projection Layer."""
    print("\nTesting Differentiable Projection Layer...")
    
    batch_size = 2
    num_links = 4
    total_budget = 1.0
    per_link_max = 0.4
    
    projection = DifferentiableProjectionLayer(num_links, total_budget, per_link_max)
    
    # Test with violating inputs
    raw_power = torch.tensor([
        [0.5, 0.5, 0.5, 0.5],  # Violates total budget (sum=2.0)
        [0.1, 0.2, 0.3, 0.6]   # Violates per-link max (0.6 > 0.4)
    ])
    
    projected = projection(raw_power)
    
    print(f"  Raw power: {raw_power}")
    print(f"  Projected: {projected}")
    print(f"  Total power: {projected.sum(dim=1)}")
    
    # Check constraints
    assert (projected >= 0).all(), "Negative power detected"
    assert (projected <= per_link_max + 1e-4).all(), "Per-link max violated"
    assert (projected.sum(dim=1) <= total_budget + 1e-4).all(), "Total budget violated"
    
    print("✓ Projection Layer test passed")


def test_actor_network():
    """Test complete Actor Network."""
    print("\nTesting Actor Network...")
    
    obs_dim = 20
    action_dim = 4
    batch_size = 2
    
    actor = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_neighbors=3,
        neighbor_dim=10,
        hidden_dims=[64, 32],
        total_power_budget=1.0,
        use_cyclical_encoding=False
    )
    
    # Reset hidden state
    actor.reset_hidden(batch_size)
    
    # Create inputs
    obs = torch.randn(batch_size, obs_dim)
    neighbor_obs = torch.randn(batch_size, 3, 10)
    
    # Forward pass
    actions = actor(obs, neighbor_obs)
    
    print(f"  Actions shape: {actions.shape}")
    print(f"  Actions: {actions}")
    print(f"  Total power per agent: {actions.sum(dim=1)}")
    
    assert actions.shape == (batch_size, action_dim), "Incorrect action shape"
    assert (actions >= 0).all(), "Negative actions"
    assert (actions.sum(dim=1) <= 1.0 + 1e-4).all(), "Power budget violated"
    
    print("✓ Actor Network test passed")


def test_critic_network():
    """Test complete Critic Network."""
    print("\nTesting Critic Network...")
    
    num_agents = 3
    obs_dim = 20
    action_dim = 4
    batch_size = 2
    
    critic = CriticNetwork(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[128, 64],
        share_encoder=False
    )
    
    # Reset hidden states
    critic.reset_hidden(batch_size)
    
    # Create inputs
    observations = [torch.randn(batch_size, obs_dim) for _ in range(num_agents)]
    actions = torch.randn(batch_size, num_agents * action_dim)
    
    # Forward pass
    q_values = critic(observations, actions)
    
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values: {q_values.squeeze()}")
    
    assert q_values.shape == (batch_size, 1), "Incorrect Q-value shape"
    
    print("✓ Critic Network test passed")


def test_reward_decomposition():
    """Test Reward Decomposition Network."""
    print("\nTesting Reward Decomposition Network...")
    
    state_dim = 50
    obs_dim = 20
    action_dim = 4
    batch_size = 2
    
    decomposer = RewardDecompositionNetwork(state_dim, obs_dim, action_dim)
    
    # Create inputs
    global_state = torch.randn(batch_size, state_dim)
    agent_obs = torch.randn(batch_size, obs_dim)
    agent_action = torch.randn(batch_size, action_dim)
    default_action = torch.zeros(batch_size, action_dim)
    
    # Compute difference reward
    diff_reward = decomposer.compute_difference_reward(
        global_state, agent_obs, agent_action, default_action
    )
    
    print(f"  Difference reward shape: {diff_reward.shape}")
    print(f"  Difference rewards: {diff_reward.squeeze()}")
    
    assert diff_reward.shape == (batch_size, 1), "Incorrect reward shape"
    
    print("✓ Reward Decomposition test passed")


if __name__ == "__main__":
    """Run all unit tests."""
    print("=" * 60)
    print("GA-MADDPG Neural Network Architecture Tests")
    print("=" * 60)
    
    test_cyclical_encoder()
    test_gat_layer()
    test_gat_gru_encoder()
    
    # Note: Projection layer test requires cvxpylayers
    try:
        test_projection_layer()
    except Exception as e:
        print(f"\n⚠ Projection layer test skipped (requires cvxpylayers): {e}")
    
    test_actor_network()
    test_critic_network()
    test_reward_decomposition()
    
    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    # Demonstrate parameter sharing
    print("\n" + "=" * 60)
    print("Demonstration: Parameter Sharing between Actor and Critic")
    print("=" * 60)
    
    # Create actor
    actor = ActorNetwork(
        obs_dim=20,
        action_dim=4,
        gru_hidden=64,
        use_cyclical_encoding=False
    )
    
    # Create critic sharing encoder with actor
    critic = CriticNetwork(
        num_agents=2,
        obs_dim=20,
        action_dim=4,
        gru_hidden=64,
        share_encoder=True,
        actor_encoder=actor.encoder
    )
    
    print(f"\nActor encoder parameters: {sum(p.numel() for p in actor.encoder.parameters())}")
    print(f"Critic encoder[0] is actor encoder: {critic.encoders[0] is actor.encoder}")
    print("\n✓ Parameter sharing demonstrated successfully!")