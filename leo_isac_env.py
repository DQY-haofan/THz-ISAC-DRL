"""
LEO-ISAC Multi-Agent Reinforcement Learning Environment (Refactored)
=====================================================================

This module provides a streamlined Gym-like environment for multi-agent 
reinforcement learning in LEO satellite constellations. By delegating all
physical layer computations to the PhysicalLayerInterface, this environment
focuses solely on MARL orchestration and agent interaction logic.

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings

# Import constellation management
from geom import (
    Constellation, Satellite, StateVector, OrbitalElements,
    get_visibility_graph
)

# Import the new physical layer interface
from physical_layer_interface import PhysicalLayerInterface


@dataclass
class ConstellationConfig:
    """Configuration for LEO satellite constellation."""
    n_satellites: int = 4
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    propagation_model: str = "keplerian"
    max_isl_range_km: float = 5000.0
    
    def create_walker_constellation(self) -> List[Satellite]:
        """Create a Walker constellation with evenly spaced satellites."""
        satellites = []
        altitude_m = self.altitude_km * 1000
        semi_major_axis = 6371000 + altitude_m
        
        for i in range(self.n_satellites):
            mean_anomaly = (2 * np.pi * i) / self.n_satellites
            
            elements = OrbitalElements(
                a=semi_major_axis,
                e=0.001,
                i=np.deg2rad(self.inclination_deg),
                raan=0.0,
                omega=0.0,
                nu=mean_anomaly
            )
            
            sat = Satellite(
                sat_id=f"SAT_{i}",
                orbital_elements=elements,
                propagation_model=self.propagation_model
            )
            satellites.append(sat)
            
        return satellites


@dataclass
class HardwareConfig:
    """Hardware and system configuration."""
    frequency_ghz: float = 300.0
    bandwidth_ghz: float = 10.0
    antenna_diameter_m: float = 0.5
    tx_power_max_dbm: float = 30.0
    noise_figure_db: float = 3.0
    hardware_level: str = "High-Performance"


@dataclass
class ISACConfig:
    """ISAC task configuration and reward weights."""
    w_comm: float = 1.0
    w_sens: float = 0.5
    w_penalty: float = 10.0
    sensing_mode: str = "cooperative_orbit_determination"
    min_links_for_sensing: int = 3
    time_step_s: float = 1.0
    episode_length: int = 100


class AgentObservation:
    """Container for a single agent's local observation."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.channel_states = {}  # Link quality indicators
        self.buffer_lengths = {}  # Data queues
        self.interference_level = 0.0  # Aggregate interference
        self.neighbor_activities = {}  # Neighbor transmit indicators
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.available_power = 0.0
        
    def to_vector(self) -> np.ndarray:
        """Convert observation to fixed-size vector for NN input."""
        obs_list = []
        
        # Channel quality (normalized SINR values)
        qualities = list(self.channel_states.values())[:4]
        qualities.extend([0.0] * (4 - len(qualities)))
        obs_list.extend(qualities)
        
        # Buffer states
        buffers = list(self.buffer_lengths.values())[:4]
        buffers.extend([0.0] * (4 - len(buffers)))
        obs_list.extend(buffers)
        
        # Scalar values
        obs_list.append(self.interference_level)
        obs_list.append(self.available_power)
        
        # Kinematic state (normalized)
        obs_list.extend(self.position / 1e6)
        obs_list.extend(self.velocity / 1e3)
        
        return np.array(obs_list, dtype=np.float32)


class LEO_ISAC_Env:
    """
    Streamlined Multi-agent LEO-ISAC environment.
    
    This refactored version delegates all physical layer computations to the
    PhysicalLayerInterface, focusing solely on MARL orchestration, agent
    interaction, and reward shaping.
    """
    
    def __init__(self,
                 constellation_config: Optional[ConstellationConfig] = None,
                 hardware_config: Optional[HardwareConfig] = None,
                 isac_config: Optional[ISACConfig] = None):
        """
        Initialize the LEO-ISAC environment.
        
        Args:
            constellation_config: Constellation parameters
            hardware_config: Hardware specifications
            isac_config: ISAC task and reward configuration
        """
        # Use defaults if not provided
        self.const_config = constellation_config or ConstellationConfig()
        self.hw_config = hardware_config or HardwareConfig()
        self.isac_config = isac_config or ISACConfig()
        
        # Initialize constellation
        satellites = self.const_config.create_walker_constellation()
        self.constellation = Constellation(satellites)
        self.n_agents = len(satellites)
        self.agent_ids = [sat.sat_id for sat in satellites]
        
        # Initialize physical layer interface (NEW!)
        self.phy_interface = PhysicalLayerInterface(
            self.const_config,
            self.hw_config,
            self.isac_config
        )
        
        # Convert power limit for action validation
        self.max_tx_power_w = 10**((self.hw_config.tx_power_max_dbm - 30) / 10)
        
        # State tracking
        self.current_time = 0.0
        self.episode_step = 0
        self.active_links = set()
        self.link_registry = {}  # Maps link_id to (tx, rx) pairs
        
        # Action history for neighbor information
        self.prev_actions = {agent_id: {} for agent_id in self.agent_ids}
        
        # Data buffers (simulated traffic)
        self.data_buffers = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.metrics_history = []
        
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to initial state.
        
        Returns:
            Dictionary mapping agent IDs to their initial observations
        """
        # Reset time
        self.current_time = 0.0
        self.episode_step = 0
        
        # Reset constellation to initial positions
        for sat_id, satellite in self.constellation.satellites.items():
            satellite.current_state = StateVector(
                position=satellite.initial_state.position.copy(),
                velocity=satellite.initial_state.velocity.copy(),
                clock_bias=satellite.initial_state.clock_bias,
                clock_drift=satellite.initial_state.clock_drift,
                timestamp=0.0
            )
        
        # Update network topology
        self._update_topology()
        
        # Initialize data buffers
        self._reset_data_buffers()
        
        # Reset action history
        self.prev_actions = {agent_id: {} for agent_id in self.agent_ids}
        
        # Clear metrics
        self.metrics_history = []
        
        # Update physical layer with initial state
        self._sync_physical_layer({})
        
        # Compute initial observations
        observations = self._compute_observations()
        
        return observations
    
    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute one environment step with joint actions from all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions
                    {agent_id: {'power_allocation': {link_id: power_w}, 
                               'beam_selection': {}}}
        
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        # Validate and store actions
        self._validate_actions(actions)
        self.prev_actions = actions.copy()
        
        # Propagate orbital dynamics
        self.current_time += self.isac_config.time_step_s
        self.constellation.propagate_all(self.current_time)
        
        # Update network topology
        self._update_topology()
        
        # Sync physical layer with current state and actions
        self._sync_physical_layer(actions)
        
        # Calculate rewards using physical layer interface
        rewards = self._compute_rewards(actions)
        
        # Update data buffers based on achieved rates
        self._update_data_buffers()
        
        # Compute new observations
        observations = self._compute_observations()
        
        # Check termination
        self.episode_step += 1
        done = (self.episode_step >= self.isac_config.episode_length)
        
        # Collect info
        info = self._collect_info()
        
        return observations, rewards, done, info
    
    def _update_topology(self):
        """Update network topology based on current satellite positions."""
        # Get visibility graph
        self.active_links = get_visibility_graph(
            self.constellation,
            self.current_time,
            max_range=self.const_config.max_isl_range_km * 1000,
            check_obstruction=True
        )
        
        # Update link registry
        self.link_registry.clear()
        for (sat_i, sat_j) in self.active_links:
            # Forward link
            link_id_fwd = f"{sat_i}->{sat_j}"
            self.link_registry[link_id_fwd] = (sat_i, sat_j)
            # Reverse link
            link_id_rev = f"{sat_j}->{sat_i}"
            self.link_registry[link_id_rev] = (sat_j, sat_i)
    
    def _sync_physical_layer(self, actions: Dict):
        """
        Synchronize physical layer interface with current state.
        
        This is the KEY integration point with the new interface.
        """
        # Get current satellite states
        state_matrix = self.constellation.get_state_matrix()
        
        # Prepare power allocations in the format expected by interface
        power_allocations = {}
        for agent_id, action in actions.items():
            if 'power_allocation' in action:
                power_allocations[agent_id] = action
        
        # Update physical layer with current dynamic state
        self.phy_interface.update_dynamic_state(
            current_time=self.current_time,
            satellite_states=state_matrix.flatten(),
            active_links=self.active_links,
            power_allocations=power_allocations,
            link_registry=self.link_registry # <-- 添加这一行
        )
    
    def _compute_rewards(self, actions: Dict) -> Dict[str, float]:
        """
        Compute ISAC rewards with theoretically consistent sensing reward.
        """
        rewards = {}
        
        for agent_id in self.agent_ids:
            # Communication reward
            r_comm = self.phy_interface.get_total_comm_reward(agent_id)
            
            # Sensing reward
            r_sens = self._compute_sensing_reward_a_optimal(agent_id)
            
            # --- START OF CORRECTION ---
            #
            # 在使用 r_penalty 之前，必须将其初始化为一个默认值。
            r_penalty = 0.0
            #
            # --- END OF CORRECTION ---
            
            # 只有当智能体采取了有效动作时，才计算惩罚
            if agent_id in actions and 'power_allocation' in actions[agent_id]:
                power_alloc = actions[agent_id].get('power_allocation', {})
                total_power = sum(power_alloc.values())
                
                # 注意：这里的惩罚应该由可微投影层处理，但我们保留它作为双重保障
                if total_power > self.max_tx_power_w:
                    r_penalty = self.isac_config.w_penalty * (
                        (total_power - self.max_tx_power_w) / self.max_tx_power_w
                    )
            
            # Combined ISAC reward
            rewards[agent_id] = (
                self.isac_config.w_comm * r_comm +
                self.isac_config.w_sens * r_sens -
                r_penalty
            )
            
            # (Debug print from previous step, can be kept or removed)
            if os.getenv('VERBOSE_DEBUG') == '1' and self.episode_step < 5:
                print(f"[DEBUG Env-RWD] Agt {agent_id}: R_comm={r_comm:.3f}, R_sens={r_sens:.3f}, "
                      f"R_pen={r_penalty:.3f} -> Total={rewards[agent_id]:.3f}")

        return rewards
    

    def _compute_sensing_reward_a_optimal(self, agent_id: str) -> float:
        """
        Compute sensing reward based on A-optimality (minimize trace of CRLB).
        
        A-optimal design minimizes the trace of the CRLB matrix,
        which corresponds to minimizing the sum of estimation error variances.
        This is implemented as a proxy using the trace of FIM contributions.
        
        Note: The reward is based on trace(J_link), which relates to A-optimality
        (minimizing trace of CRLB), not D-optimality (minimizing determinant of CRLB).
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Sensing reward based on FIM trace contribution
        """
        try:
            # Get current network FIM from physical layer
            if self.phy_interface.efim is not None:
                current_fim = self.phy_interface.efim
                
                # Calculate contribution from agent's links
                agent_contribution = 0.0
                for link_id, metrics in self.phy_interface.link_metrics.items():
                    if metrics['tx_id'] == agent_id:
                        # Get FIM contribution from this link
                        J_link = self.phy_interface.get_fim_contribution(link_id)
                        if J_link is not None:
                            # A-optimality: minimize trace(CRLB) = minimize trace(J^-1)
                            # As proxy, we maximize trace(J) which relates to information gain
                            agent_contribution += np.trace(J_link) / self.n_agents
                
                # Normalize by network size and apply logarithmic scaling
                reward = np.log(1 + agent_contribution)
            else:
                # No valid FIM, return zero reward
                reward = 0.0
                
        except Exception as e:
            warnings.warn(f"Error computing A-optimal reward: {e}")
            reward = 0.0
        
        return reward

    
    def _compute_observations(self) -> Dict[str, np.ndarray]:
        """
        Compute local observations for all agents with improved normalization.
        """
        # Statistics for SINR normalization (based on typical LEO ISL values)
        SINR_DB_MEAN = 10.0  # Typical SINR in dB
        SINR_DB_STD = 15.0   # Standard deviation in dB
        
        observations = {}
        
        for agent_id in self.agent_ids:
            obs = AgentObservation(agent_id)
            
            # Get satellite state
            sat = self.constellation.satellites[agent_id]
            obs.position = sat.current_state.position
            obs.velocity = sat.current_state.velocity
            obs.available_power = self.max_tx_power_w
            
            # Get link quality from physical layer with improved normalization
            for link_id in self.link_registry:
                tx_id, rx_id = self.link_registry[link_id]
                if tx_id == agent_id:
                    # Get SINR from physical layer
                    sinr_linear = self.phy_interface.get_effective_sinr(link_id)
                    
                    # Convert to dB scale and normalize
                    if sinr_linear > 0:
                        sinr_db = 10 * np.log10(sinr_linear)
                        # Standardize to approximately [-1, 1] range
                        normalized_sinr = (sinr_db - SINR_DB_MEAN) / SINR_DB_STD
                        # Apply tanh for bounded output
                        normalized_sinr = np.tanh(normalized_sinr)
                    else:
                        normalized_sinr = -1.0  # Minimum value for no signal
                    
                    obs.channel_states[link_id] = normalized_sinr
                    obs.buffer_lengths[link_id] = self.data_buffers[tx_id][rx_id]
            
            # Estimate interference level (simplified)
            obs.interference_level = self._estimate_interference(agent_id)
            
            # Neighbor activity from previous actions
            for neighbor_id in self._get_neighbors(agent_id):
                if neighbor_id in self.prev_actions:
                    total_power = sum(
                        self.prev_actions[neighbor_id].get('power_allocation', {}).values()
                    )
                    obs.neighbor_activities[neighbor_id] = total_power > 0
            
            observations[agent_id] = obs.to_vector()
        
        return observations
    
    def _estimate_interference(self, agent_id: str) -> float:
        """Estimate normalized interference level for an agent."""
        # Query aggregated interference from physical layer
        total_interference = 0.0
        
        # Sum interference on all agent's receiving links
        for link_id in self.link_registry:
            tx_id, rx_id = self.link_registry[link_id]
            if rx_id == agent_id:
                # Check all potential interferers
                for other_link_id in self.link_registry:
                    if other_link_id != link_id:
                        alpha = self.phy_interface.get_interference_coefficient(
                            link_id, other_link_id
                        )
                        total_interference += alpha
        
        # Normalize to [0, 1] range
        return np.tanh(total_interference)
    
    def _get_neighbors(self, agent_id: str) -> List[str]:
        """Get one-hop neighbors of an agent."""
        neighbors = set()
        
        for link_id, (tx_id, rx_id) in self.link_registry.items():
            if tx_id == agent_id:
                neighbors.add(rx_id)
            elif rx_id == agent_id:
                neighbors.add(tx_id)
        
        return list(neighbors)
    
    def _reset_data_buffers(self):
        """Initialize data buffers with random traffic demand."""
        self.data_buffers.clear()
        
        for tx_id in self.agent_ids:
            for rx_id in self.agent_ids:
                if tx_id != rx_id:
                    # Random initial buffer (0-100 Mb)
                    self.data_buffers[tx_id][rx_id] = np.random.uniform(0, 100)
    
    def _update_data_buffers(self):
        """Update data buffers based on achieved throughput."""
        # Generate new traffic
        for tx_id in self.agent_ids:
            for rx_id in self.agent_ids:
                if tx_id != rx_id:
                    # Add random traffic (0-10 Mb/s)
                    self.data_buffers[tx_id][rx_id] += np.random.uniform(0, 10)
        
        # Drain buffers based on achieved rates from physical layer
        for link_id in self.link_registry:
            tx_id, rx_id = self.link_registry[link_id]
            
            # Get throughput from physical layer
            throughput_gbps = self.phy_interface.get_throughput_gbps(link_id)
            
            if throughput_gbps > 0:
                # Drain buffer (convert Gbps to Mb)
                drained = min(
                    throughput_gbps * 1000 * self.isac_config.time_step_s,
                    self.data_buffers[tx_id][rx_id]
                )
                self.data_buffers[tx_id][rx_id] -= drained
    
    def _validate_actions(self, actions: Dict):
        """Validate agent actions for feasibility."""
        for agent_id, action in actions.items():
            if agent_id not in self.agent_ids:
                raise ValueError(f"Unknown agent ID: {agent_id}")
            
            # Check power constraint
            total_power = sum(action.get('power_allocation', {}).values())
            if total_power > self.max_tx_power_w * 1.1:  # 10% tolerance
                warnings.warn(
                    f"Agent {agent_id} exceeds power budget: "
                    f"{total_power:.2f} W > {self.max_tx_power_w:.2f} W"
                )
    
    def _collect_info(self) -> Dict:
        """
        Collect additional information from physical layer including privileged information.
        
        This includes the true global interference matrix for centralized critic training.
        """
        info = {
            'time': self.current_time,
            'step': self.episode_step,
            'n_active_links': len(self.active_links),
            'gdop': self.phy_interface.get_network_gdop()
        }
        
        # Calculate total throughput
        total_throughput = 0.0
        for link_id in self.link_registry:
            total_throughput += self.phy_interface.get_throughput_gbps(link_id)
        info['total_throughput'] = total_throughput
        
        # 添加特权信息：真实的全局干扰矩阵
        # 提取所有αℓm值并展平为向量
        interference_matrix_flat = []
        for victim_link in self.link_registry:
            for interferer_link in self.link_registry:
                if victim_link != interferer_link:
                    alpha_lm = self.phy_interface.get_interference_coefficient(
                        victim_link, interferer_link
                    )
                    interference_matrix_flat.append(alpha_lm)
        
        info['interference_matrix'] = np.array(interference_matrix_flat, dtype=np.float32)
        
        # 添加其他有用的全局信息
        info['link_sinrs'] = {}
        for link_id in self.link_registry:
            info['link_sinrs'][link_id] = self.phy_interface.get_effective_sinr(link_id)
        
        return info
    
    def render(self, mode: str = 'human'):
        """Render the environment (optional, for visualization)."""
        if mode == 'human':
            print(f"\n=== LEO-ISAC Environment State ===")
            print(f"Time: {self.current_time:.1f} s")
            print(f"Step: {self.episode_step}/{self.isac_config.episode_length}")
            print(f"Active Links: {len(self.active_links)}")
            print(f"GDOP: {self.phy_interface.get_network_gdop():.2f} m")
            
            # Print satellite positions
            for sat_id, sat in self.constellation.satellites.items():
                pos = sat.current_state.position / 1000  # Convert to km
                print(f"  {sat_id}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] km")
    
    def close(self):
        """Clean up environment resources."""
        pass


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    """Test the refactored environment with physical layer interface."""
    print("=" * 60)
    print("Testing Refactored LEO-ISAC Environment")
    print("=" * 60)
    
    # Create environment with default settings
    env = LEO_ISAC_Env()
    
    # Reset
    obs = env.reset()
    print(f"✓ Reset successful, {env.n_agents} agents initialized")
    
    # Test step with random actions
    actions = {}
    for agent_id in env.agent_ids:
        power_alloc = {}
        for link_id in env.link_registry:
            if env.link_registry[link_id][0] == agent_id:  # If agent is transmitter
                power_alloc[link_id] = np.random.uniform(0, env.max_tx_power_w/4)
        
        actions[agent_id] = {
            'power_allocation': power_alloc,
            'beam_selection': {}
        }
    
    # Execute step
    obs_next, rewards, done, info = env.step(actions)
    
    print(f"✓ Step successful")
    print(f"  Rewards: {list(rewards.values())}")
    print(f"  Throughput: {info['total_throughput']:.2f} Gbps")
    print(f"  GDOP: {info['gdop']:.2f} m")
    
    # Run a few more steps
    for _ in range(5):
        obs_next, rewards, done, info = env.step(actions)
    
    print(f"✓ Multiple steps successful")
    
    env.close()
    print("\n✓ Refactored environment with physical layer interface works!")