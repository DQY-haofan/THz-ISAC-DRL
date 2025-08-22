"""
LEO-ISAC Multi-Agent Reinforcement Learning Environment
========================================================

This module provides a Gym-like environment for multi-agent reinforcement learning
in LEO satellite constellations with integrated sensing and communications (ISAC).
It encapsulates the high-fidelity physical models into a unified interface for
MARL agent training.

Based on the MMDP formalization from "Network LEO-ISAC Resource Allocation as MMDP"
and integrates all physical layer models (geometry, hardware, interference, FIM).

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import warnings
from collections import defaultdict

# Import physical layer modules
from geom import (
    Constellation, Satellite, StateVector, OrbitalElements,
    calculate_geometry, check_earth_obstruction, get_visibility_graph
)
from hardware import (
    get_hardware_params, calculate_channel_gain, calculate_effective_sinr,
    calculate_noise_power, calculate_antenna_gain, calculate_beamwidth
)
from interference import (
    LinkParameters, calculate_alpha_lm, calculate_network_interference,
    calculate_normalized_interference_coeff
)
from performance_model import (
    calculate_effective_sinr as calc_sinr_eff,
    calculate_range_variance, PerformanceMetrics
)
from fim import (
    InformationFilter, build_jacobian, update_info, calculate_efim,
    get_performance_metrics, create_state_transition_matrix, 
    create_process_noise_covariance
)


@dataclass
class ConstellationConfig:
    """Configuration for LEO satellite constellation."""
    n_satellites: int = 4
    altitude_km: float = 550.0  # Altitude above Earth surface
    inclination_deg: float = 53.0  # Orbital inclination
    propagation_model: str = "keplerian"
    
    def create_walker_constellation(self) -> List[Satellite]:
        """Create a Walker constellation with evenly spaced satellites."""
        satellites = []
        altitude_m = self.altitude_km * 1000  # Convert to meters
        semi_major_axis = 6371000 + altitude_m  # Earth radius + altitude
        
        for i in range(self.n_satellites):
            # Evenly distribute satellites in a single orbital plane
            mean_anomaly = (2 * np.pi * i) / self.n_satellites
            
            # Create orbital elements
            elements = OrbitalElements(
                a=semi_major_axis,
                e=0.001,  # Nearly circular orbit
                i=np.deg2rad(self.inclination_deg),
                raan=0.0,  # Single plane for simplicity
                omega=0.0,
                nu=mean_anomaly
            )
            
            # Create satellite
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
    frequency_ghz: float = 300.0  # Carrier frequency in GHz
    bandwidth_ghz: float = 10.0   # Signal bandwidth in GHz
    antenna_diameter_m: float = 0.5  # Antenna diameter
    tx_power_max_dbm: float = 30.0  # Maximum transmit power in dBm
    noise_figure_db: float = 3.0  # Receiver noise figure
    hardware_level: str = "High-Performance"  # Hardware quality level
    max_isl_range_km: float = 5000.0  # Maximum ISL range


@dataclass
class ISACConfig:
    """ISAC task configuration and reward weights."""
    # Reward weights
    w_comm: float = 1.0   # Communication reward weight
    w_sens: float = 0.5   # Sensing reward weight  
    w_penalty: float = 10.0  # Constraint penalty weight
    
    # Task parameters
    sensing_mode: str = "cooperative_orbit_determination"  # or "debris_tracking"
    min_links_for_sensing: int = 3  # Minimum ISLs for 3D observability
    
    # Simulation parameters
    time_step_s: float = 1.0  # Simulation time step
    episode_length: int = 100  # Steps per episode


class AgentObservation:
    """Container for a single agent's local observation."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.channel_gains = {}  # Link channel gains
        self.buffer_lengths = {}  # Data buffer queues
        self.total_interference = 0.0  # Aggregate interference
        self.neighbor_powers = {}  # Neighbor transmit powers
        self.position = np.zeros(3)  # Own position (m)
        self.velocity = np.zeros(3)  # Own velocity (m/s)
        self.available_power = 0.0  # Power budget (W)
        
    def to_vector(self) -> np.ndarray:
        """Convert observation to fixed-size vector for neural network input."""
        # This is a simplified version - in practice, need careful padding/normalization
        obs_list = []
        
        # Channel gains (pad to max expected links)
        gains = list(self.channel_gains.values())[:4]  # Max 4 links
        gains.extend([0.0] * (4 - len(gains)))  # Pad if fewer
        obs_list.extend(gains)
        
        # Buffer lengths
        buffers = list(self.buffer_lengths.values())[:4]
        buffers.extend([0.0] * (4 - len(buffers)))
        obs_list.extend(buffers)
        
        # Scalar values
        obs_list.append(self.total_interference)
        obs_list.append(self.available_power)
        
        # Position and velocity
        obs_list.extend(self.position / 1e6)  # Normalize to Mm
        obs_list.extend(self.velocity / 1e3)  # Normalize to km/s
        
        return np.array(obs_list, dtype=np.float32)


class LEO_ISAC_Env:
    """
    Multi-agent LEO-ISAC environment for reinforcement learning.
    
    This environment simulates a LEO satellite constellation performing
    joint communication and sensing tasks. Each satellite is an independent
    agent that must learn to allocate resources (power, beams) to optimize
    both communication throughput and network sensing performance.
    
    The environment follows OpenAI Gym conventions with extensions for
    multi-agent scenarios.
    """
    
    def __init__(self, 
                 constellation_config: Optional[ConstellationConfig] = None,
                 hardware_config: Optional[HardwareConfig] = None,
                 isac_config: Optional[ISACConfig] = None):
        """
        Initialize the LEO-ISAC environment.
        
        Args:
            constellation_config: Constellation parameters
            hardware_config: Hardware and system parameters
            isac_config: ISAC task and reward configuration
        """
        # Use defaults if not provided
        self.const_config = constellation_config or ConstellationConfig()
        self.hw_config = hardware_config or HardwareConfig()
        self.isac_config = isac_config or ISACConfig()
        
        # Convert units to SI
        self.frequency_hz = self.hw_config.frequency_ghz * 1e9
        self.bandwidth_hz = self.hw_config.bandwidth_ghz * 1e9
        self.max_tx_power_w = 10**((self.hw_config.tx_power_max_dbm - 30) / 10)
        self.max_isl_range_m = self.hw_config.max_isl_range_km * 1000
        
        # Initialize constellation
        satellites = self.const_config.create_walker_constellation()
        self.constellation = Constellation(satellites)
        self.n_agents = len(satellites)
        self.agent_ids = [sat.sat_id for sat in satellites]
        
        # Hardware parameters
        self.hw_profile = get_hardware_params(self.hw_config.hardware_level)
        
        # Antenna parameters
        self.antenna_gain = calculate_antenna_gain(
            self.frequency_hz, 
            self.hw_config.antenna_diameter_m
        )
        self.beamwidth = calculate_beamwidth(
            self.frequency_hz,
            self.hw_config.antenna_diameter_m
        )
        
        # Noise power
        self.noise_power = calculate_noise_power(
            self.bandwidth_hz,
            self.hw_config.noise_figure_db
        )
        
        # Information filter for sensing
        self.info_filter = InformationFilter(
            n_states_per_sat=8,
            n_satellites=self.n_agents
        )
        
        # State tracking
        self.current_time = 0.0
        self.episode_step = 0
        self.active_links = set()
        self.link_states = {}  # Store link-specific states
        
        # Action history for neighbor information
        self.prev_actions = {agent_id: {} for agent_id in self.agent_ids}
        
        # Data buffers (simulated traffic)
        self.data_buffers = defaultdict(lambda: defaultdict(float))
        
        # FIM for sensing performance
        self.network_fim = np.zeros((
            self.info_filter.n_states_total,
            self.info_filter.n_states_total
        ))
        
        # Performance metrics tracking
        self.metrics_history = []
        
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to initial state.
        
        Returns:
            Dictionary mapping agent IDs to their initial observations
        """
        # Reset time and counters
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
        
        # Initialize data buffers with random traffic
        self._reset_data_buffers()
        
        # Reset FIM
        self.network_fim = np.eye(self.info_filter.n_states_total) * 0.1
        
        # Reset action history
        self.prev_actions = {agent_id: {} for agent_id in self.agent_ids}
        
        # Clear metrics history
        self.metrics_history = []
        
        # Compute initial observations
        observations = self._compute_observations()
        
        return observations
    
    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute one environment step with joint actions from all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions.
                    Each action is a dict with keys:
                    - 'power_allocation': Dict[link_id, float] - Power per link (W)
                    - 'beam_selection': Dict[link_id, int] - Beam index per link
        
        Returns:
            Tuple of (observations, rewards, done, info):
            - observations: Next observations for all agents
            - rewards: Rewards for all agents
            - done: Whether episode is finished
            - info: Additional information dictionary
        """
        # Validate and store actions
        self._validate_actions(actions)
        self.prev_actions = actions.copy()
        
        # Apply actions to update link states
        self._apply_actions(actions)
        
        # Propagate orbital dynamics
        self.current_time += self.isac_config.time_step_s
        self.constellation.propagate_all(self.current_time)
        
        # Update network topology after propagation
        self._update_topology()
        
        # Compute performance metrics
        link_metrics = self._compute_link_metrics(actions)
        
        # Calculate rewards based on ISAC reward function
        rewards = self._compute_rewards(actions, link_metrics)
        
        # Update FIM with measurements
        self._update_network_fim(link_metrics)
        
        # Update data buffers
        self._update_data_buffers(link_metrics)
        
        # Compute new observations
        observations = self._compute_observations()
        
        # Check termination
        self.episode_step += 1
        done = (self.episode_step >= self.isac_config.episode_length)
        
        # Collect info
        info = self._collect_info(link_metrics)
        
        return observations, rewards, done, info
    
    def _update_topology(self):
        """Update network topology based on current satellite positions."""
        self.active_links = get_visibility_graph(
            self.constellation,
            self.current_time,
            max_range=self.max_isl_range_m,
            check_obstruction=True
        )
        
        # Create bidirectional link registry
        self.link_states = {}
        for (sat_i, sat_j) in self.active_links:
            # Forward link
            link_id_fwd = f"{sat_i}->{sat_j}"
            self.link_states[link_id_fwd] = {
                'tx': sat_i,
                'rx': sat_j,
                'active': True
            }
            # Reverse link
            link_id_rev = f"{sat_j}->{sat_i}"
            self.link_states[link_id_rev] = {
                'tx': sat_j,
                'rx': sat_i,
                'active': True
            }
    
    def _compute_observations(self) -> Dict[str, np.ndarray]:
        """Compute local observations for all agents."""
        observations = {}
        
        for agent_id in self.agent_ids:
            obs = AgentObservation(agent_id)
            sat = self.constellation.satellites[agent_id]
            
            # Own state
            obs.position = sat.current_state.position
            obs.velocity = sat.current_state.velocity
            obs.available_power = self.max_tx_power_w
            
            # Outgoing link information
            for link_id, link_state in self.link_states.items():
                if link_state['tx'] == agent_id:
                    rx_id = link_state['rx']
                    rx_sat = self.constellation.satellites[rx_id]
                    
                    # Calculate channel gain
                    geometry = calculate_geometry(sat, rx_sat)
                    channel_gain = calculate_channel_gain(
                        geometry['distance'],
                        self.frequency_hz,
                        self.antenna_gain,
                        self.antenna_gain,
                        self.hw_profile.sigma_e,
                        self.beamwidth
                    )
                    obs.channel_gains[link_id] = channel_gain
                    
                    # Buffer length (simulated)
                    obs.buffer_lengths[link_id] = self.data_buffers[agent_id][rx_id]
            
            # Compute total interference (simplified - aggregate from all other transmitters)
            obs.total_interference = self._estimate_interference(agent_id)
            
            # Neighbor power information from previous step
            for neighbor_id in self._get_neighbors(agent_id):
                if neighbor_id in self.prev_actions:
                    total_power = sum(self.prev_actions[neighbor_id].get(
                        'power_allocation', {}
                    ).values())
                    obs.neighbor_powers[neighbor_id] = total_power
            
            observations[agent_id] = obs.to_vector()
        
        return observations
    
    def _compute_link_metrics(self, actions: Dict) -> Dict[str, Dict]:
        """Compute performance metrics for all active links."""
        link_metrics = {}
        
        # First pass: compute individual link SNRs
        for link_id, link_state in self.link_states.items():
            if not link_state['active']:
                continue
                
            tx_id = link_state['tx']
            rx_id = link_state['rx']
            
            # Get action for this link
            if tx_id not in actions:
                continue
            
            tx_power = actions[tx_id].get('power_allocation', {}).get(link_id, 0.0)
            
            if tx_power <= 0:
                continue
            
            # Get satellites
            tx_sat = self.constellation.satellites[tx_id]
            rx_sat = self.constellation.satellites[rx_id]
            
            # Calculate geometry and channel
            geometry = calculate_geometry(tx_sat, rx_sat)
            channel_gain = calculate_channel_gain(
                geometry['distance'],
                self.frequency_hz,
                self.antenna_gain,
                self.antenna_gain,
                self.hw_profile.sigma_e,
                self.beamwidth
            )
            
            # Pre-impairment SNR
            snr0 = (tx_power * channel_gain) / self.noise_power
            
            link_metrics[link_id] = {
                'tx_id': tx_id,
                'rx_id': rx_id,
                'tx_power': tx_power,
                'channel_gain': channel_gain,
                'distance': geometry['distance'],
                'snr0': snr0
            }
        
        # Second pass: compute interference and effective SINR
        for link_id, metrics in link_metrics.items():
            # Calculate interference from other links
            interference = self._calculate_link_interference(link_id, link_metrics)
            
            # Normalized interference
            normalized_interference = interference / self.noise_power
            
            # Effective SINR with hardware impairments
            sinr_eff = calc_sinr_eff(
                metrics['snr0'],
                self.hw_profile.gamma_eff,
                self.hw_profile.sigma_phi_squared,
                normalized_interference
            )
            
            metrics['interference'] = interference
            metrics['sinr_eff'] = sinr_eff
            
            # Range measurement variance for sensing
            range_var = calculate_range_variance(
                sinr_eff,
                self.hw_profile.sigma_phi_squared,
                self.frequency_hz,
                bandwidth=self.bandwidth_hz
            )
            metrics['range_variance'] = range_var
        
        return link_metrics
    
    def _compute_rewards(self, actions: Dict, link_metrics: Dict) -> Dict[str, float]:
        """
        Compute ISAC rewards for all agents based on the designed reward function.
        
        Implements: r_i = w_C * R_comm + w_S * R_sens - w_P * R_penalty
        """
        rewards = {}
        
        for agent_id in self.agent_ids:
            # Initialize reward components
            r_comm = 0.0  # Communication reward
            r_sens = 0.0  # Sensing reward
            r_penalty = 0.0  # Penalty for constraint violation
            
            # Get agent's actions
            agent_action = actions.get(agent_id, {})
            power_allocation = agent_action.get('power_allocation', {})
            
            # Communication reward: sum of log spectral efficiencies
            for link_id, tx_power in power_allocation.items():
                if link_id in link_metrics:
                    sinr_eff = link_metrics[link_id]['sinr_eff']
                    # Log spectral efficiency
                    r_comm += np.log(1 + sinr_eff)
            
            # Sensing reward: sum of FIM trace contributions
            for link_id, tx_power in power_allocation.items():
                if link_id in link_metrics:
                    # Build Jacobian for this measurement
                    tx_idx = self.agent_ids.index(
                        link_metrics[link_id]['tx_id']
                    )
                    rx_idx = self.agent_ids.index(
                        link_metrics[link_id]['rx_id']
                    )
                    
                    # Get current network state
                    network_state = self.constellation.get_state_matrix().flatten()
                    
                    # Build Jacobian
                    H = build_jacobian(tx_idx, rx_idx, network_state)
                    
                    # FIM contribution (simplified - using scalar measurement)
                    range_var = link_metrics[link_id]['range_variance']
                    if range_var > 0:
                        # Convert to time variance for FIM
                        toa_var = range_var / (299792458.0**2)
                        J_link = (H.T @ H) / toa_var
                        # Trace of FIM contribution
                        r_sens += np.trace(J_link)
            
            # Penalty for power constraint violation
            total_power = sum(power_allocation.values())
            if total_power > self.max_tx_power_w:
                r_penalty = self.isac_config.w_penalty * (
                    total_power - self.max_tx_power_w
                ) / self.max_tx_power_w
            
            # Combined ISAC reward
            rewards[agent_id] = (
                self.isac_config.w_comm * r_comm +
                self.isac_config.w_sens * r_sens -
                r_penalty
            )
        
        return rewards
    
    def _calculate_link_interference(self, victim_link_id: str, 
                                     all_link_metrics: Dict) -> float:
        """Calculate total interference on a victim link from all other links."""
        if victim_link_id not in self.link_states:
            return 0.0
        
        victim_rx_id = self.link_states[victim_link_id]['rx']
        victim_rx = self.constellation.satellites[victim_rx_id]
        
        total_interference = 0.0
        
        # Sum interference from all other active transmissions
        for interferer_link_id, interferer_metrics in all_link_metrics.items():
            if interferer_link_id == victim_link_id:
                continue
            
            interferer_tx_id = interferer_metrics['tx_id']
            if interferer_tx_id == victim_rx_id:
                continue  # Don't interfere with own transmissions
            
            interferer_tx = self.constellation.satellites[interferer_tx_id]
            
            # Calculate interference path geometry
            geometry = calculate_geometry(interferer_tx, victim_rx)
            
            # Simplified interference calculation
            interference_channel_gain = calculate_channel_gain(
                geometry['distance'],
                self.frequency_hz,
                self.antenna_gain,
                self.antenna_gain,
                self.hw_profile.sigma_e,
                self.beamwidth
            )
            
            # Add interference power
            total_interference += (
                interferer_metrics['tx_power'] * interference_channel_gain
            )
        
        return total_interference
    
    def _estimate_interference(self, agent_id: str) -> float:
        """Estimate total interference experienced by an agent (simplified)."""
        # This is a simplified aggregate interference estimate
        # In practice, would compute per-link interference more carefully
        
        agent_sat = self.constellation.satellites[agent_id]
        total_interference = 0.0
        
        for other_id in self.agent_ids:
            if other_id == agent_id:
                continue
            
            # Get previous transmit power (if available)
            if other_id in self.prev_actions:
                other_power = sum(
                    self.prev_actions[other_id].get('power_allocation', {}).values()
                )
                
                if other_power > 0:
                    other_sat = self.constellation.satellites[other_id]
                    geometry = calculate_geometry(other_sat, agent_sat)
                    
                    # Rough interference estimate
                    if geometry['distance'] < self.max_isl_range_m:
                        channel_gain = (3e8 / self.frequency_hz) ** 2 / (
                            (4 * np.pi * geometry['distance']) ** 2
                        )
                        total_interference += other_power * channel_gain * 0.01
        
        return total_interference
    
    def _get_neighbors(self, agent_id: str) -> List[str]:
        """Get one-hop neighbors of an agent."""
        neighbors = set()
        
        for link_id, link_state in self.link_states.items():
            if link_state['tx'] == agent_id:
                neighbors.add(link_state['rx'])
            elif link_state['rx'] == agent_id:
                neighbors.add(link_state['tx'])
        
        return list(neighbors)
    
    def _update_network_fim(self, link_metrics: Dict):
        """Update network FIM with new measurements."""
        # Simplified FIM update - in practice would use full recursive filtering
        
        # Create measurement Jacobian list
        H_list = []
        R_list = []
        
        for link_id, metrics in link_metrics.items():
            if metrics['sinr_eff'] <= 0:
                continue
            
            tx_idx = self.agent_ids.index(metrics['tx_id'])
            rx_idx = self.agent_ids.index(metrics['rx_id'])
            
            network_state = self.constellation.get_state_matrix().flatten()
            H = build_jacobian(tx_idx, rx_idx, network_state)
            
            H_list.append(H)
            R_list.append(metrics['range_variance'])
        
        # Update FIM (simplified - just accumulate information)
        for H, R in zip(H_list, R_list):
            if R > 0:
                toa_var = R / (299792458.0**2)
                self.network_fim += (H.T @ H) / toa_var
    
    def _reset_data_buffers(self):
        """Initialize data buffers with random traffic demand."""
        self.data_buffers.clear()
        
        for tx_id in self.agent_ids:
            for rx_id in self.agent_ids:
                if tx_id != rx_id:
                    # Random initial buffer (0-100 Mb)
                    self.data_buffers[tx_id][rx_id] = np.random.uniform(0, 100)
    
    def _update_data_buffers(self, link_metrics: Dict):
        """Update data buffers based on transmission rates."""
        # Generate new traffic
        for tx_id in self.agent_ids:
            for rx_id in self.agent_ids:
                if tx_id != rx_id:
                    # Add random traffic (0-10 Mb/s)
                    self.data_buffers[tx_id][rx_id] += np.random.uniform(0, 10)
        
        # Drain buffers based on achieved rates
        for link_id, metrics in link_metrics.items():
            if metrics['sinr_eff'] > 0:
                tx_id = metrics['tx_id']
                rx_id = metrics['rx_id']
                
                # Shannon capacity in bits/s
                rate = self.bandwidth_hz * np.log2(1 + metrics['sinr_eff'])
                
                # Drain buffer (convert to Mb)
                drained = min(
                    rate * self.isac_config.time_step_s / 1e6,
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
    
    def _apply_actions(self, actions: Dict):
        """Apply agent actions to update system state."""
        # Actions primarily affect power allocation and beam selection
        # These are already captured in link_metrics computation
        # This method is a placeholder for any additional state updates
        pass
    
    def _collect_info(self, link_metrics: Dict) -> Dict:
        """Collect additional information for debugging and analysis."""
        info = {
            'time': self.current_time,
            'step': self.episode_step,
            'n_active_links': len([m for m in link_metrics.values() 
                                   if m['sinr_eff'] > 0]),
            'total_throughput': sum([
                self.bandwidth_hz * np.log2(1 + m['sinr_eff']) 
                for m in link_metrics.values() if m['sinr_eff'] > 0
            ]) / 1e9,  # Gbps
        }
        
        # Add sensing performance metrics
        try:
            efim = calculate_efim(
                self.network_fim,
                self.info_filter.kinematic_indices,
                self.info_filter.clock_indices
            )
            metrics = get_performance_metrics(efim, self.n_agents)
            info['gdop'] = metrics.get('GDOP', np.inf)
            info['fim_rank'] = metrics.get('rank', 0)
        except:
            info['gdop'] = np.inf
            info['fim_rank'] = 0
        
        return info
    
    def render(self, mode: str = 'human'):
        """Render the environment (optional, for visualization)."""
        if mode == 'human':
            print(f"\n=== LEO-ISAC Environment State ===")
            print(f"Time: {self.current_time:.1f} s")
            print(f"Episode Step: {self.episode_step}/{self.isac_config.episode_length}")
            print(f"Active Links: {len(self.active_links)}")
            
            # Print satellite positions
            for sat_id, sat in self.constellation.satellites.items():
                pos = sat.current_state.position / 1000  # Convert to km
                print(f"  {sat_id}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] km")
    
    def close(self):
        """Clean up environment resources."""
        pass


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_env() -> LEO_ISAC_Env:
    """Create a LEO-ISAC environment with default settings."""
    const_config = ConstellationConfig(
        n_satellites=4,
        altitude_km=550,
        inclination_deg=53
    )
    
    hw_config = HardwareConfig(
        frequency_ghz=300,
        bandwidth_ghz=10,
        antenna_diameter_m=0.5,
        tx_power_max_dbm=30,
        hardware_level="High-Performance"
    )
    
    isac_config = ISACConfig(
        w_comm=1.0,
        w_sens=0.5,
        w_penalty=10.0,
        time_step_s=1.0,
        episode_length=100
    )
    
    return LEO_ISAC_Env(const_config, hw_config, isac_config)


# ============================================================================
# Unit Tests
# ============================================================================

def test_environment_basic():
    """Test basic environment functionality."""
    print("Testing LEO-ISAC Environment...")
    
    # Create environment
    env = create_default_env()
    
    # Test reset
    obs = env.reset()
    assert len(obs) == env.n_agents, "Should have observations for all agents"
    print(f"✓ Reset successful, {env.n_agents} agents initialized")
    
    # Test step with random actions
    actions = {}
    for agent_id in env.agent_ids:
        # Random power allocation
        power_alloc = {}
        for link_id in env.link_states:
            if env.link_states[link_id]['tx'] == agent_id:
                power_alloc[link_id] = np.random.uniform(0, env.max_tx_power_w/4)
        
        actions[agent_id] = {
            'power_allocation': power_alloc,
            'beam_selection': {}
        }
    
    # Execute step
    obs_next, rewards, done, info = env.step(actions)
    
    assert len(obs_next) == env.n_agents, "Should have next observations"
    assert len(rewards) == env.n_agents, "Should have rewards for all agents"
    print(f"✓ Step successful, rewards: {list(rewards.values())}")
    print(f"✓ Info: {info}")
    
    # Test multiple steps
    for _ in range(5):
        obs_next, rewards, done, info = env.step(actions)
    
    print(f"✓ Multiple steps successful")
    
    env.close()
    print("✓ Environment tests passed")


def test_reward_computation():
    """Test ISAC reward function computation."""
    print("\nTesting ISAC Reward Computation...")
    
    env = create_default_env()
    obs = env.reset()
    
    # Create specific test actions
    actions = {}
    for i, agent_id in enumerate(env.agent_ids):
        power_alloc = {}
        
        # Allocate different power levels to test reward response
        for link_id in env.link_states:
            if env.link_states[link_id]['tx'] == agent_id:
                # Vary power allocation
                power_alloc[link_id] = env.max_tx_power_w * (0.1 + 0.2 * i)
        
        actions[agent_id] = {
            'power_allocation': power_alloc,
            'beam_selection': {}
        }
    
    # Get rewards
    obs_next, rewards, done, info = env.step(actions)
    
    print("Agent Rewards:")
    for agent_id, reward in rewards.items():
        print(f"  {agent_id}: {reward:.4f}")
    
    # Test penalty for power violation
    actions_violation = {}
    for agent_id in env.agent_ids:
        power_alloc = {}
        for link_id in env.link_states:
            if env.link_states[link_id]['tx'] == agent_id:
                # Exceed power budget
                power_alloc[link_id] = env.max_tx_power_w
        
        actions_violation[agent_id] = {
            'power_allocation': power_alloc,
            'beam_selection': {}
        }
    
    obs_next, rewards_penalty, done, info = env.step(actions_violation)
    
    print("\nRewards with power violation:")
    for agent_id, reward in rewards_penalty.items():
        print(f"  {agent_id}: {reward:.4f} (should be negative)")
    
    env.close()
    print("✓ Reward computation tests passed")


if __name__ == "__main__":
    """Run tests and demonstrate environment usage."""
    print("=" * 60)
    print("LEO-ISAC Multi-Agent Environment Tests")
    print("=" * 60)
    
    test_environment_basic()
    test_reward_computation()
    
    print("\n" + "=" * 60)
    print("Environment Demo: Running a Complete Episode")
    print("=" * 60)
    
    # Create environment
    env = create_default_env()
    obs = env.reset()
    
    print(f"\nInitial Network State:")
    print(f"  Satellites: {env.n_agents}")
    print(f"  Active ISLs: {len(env.active_links)}")
    
    total_rewards = {agent_id: 0.0 for agent_id in env.agent_ids}
    
    # Run episode
    for step in range(10):  # Short demo
        # Simple heuristic policy for demonstration
        actions = {}
        for agent_id in env.agent_ids:
            power_alloc = {}
            
            # Allocate equal power to all outgoing links
            outgoing_links = [
                lid for lid, ls in env.link_states.items()
                if ls['tx'] == agent_id
            ]
            
            if outgoing_links:
                power_per_link = env.max_tx_power_w * 0.8 / len(outgoing_links)
                for link_id in outgoing_links:
                    power_alloc[link_id] = power_per_link
            
            actions[agent_id] = {
                'power_allocation': power_alloc,
                'beam_selection': {}
            }
        
        # Step
        obs, rewards, done, info = env.step(actions)
        
        # Accumulate rewards
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward
        
        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"  Throughput: {info['total_throughput']:.2f} Gbps")
            print(f"  GDOP: {info['gdop']:.2f} m")
            print(f"  Mean reward: {np.mean(list(rewards.values())):.4f}")
    
    print("\n" + "=" * 60)
    print("Episode Summary:")
    print("=" * 60)
    print("Total Rewards by Agent:")
    for agent_id, total in total_rewards.items():
        print(f"  {agent_id}: {total:.2f}")
    print(f"Average: {np.mean(list(total_rewards.values())):.2f}")
    
    env.close()
    print("\n✓ Environment demonstration complete!")