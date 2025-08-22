"""
Physical Layer Interface for THz LEO-ISL ISAC Network
======================================================

This module provides a unified interface that encapsulates all physical layer
computations, serving as a "black-box oracle" between the MARL framework and
the complex physical models. It abstracts away the details of channel modeling,
hardware impairments, interference calculations, and Fisher Information Matrix
operations.

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import warnings

# Import all physical layer modules
from geom import (
    Constellation, Satellite, StateVector,
    calculate_geometry, check_earth_obstruction
)
from hardware import (
    get_hardware_params, calculate_channel_gain,
    calculate_antenna_gain, calculate_beamwidth,
    calculate_noise_power
)
from interference import (
    LinkParameters, calculate_alpha_lm,
    calculate_normalized_interference_coeff
)
from performance_model import (
    calculate_effective_sinr, calculate_range_variance_m2
)
from fim import (
    InformationFilter, build_jacobian, update_info,
    calculate_efim, get_performance_metrics,
    create_state_transition_matrix, create_process_noise_covariance
)
from ioo import (
    calculate_bistatic_geometry, calculate_sinr_ioo,
    calculate_j_ioo, calculate_bistatic_measurement_variance
)


class PhysicalLayerInterface:
    """
    Unified interface for all physical layer computations in THz LEO-ISL ISAC.
    
    This class serves as the single point of contact between the MARL environment
    and the complex physical layer models. It encapsulates channel modeling,
    hardware impairments, interference analysis, and sensing performance metrics.
    
    Key Design Principles:
    - Separation of Concerns: Decouples physical modeling from MARL logic
    - Caching: Stores intermediate results to avoid redundant computations
    - Unit Consistency: Handles all unit conversions internally
    - Extensibility: New physical effects can be added without changing the API
    """
    
    def __init__(self, 
                 constellation_config: Any,
                 hardware_config: Any,
                 isac_config: Any):
        """
        Initialize the physical layer interface with static configuration.
        
        Args:
            constellation_config: Constellation parameters (n_satellites, altitude, etc.)
            hardware_config: Hardware specifications (frequency, bandwidth, power, etc.)
            isac_config: ISAC parameters (reward weights, sensing modes, etc.)
        """
        # Store configurations
        self.const_config = constellation_config
        self.hw_config = hardware_config
        self.isac_config = isac_config
        
        # Convert units to SI
        self.frequency_hz = hardware_config.frequency_ghz * 1e9
        self.bandwidth_hz = hardware_config.bandwidth_ghz * 1e9
        self.max_tx_power_w = 10**((hardware_config.tx_power_max_dbm - 30) / 10)
        
        # Initialize hardware profile
        self.hw_profile = get_hardware_params(hardware_config.hardware_level)
        
        # Calculate antenna parameters
        self.antenna_gain = calculate_antenna_gain(
            self.frequency_hz,
            hardware_config.antenna_diameter_m,
            efficiency=0.55
        )
        self.beamwidth = calculate_beamwidth(
            self.frequency_hz,
            hardware_config.antenna_diameter_m
        )
        
        # Calculate noise power
        self.noise_power = calculate_noise_power(
            self.bandwidth_hz,
            hardware_config.noise_figure_db
        )
        
        # Initialize Information Filter for sensing
        self.info_filter = InformationFilter(
            n_states_per_sat=8,
            n_satellites=constellation_config.n_satellites
        )
        
        # Cache for computed results
        self._cache = {}
        self._cache_valid = False
        
        # State variables (will be updated dynamically)
        self.current_time = 0.0
        self.satellite_states = None
        self.active_links = set()
        self.power_allocations = {}
        
        # Cached physical metrics
        self.link_metrics = {}
        self.interference_matrix = {}
        self.network_fim = None
        self.efim = None
        
    def update_dynamic_state(self,
                            current_time: float,
                            satellite_states: np.ndarray,
                            active_links: Set[Tuple[str, str]],
                            power_allocations: Dict[str, Dict[str, float]]):
        """
        Update the interface with current dynamic network state.
        
        This method should be called at the beginning of each time step to
        inject current network state. It triggers all necessary computations
        and caches results for subsequent queries.
        
        Args:
            current_time: Current simulation time (seconds)
            satellite_states: Complete state matrix (8N_v x 1) with positions, 
                            velocities, clock biases and drifts
            active_links: Set of currently active ISL links as (tx_id, rx_id) tuples
            power_allocations: Power allocation per agent and link
                              {agent_id: {link_id: power_w}}
        """
        # Invalidate cache when state changes
        self._cache_valid = False
        self._cache.clear()
        
        # Store current state
        self.current_time = current_time
        self.satellite_states = satellite_states
        self.active_links = active_links
        self.power_allocations = power_allocations
        
        # Trigger comprehensive physical layer computation
        self._compute_all_metrics()
        
        # Mark cache as valid
        self._cache_valid = True
        
    def get_effective_sinr(self, link_id: str) -> float:
        """
        Get the effective SINR for a specific link.
        
        Args:
            link_id: Link identifier (e.g., "SAT_0->SAT_1")
            
        Returns:
            Effective SINR (linear scale) incorporating all impairments
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return 0.0
            
        if link_id not in self.link_metrics:
            return 0.0
            
        return self.link_metrics[link_id].get('sinr_eff', 0.0)
    
    def get_range_variance_m2(self, link_id: str) -> float:
        """
        Get the range measurement variance for a specific link.
        
        Args:
            link_id: Link identifier
            
        Returns:
            Range measurement variance in m² (NOT s²)
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return np.inf
            
        if link_id not in self.link_metrics:
            return np.inf
            
        return self.link_metrics[link_id].get('range_variance', np.inf)
    
    def get_fim_contribution(self, link_id: str) -> np.ndarray:
        """
        Get the Fisher Information Matrix contribution from a specific link.
        
        Args:
            link_id: Link identifier
            
        Returns:
            FIM contribution J_ℓ (8N_v x 8N_v sparse matrix)
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return np.zeros((self.info_filter.n_states_total, 
                           self.info_filter.n_states_total))
            
        if link_id not in self.link_metrics:
            return np.zeros((self.info_filter.n_states_total,
                           self.info_filter.n_states_total))
            
        metrics = self.link_metrics[link_id]
        
        # Build Jacobian for this link
        tx_idx = self._get_satellite_index(metrics['tx_id'])
        rx_idx = self._get_satellite_index(metrics['rx_id'])
        
        H = build_jacobian(tx_idx, rx_idx, self.satellite_states)
        
        # Get measurement variance in s² (convert from m²)
        range_var_m2 = metrics['range_variance']
        toa_var_s2 = range_var_m2 / (299792458.0**2)
        
        # Compute FIM contribution
        J_link = H.T @ H / toa_var_s2
        
        return J_link
    
    def get_total_comm_reward(self, agent_id: str) -> float:
        """
        Calculate total communication reward for an agent.
        
        Implements: R_comm = Σ_ℓ log(1 + SINR_eff,ℓ)
        
        Args:
            agent_id: Agent/satellite identifier
            
        Returns:
            Total communication reward (sum of log spectral efficiencies)
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return 0.0
            
        total_reward = 0.0
        
        # Sum over all links where agent is transmitter
        for link_id, metrics in self.link_metrics.items():
            if metrics['tx_id'] == agent_id:
                sinr_eff = metrics.get('sinr_eff', 0.0)
                if sinr_eff > 0:
                    total_reward += np.log(1 + sinr_eff)
                    
        return total_reward
    
    def get_total_sens_reward(self, agent_id: str) -> float:
        """
        Calculate total sensing reward for an agent.
        
        Based on Fisher Information contribution to network observability.
        
        Args:
            agent_id: Agent/satellite identifier
            
        Returns:
            Total sensing reward (trace of FIM contributions)
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return 0.0
            
        total_reward = 0.0
        
        # Sum FIM trace contributions from agent's links
        for link_id, metrics in self.link_metrics.items():
            if metrics['tx_id'] == agent_id:
                J_link = self.get_fim_contribution(link_id)
                total_reward += np.trace(J_link)
                
        return total_reward
    
    def get_network_gdop(self) -> float:
        """
        Get the current network Geometric Dilution of Precision.
        
        Returns:
            GDOP value in meters, or inf if network is unobservable
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return np.inf
            
        if self.efim is None:
            return np.inf
            
        # Extract performance metrics from EFIM
        metrics = get_performance_metrics(
            self.efim,
            n_satellites=self.const_config.n_satellites
        )
        
        return metrics.get('GDOP', np.inf)
    
    def get_throughput_gbps(self, link_id: str) -> float:
        """
        Get the achievable throughput for a specific link.
        
        Args:
            link_id: Link identifier
            
        Returns:
            Throughput in Gbps based on Shannon capacity
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return 0.0
            
        if link_id not in self.link_metrics:
            return 0.0
            
        sinr_eff = self.link_metrics[link_id].get('sinr_eff', 0.0)
        if sinr_eff <= 0:
            return 0.0
            
        # Shannon capacity
        capacity_bps = self.bandwidth_hz * np.log2(1 + sinr_eff)
        return capacity_bps / 1e9  # Convert to Gbps
    
    def get_interference_coefficient(self, victim_link: str, 
                                    interferer_link: str) -> float:
        """
        Get the interference coefficient between two links.
        
        Args:
            victim_link: Victim link identifier
            interferer_link: Interfering link identifier
            
        Returns:
            Interference coefficient α_ℓm (power ratio)
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return 0.0
            
        key = (victim_link, interferer_link)
        return self.interference_matrix.get(key, 0.0)
    
    def enable_opportunistic_sensing(self, 
                                    target_position: np.ndarray,
                                    rcs: float = 1.0) -> Dict[str, float]:
        """
        Enable opportunistic sensing mode and compute potential gains.
        
        Args:
            target_position: Target position in ECEF coordinates (meters)
            rcs: Radar cross section (m²)
            
        Returns:
            Dictionary of opportunistic sensing metrics
        """
        if not self._cache_valid:
            warnings.warn("Cache invalid, call update_dynamic_state first")
            return {}
            
        opp_metrics = {}
        
        # Evaluate each active link for opportunistic sensing
        for link_id, metrics in self.link_metrics.items():
            tx_idx = self._get_satellite_index(metrics['tx_id'])
            rx_idx = self._get_satellite_index(metrics['rx_id'])
            
            # Get satellite positions from state
            tx_pos = self.satellite_states[tx_idx*8:tx_idx*8+3]
            rx_pos = self.satellite_states[rx_idx*8:rx_idx*8+3]
            
            # Calculate bistatic geometry
            geometry = calculate_bistatic_geometry(tx_pos, rx_pos, target_position)
            
            # Estimate bistatic SINR (simplified)
            tx_power = metrics.get('tx_power', 0.0)
            if tx_power > 0:
                # Simplified bistatic radar parameters
                params = type('obj', (object,), {
                    'tx_power': tx_power,
                    'tx_gain': self.antenna_gain,
                    'rx_gain': self.antenna_gain,
                    'wavelength': 299792458.0 / self.frequency_hz,
                    'bistatic_rcs': rcs,
                    'processing_gain': 1e6,  # 60 dB
                    'processing_loss': 2.0,
                    'noise_power': self.noise_power
                })()
                
                sinr_ioo = calculate_sinr_ioo(params, geometry)
                
                # Calculate measurement variance
                variance = calculate_bistatic_measurement_variance(
                    sinr_ioo,
                    self.hw_profile.sigma_phi_squared,
                    self.frequency_hz,
                    self.bandwidth_hz
                )
                
                # Calculate FIM contribution
                J_ioo = calculate_j_ioo(geometry.gradient, variance)
                
                opp_metrics[link_id] = {
                    'bistatic_range': geometry.bistatic_range,
                    'sinr_db': 10*np.log10(sinr_ioo) if sinr_ioo > 0 else -np.inf,
                    'range_std': np.sqrt(variance),
                    'info_gain': np.trace(J_ioo)
                }
                
        return opp_metrics
    
    # ==================== Private Methods ====================
    
    def _compute_all_metrics(self):
        """
        Perform comprehensive physical layer computation.
        
        This is the main computational engine that calculates all physical
        metrics when the dynamic state is updated.
        """
        # Clear previous results
        self.link_metrics.clear()
        self.interference_matrix.clear()
        
        # Phase 1: Compute individual link metrics
        self._compute_link_metrics()
        
        # Phase 2: Compute interference matrix
        self._compute_interference_matrix()
        
        # Phase 3: Update link metrics with interference
        self._update_metrics_with_interference()
        
        # Phase 4: Update network FIM
        self._update_network_fim()
        
    def _compute_link_metrics(self):
        """Compute basic metrics for all active links."""
        
        # Create mapping of link IDs to (tx, rx) pairs
        link_mapping = {}
        for tx_id, rx_id in self.active_links:
            link_id = f"{tx_id}->{rx_id}"
            link_mapping[link_id] = (tx_id, rx_id)
        
        # Compute metrics for each link
        for link_id, (tx_id, rx_id) in link_mapping.items():
            # Get power allocation
            tx_power = 0.0
            if tx_id in self.power_allocations:
                power_alloc = self.power_allocations[tx_id].get('power_allocation', {})
                tx_power = power_alloc.get(link_id, 0.0)
            
            if tx_power <= 0:
                continue
                
            # Get satellite indices and positions
            tx_idx = self._get_satellite_index(tx_id)
            rx_idx = self._get_satellite_index(rx_id)
            
            tx_pos = self.satellite_states[tx_idx*8:tx_idx*8+3]
            rx_pos = self.satellite_states[rx_idx*8:rx_idx*8+3]
            
            # Calculate distance
            distance = np.linalg.norm(rx_pos - tx_pos)
            
            # Calculate channel gain
            channel_gain = calculate_channel_gain(
                distance,
                self.frequency_hz,
                self.antenna_gain,
                self.antenna_gain,
                self.hw_profile.sigma_e,
                self.beamwidth
            )
            
            # Pre-impairment SNR
            snr0 = (tx_power * channel_gain) / self.noise_power
            
            # Store initial metrics
            self.link_metrics[link_id] = {
                'tx_id': tx_id,
                'rx_id': rx_id,
                'tx_power': tx_power,
                'channel_gain': channel_gain,
                'distance': distance,
                'snr0': snr0,
                'interference': 0.0,  # Will be updated
                'sinr_eff': snr0,  # Initial estimate
                'range_variance': np.inf  # Will be updated
            }
    
    def _compute_interference_matrix(self):
        """Compute interference coefficients between all link pairs."""
        
        link_ids = list(self.link_metrics.keys())
        
        for victim_id in link_ids:
            for interferer_id in link_ids:
                if victim_id == interferer_id:
                    continue
                    
                victim = self.link_metrics[victim_id]
                interferer = self.link_metrics[interferer_id]
                
                # Skip if same receiver (no self-interference)
                if victim['rx_id'] == interferer['rx_id']:
                    continue
                    
                # Get positions for interference path
                int_tx_idx = self._get_satellite_index(interferer['tx_id'])
                vic_rx_idx = self._get_satellite_index(victim['rx_id'])
                
                int_tx_pos = self.satellite_states[int_tx_idx*8:int_tx_idx*8+3]
                vic_rx_pos = self.satellite_states[vic_rx_idx*8:vic_rx_idx*8+3]
                
                # Calculate interference path distance
                int_distance = np.linalg.norm(vic_rx_pos - int_tx_pos)
                
                # Simplified interference coefficient (homogeneous network)
                # α_ℓm = (d_ℓ/d_ℓm)² * angular_attenuation
                path_loss_ratio = (victim['distance'] / int_distance)**2
                
                # Simplified angular attenuation (assuming some angular separation)
                angular_atten = 0.01  # -20 dB nominal
                
                alpha_lm = path_loss_ratio * angular_atten
                
                self.interference_matrix[(victim_id, interferer_id)] = alpha_lm
    
    def _update_metrics_with_interference(self):
        """Update link metrics with interference and compute effective SINR."""
        
        for link_id, metrics in self.link_metrics.items():
            # Sum interference from all other links
            total_interference = 0.0
            
            for interferer_id in self.link_metrics:
                if interferer_id == link_id:
                    continue
                    
                key = (link_id, interferer_id)
                if key in self.interference_matrix:
                    alpha = self.interference_matrix[key]
                    int_power = self.link_metrics[interferer_id]['tx_power']
                    total_interference += alpha * int_power
            
            # Normalized interference
            norm_interference = total_interference / self.noise_power
            
            # Calculate effective SINR with all impairments
            sinr_eff = calculate_effective_sinr(
                metrics['snr0'],
                self.hw_profile.gamma_eff,
                self.hw_profile.sigma_phi_squared,
                norm_interference,
                hardware_on=True,
                interference_on=True,
                phase_noise_on=True
            )
            
            # Calculate range variance
            if sinr_eff > 0:
                range_var = calculate_range_variance(
                    sinr_eff,
                    self.hw_profile.sigma_phi_squared,
                    self.frequency_hz,
                    bandwidth=self.bandwidth_hz
                )
            else:
                range_var = np.inf
            
            # Update metrics
            metrics['interference'] = total_interference
            metrics['sinr_eff'] = sinr_eff
            metrics['range_variance'] = range_var
    
    def _update_network_fim(self):
        """Update the network Fisher Information Matrix."""
        
        n_states = self.info_filter.n_states_total
        self.network_fim = np.zeros((n_states, n_states))
        
        # Build list of active links with measurements
        active_link_pairs = []
        range_variances = []
        
        for link_id, metrics in self.link_metrics.items():
            if metrics['sinr_eff'] > 0 and metrics['range_variance'] < np.inf:
                tx_idx = self._get_satellite_index(metrics['tx_id'])
                rx_idx = self._get_satellite_index(metrics['rx_id'])
                active_link_pairs.append((tx_idx, rx_idx))
                range_variances.append(metrics['range_variance'])
        
        if active_link_pairs:
            # Use the FIM update function
            # Note: update_info expects TOA variances in s², so we convert
            dummy_prior = np.eye(n_states) * 1e-6  # Small prior information
            dummy_y = np.zeros((n_states, 1))
            dummy_z = [0.0] * len(active_link_pairs)
            
            self.network_fim, _ = update_info(
                dummy_prior,
                dummy_y,
                active_link_pairs,
                self.satellite_states,
                range_variances,  # Will be converted internally
                dummy_z
            )
            
            # Calculate EFIM
            try:
                self.efim = calculate_efim(
                    self.network_fim,
                    self.info_filter.kinematic_indices,
                    self.info_filter.clock_indices
                )
            except:
                self.efim = None
    
    def _get_satellite_index(self, sat_id: str) -> int:
        """
        Get the index of a satellite in the state vector.
        
        Args:
            sat_id: Satellite identifier (e.g., "SAT_0")
            
        Returns:
            Integer index
        """
        # Extract number from SAT_X format
        try:
            return int(sat_id.split('_')[1])
        except:
            return 0