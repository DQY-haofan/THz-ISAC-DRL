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
from collections import defaultdict
import os
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
        """
        # Store configurations
        self.const_config = constellation_config
        self.hw_config = hardware_config
        self.isac_config = isac_config
        
        # Convert units to SI
        self.frequency_hz = hardware_config.frequency_ghz * 1e9
        self.bandwidth_hz = hardware_config.bandwidth_ghz * 1e9
        self.max_tx_power_w = 10**((hardware_config.tx_power_max_dbm - 30) / 10)
        
        
        self.link_registry: Dict[str, Tuple[str, str]] = {}

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
        
        # 添加递归信息矩阵和向量作为持久状态变量
        n_states = self.info_filter.n_states_total
        self.info_matrix_J = np.eye(n_states) * 1e-6  # 小的初始先验
        self.info_vector_y = np.zeros((n_states, 1))
        
        # 添加时间跟踪
        self.last_update_time = 0.0
        
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
        
    def reset(self):
        """
        重置递归FIM状态（由环境的reset方法调用）。
        """
        n_states = self.info_filter.n_states_total
        self.info_matrix_J = np.eye(n_states) * 1e-6
        self.info_vector_y = np.zeros((n_states, 1))
        self.last_update_time = 0.0

    def update_dynamic_state(self,
                        current_time: float,
                        satellite_states: np.ndarray,
                        active_links: Set[Tuple[str, str]],
                        power_allocations: Dict[str, Dict[str, float]],
                        link_registry: Dict[str, Tuple[str, str]]): # <-- 添加这一行
        """
        Update the interface with current dynamic network state.
        
        在测量更新之前执行时间更新（预测）步骤。
        """
        # Invalidate cache when state changes
        self._cache_valid = False
        self._cache.clear()
        self.link_registry = link_registry
        # Store current state
        self.current_time = current_time
        self.satellite_states = satellite_states
        self.active_links = active_links
        self.power_allocations = power_allocations
        
        # 执行时间更新（预测）如果时间已经推进
        if current_time > self.last_update_time:
            dt = current_time - self.last_update_time
            
            # 创建状态转移矩阵和过程噪声协方差
            from fim import create_state_transition_matrix, create_process_noise_covariance, predict_info
            
            F = create_state_transition_matrix(dt, self.const_config.n_satellites)
            Q = create_process_noise_covariance(dt, self.const_config.n_satellites)
            
            # 执行信息预测
            self.info_matrix_J, self.info_vector_y = predict_info(
                self.info_matrix_J, self.info_vector_y, F, Q
            )
            
            self.last_update_time = current_time
        
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
        # 重新初始化所有缓存，确保从干净的状态开始
        self.link_metrics = {}
        self.interference_matrix = {}
        self.network_fim = None
        self.efim = None

        # Phase 1: 计算所有活动链路的基础指标
        self._compute_link_metrics()
        
        # --- DEBUG PRINT 1: 检查 snr0 是否被正确计算 ---
        # 只有在verbose模式下，并且link_metrics非空时打印
        if os.getenv('VERBOSE_DEBUG') == '1' and self.link_metrics:
            first_link = next(iter(self.link_metrics.keys()))
            fm = self.link_metrics[first_link]
            print(f"[DEBUG Phy-L1] Link {first_link}: tx_power={fm.get('tx_power', 0):.2f}, "
                  f"channel_gain={fm.get('channel_gain', 0):.2e}, snr0={fm.get('snr0', 0):.2f}")
        # ---------------------------------------------

        # Phase 2 & 3: 计算干扰并更新SINR
        self._compute_interference_and_update_sinr()

        # Phase 4: 更新网络FIM用于感知
        self._update_network_fim()
        

    def _compute_link_metrics(self):
        """Compute basic metrics for all active links."""
        
        # 遍历 link_registry 以处理所有定义的链路
        for link_id, (tx_id, rx_id) in self.link_registry.items():
            
            # --- START OF CORRECTION ---
            #
            # 步骤 1: 无论功率如何，首先计算基础几何，这是所有后续计算的基础
            #
            tx_idx = self._get_satellite_index(tx_id)
            rx_idx = self._get_satellite_index(rx_id)
            
            tx_pos = self.satellite_states[tx_idx*8:tx_idx*8+3]
            rx_pos = self.satellite_states[rx_idx*8:rx_idx*8+3]
            
            distance = np.linalg.norm(rx_pos - tx_pos)
            
            # 步骤 2: 获取此链路的发射功率
            #
            tx_power = 0.0
            if tx_id in self.power_allocations:
                power_alloc = self.power_allocations[tx_id].get('power_allocation', {})
                tx_power = power_alloc.get(link_id, 0.0)
            
            # 步骤 3: 根据功率，计算信道增益和SNR0
            #
            if tx_power > 0:
                channel_gain = calculate_channel_gain(
                    distance,
                    self.frequency_hz,
                    self.antenna_gain,
                    self.antenna_gain,
                    self.hw_profile.sigma_e,
                    self.beamwidth
                )
                snr0 = (tx_power * channel_gain) / self.noise_power
            else:
                channel_gain = 0.0
                snr0 = 0.0

            # 步骤 4: 无论功率如何，都为链路创建指标条目，确保字典完整
            #
            self.link_metrics[link_id] = {
                'tx_id': tx_id,
                'rx_id': rx_id,
                'tx_power': tx_power,
                'channel_gain': channel_gain,
                'distance': distance,
                'snr0': snr0,
                'interference': 0.0,
                'sinr_eff': snr0,      # 初始估计
                'range_variance': np.inf 
            }
            #
            # --- END OF CORRECTION ---
    
    # def _compute_interference_matrix(self):
    #     """
    #     Compute the interference matrix using high-fidelity physics-based model.
        
    #     This implements the geometric-stochastic interference model from interference.py,
    #     incorporating path loss, antenna patterns, and pointing errors.
    #     """
    #     import numpy as np
    #     from interference import LinkParameters, calculate_alpha_lm
        
    #     # Initialize interference matrix
    #     n_links = len(self.link_registry)
    #     self.interference_matrix = np.zeros((n_links, n_links))
        
    #     # Get link IDs for indexing
    #     link_ids = list(self.link_registry.keys())
    #     link_id_to_idx = {lid: idx for idx, lid in enumerate(link_ids)}
        
    #     # Iterate over all link pairs
    #     for victim_idx, victim_id in enumerate(link_ids):
    #         for interferer_idx, interferer_id in enumerate(link_ids):
    #             if victim_id == interferer_id:
    #                 continue  # No self-interference
                
    #             # Get transmitter and receiver for each link
    #             victim_tx, victim_rx = self.link_registry[victim_id]
    #             interferer_tx, interferer_rx = self.link_registry[interferer_id]
                
    #             # Get link metrics
    #             victim_metrics = self.link_metrics.get(victim_id, {})
    #             interferer_metrics = self.link_metrics.get(interferer_id, {})
                
    #             # Skip if metrics not available
    #             if not victim_metrics or not interferer_metrics:
    #                 continue
                
    #             # Extract positions from satellite states
    #             victim_tx_pos = self.satellite_states[victim_tx]['position']
    #             victim_rx_pos = self.satellite_states[victim_rx]['position']
    #             interferer_tx_pos = self.satellite_states[interferer_tx]['position']
                
    #             # Calculate distances (in meters)
    #             victim_distance = np.linalg.norm(victim_rx_pos - victim_tx_pos)
    #             interferer_to_victim_distance = np.linalg.norm(victim_rx_pos - interferer_tx_pos)
                
    #             # Calculate angular separation (theta_lm)
    #             # This is the angle between the interferer's beam and the victim receiver
    #             interferer_beam_dir = (interferer_rx_pos - interferer_tx_pos) / \
    #                                 np.linalg.norm(interferer_rx_pos - interferer_tx_pos)
    #             interferer_to_victim_dir = (victim_rx_pos - interferer_tx_pos) / \
    #                                     interferer_to_victim_distance
                
    #             # Angular separation in radians
    #             cos_theta = np.clip(np.dot(interferer_beam_dir, interferer_to_victim_dir), -1, 1)
    #             theta_lm = np.arccos(cos_theta)
                
    #             # Create LinkParameters for target link
    #             target_link = LinkParameters(
    #                 power=victim_metrics.get('tx_power', self.default_tx_power),
    #                 gain_tx=victim_metrics.get('gain_tx', self.default_antenna_gain),
    #                 gain_rx=victim_metrics.get('gain_rx', self.default_antenna_gain),
    #                 beamwidth=victim_metrics.get('beamwidth', self.default_beamwidth),
    #                 sigma_e=victim_metrics.get('pointing_error', self.default_pointing_error),
    #                 distance=victim_distance
    #             )
                
    #             # Create LinkParameters for interfering transmitter
    #             interferer_link = LinkParameters(
    #                 power=interferer_metrics.get('tx_power', self.default_tx_power),
    #                 gain_tx=interferer_metrics.get('gain_tx', self.default_antenna_gain),
    #                 gain_rx=interferer_metrics.get('gain_rx', self.default_antenna_gain),
    #                 beamwidth=interferer_metrics.get('beamwidth', self.default_beamwidth),
    #                 sigma_e=interferer_metrics.get('pointing_error', self.default_pointing_error),
    #                 distance=np.linalg.norm(interferer_rx_pos - interferer_tx_pos)
    #             )
                
    #             # Calculate interference coefficient using physics-based model
    #             alpha_lm = calculate_alpha_lm(
    #                 target_link=target_link,
    #                 interferer_tx=interferer_link,
    #                 distance_lm=interferer_to_victim_distance,
    #                 theta_lm=theta_lm
    #             )
                
    #             # Store in interference matrix
    #             self.interference_matrix[victim_idx, interferer_idx] = alpha_lm
    
    #     return self.interference_matrix
    
    def _compute_interference_and_update_sinr(self):
        """
        Compute interference and update SINR and range variance for all links.
        """
        # 临时存储每个链路的归一化干扰总和
        normalized_interference_map = defaultdict(float)

        # 遍历每一对链路来计算干扰
        for victim_id, victim_metrics in self.link_metrics.items():
            if victim_metrics.get('snr0', 0) <= 0:
                continue

            total_normalized_interference = 0
            for interferer_id, interferer_metrics in self.link_metrics.items():
                if victim_id == interferer_id:
                    continue
                
                # ... (此处应插入你已实现的、精确的alpha_lm计算逻辑) ...
                # 作为一个临时的、功能正确的占位符，我们使用简化模型：
                alpha_lm = 1e-3 # 假设一个固定的、小的干扰系数
                
                # 计算归一化干扰 alpha_tilde = SNR0 * alpha
                alpha_tilde = victim_metrics['snr0'] * alpha_lm
                total_normalized_interference += alpha_tilde
            
            normalized_interference_map[victim_id] = total_normalized_interference

        # 现在，用计算出的干扰来更新每个链路的最终指标
        for link_id, metrics in self.link_metrics.items():
            if metrics.get('snr0', 0) <= 0:
                metrics['sinr_eff'] = 0.0
                metrics['range_variance'] = np.inf
                continue

            sinr_eff = calculate_effective_sinr(
                metrics['snr0'],
                self.hw_profile.gamma_eff,
                self.hw_profile.sigma_phi_squared,
                normalized_interference_map[link_id]
            )
            
            range_var = calculate_range_variance_m2(
                sinr_eff,
                self.hw_profile.sigma_phi_squared,
                self.frequency_hz,
                bandwidth=self.bandwidth_hz
            )

            metrics['sinr_eff'] = sinr_eff
            metrics['range_variance'] = range_var

    def get_all_direct_channel_gains(self) -> Dict[str, complex]:
        """
        Get all direct channel gains for active links.
        
        Returns:
            Dictionary mapping link_id to complex channel gain h_ℓ
        """
        import numpy as np
        from hardware import calculate_channel_gain, calculate_antenna_gain, calculate_beamwidth
        
        direct_channels = {}
        
        for link_id, (tx_id, rx_id) in self.link_registry.items():
            # Get link metrics
            metrics = self.link_metrics.get(link_id, {})
            if not metrics.get('active', False):
                continue
                
            # Get positions
            tx_pos = self.satellite_states[tx_id]['position']
            rx_pos = self.satellite_states[rx_id]['position']
            distance = np.linalg.norm(rx_pos - tx_pos)
            
            # Calculate antenna parameters
            G_T = metrics.get('gain_tx', calculate_antenna_gain(
                self.frequency_hz, self.antenna_diameter))
            G_R = metrics.get('gain_rx', G_T)
            theta_B = calculate_beamwidth(self.frequency_hz, self.antenna_diameter)
            sigma_e = metrics.get('pointing_error', self.default_pointing_error)
            
            # Calculate channel gain magnitude
            channel_gain_magnitude = calculate_channel_gain(
                d=distance,
                f_c=self.frequency_hz,
                G_T=G_T,
                G_R=G_R,
                sigma_e=sigma_e,
                theta_B=theta_B
            )
            
            # Add random phase (uniform distribution)
            phase = np.random.uniform(0, 2*np.pi)
            direct_channels[link_id] = np.sqrt(channel_gain_magnitude) * np.exp(1j * phase)
        
        return direct_channels


    def get_all_interference_channel_gains(self) -> Dict[Tuple[str, str], complex]:
        """
        Get all interference channel gains between links.
        
        Returns:
            Dictionary mapping (interferer_link_id, victim_link_id) to complex channel gain g_ℓ',ℓ
        """
        import numpy as np
        from hardware import calculate_channel_gain, calculate_antenna_gain, calculate_beamwidth
        
        interference_channels = {}
        
        # Ensure interference matrix is computed
        if self.interference_matrix is None:
            self._compute_interference_matrix()
        
        link_ids = list(self.link_registry.keys())
        
        for victim_idx, victim_id in enumerate(link_ids):
            for interferer_idx, interferer_id in enumerate(link_ids):
                if victim_id == interferer_id:
                    continue
                    
                # Get interference coefficient
                alpha_lm = self.interference_matrix[victim_idx, interferer_idx]
                
                if alpha_lm > 0:
                    # Get link information
                    victim_tx, victim_rx = self.link_registry[victim_id]
                    interferer_tx, _ = self.link_registry[interferer_id]
                    
                    # Get positions
                    victim_rx_pos = self.satellite_states[victim_rx]['position']
                    interferer_tx_pos = self.satellite_states[interferer_tx]['position']
                    distance = np.linalg.norm(victim_rx_pos - interferer_tx_pos)
                    
                    # Calculate interference channel gain from alpha_lm
                    # Since alpha_lm = |g_lm|^2 / |h_l|^2 * (other factors)
                    # We approximate the interference channel gain
                    victim_metrics = self.link_metrics.get(victim_id, {})
                    path_loss = (self.wavelength / (4 * np.pi * distance))**2
                    
                    # Get antenna gains
                    G_T = victim_metrics.get('gain_tx', self.default_antenna_gain)
                    G_R = victim_metrics.get('gain_rx', self.default_antenna_gain)
                    
                    # Compute interference channel gain magnitude
                    interference_gain_magnitude = alpha_lm * path_loss * G_T * G_R
                    
                    # Add random phase
                    phase = np.random.uniform(0, 2*np.pi)
                    interference_channels[(interferer_id, victim_id)] = \
                        np.sqrt(interference_gain_magnitude) * np.exp(1j * phase)
        
        return interference_channels


#     def _update_metrics_with_interference(self):
#         """Update link metrics with interference and compute effective SINR."""
        
#         from performance_model import calculate_effective_sinr, calculate_range_variance_m2

#         for link_id, metrics in self.link_metrics.items():
#             # Sum interference from all other links
#             total_interference = 0.0
            
#             for interferer_id in self.link_metrics:
#                 if interferer_id == link_id:
#                     continue
                    
#                 key = (link_id, interferer_id)
#                 if key in self.interference_matrix:
#                     alpha = self.interference_matrix[key]
#                     int_power = self.link_metrics[interferer_id]['tx_power']
#                     total_interference += alpha * int_power
            
#             # Normalized interference
#             norm_interference = total_interference / self.noise_power
            
#             # Calculate effective SINR with all impairments
#             sinr_eff = calculate_effective_sinr(
#                 metrics['snr0'],
#                 self.hw_profile.gamma_eff,
#                 self.hw_profile.sigma_phi_squared,
#                 norm_interference,
#                 hardware_on=True,
#                 interference_on=True,
#                 phase_noise_on=True
#             )
            
#             # Calculate range variance
#             if sinr_eff > 0:
# # 修正函数调用 - 使用正确的函数名
#                 range_var = calculate_range_variance_m2(
#                     sinr_eff=sinr_eff,
#                     sigma_phi_squared=sigma_phi_squared,
#                     f_c=self.frequency_hz,
#                     bandwidth=self.bandwidth_hz
#                 )
#             else:
#                 range_var = np.inf
            
#             # Update metrics
#             metrics['interference'] = total_interference
#             metrics['sinr_eff'] = sinr_eff
#             metrics['range_variance'] = range_var
    
    def _update_network_fim(self):
        """
        更新网络Fisher信息矩阵（仅执行测量更新/校正步骤）。
        """
        from fim import update_info, calculate_efim
        
        # Build list of active links with measurements
        active_link_pairs = []
        range_variances = []
        measurements = []  # 实际测量值
        
        for link_id, metrics in self.link_metrics.items():
            if metrics['sinr_eff'] > 0 and metrics['range_variance'] < np.inf:
                tx_idx = self._get_satellite_index(metrics['tx_id'])
                rx_idx = self._get_satellite_index(metrics['rx_id'])
                active_link_pairs.append((tx_idx, rx_idx))
                range_variances.append(metrics['range_variance'])
                
                # 添加模拟的TOA测量（实际系统中应该是真实测量）
                measurements.append(metrics.get('toa_measurement', 0.0))
        
        if active_link_pairs:
            # 执行测量更新（使用现有的info_matrix_J作为先验）
            self.info_matrix_J, self.info_vector_y = update_info(
                self.info_matrix_J,  # 使用递归的先验
                self.info_vector_y,
                active_link_pairs,
                self.satellite_states,
                range_variances,  # 将在内部转换
                measurements
            )
            
            # 存储完整的网络FIM
            self.network_fim = self.info_matrix_J.copy()
            
            # 计算EFIM
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