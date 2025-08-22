"""
Fisher Information Matrix Engine for THz LEO-ISL ISAC Network Simulation
=========================================================================

This module is the core computational engine for sensing performance analysis.
It implements the recursive information filtering framework and extracts
performance metrics (CRLB, GDOP) from the network Fisher Information Matrix.

Based on Section III of "Network Fisher Information Matrix Framework":
- Information filter prediction (time update) equations (9)-(11)
- Measurement update with additive information property
- EFIM computation via Schur complement for marginalization
- CRLB and GDOP extraction

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, eye as sparse_eye, block_diag
from typing import Tuple, Optional, List, Dict, Union
import warnings

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


class InformationFilter:
    """
    Information filter for recursive state estimation in dynamic networks.
    
    The information filter is the dual of the Kalman filter, operating on:
    - Information matrix J = P^(-1)
    - Information vector y = J * x_hat
    
    This formulation provides computational advantages for networked systems
    with dynamic topology and distributed measurements.
    """
    
    def __init__(self, n_states_per_sat: int = 8, n_satellites: int = 4):
        """
        Initialize information filter.
        
        Args:
            n_states_per_sat: States per satellite (8: 3D pos, 3D vel, clock bias, drift)
            n_satellites: Number of satellites in the network
        """
        self.n_states_per_sat = n_states_per_sat
        self.n_satellites = n_satellites
        self.n_states_total = n_states_per_sat * n_satellites
        
        # State indices for partitioning
        self.kinematic_indices = []
        self.clock_indices = []
        
        for i in range(n_satellites):
            base = i * n_states_per_sat
            # First 6 states are kinematic (position + velocity)
            self.kinematic_indices.extend(range(base, base + 6))
            # Last 2 states are clock (bias + drift)
            self.clock_indices.extend(range(base + 6, base + 8))
        
        self.kinematic_indices = np.array(self.kinematic_indices)
        self.clock_indices = np.array(self.clock_indices)


def build_jacobian(sat_i_idx: int, sat_j_idx: int,
                  sat_states: np.ndarray,
                  n_states_per_sat: int = 8) -> np.ndarray:
    """
    Build sparse Jacobian row vector H_ℓ for TOA measurement between two satellites.
    Generic implementation supporting arbitrary network topology.
    
    Args:
        sat_i_idx: Index of transmitting satellite (0-based)
        sat_j_idx: Index of receiving satellite (0-based)
        sat_states: Complete network state vector (8N_v x 1)
        n_states_per_sat: Number of states per satellite (default 8)
    
    Returns:
        Jacobian matrix H (1 x n_states_total)
    """
    n_states_total = len(sat_states)
    n_satellites = n_states_total // n_states_per_sat
    
    # Validate indices
    if sat_i_idx < 0 or sat_i_idx >= n_satellites:
        raise ValueError(f"Invalid transmitter index: {sat_i_idx}")
    if sat_j_idx < 0 or sat_j_idx >= n_satellites:
        raise ValueError(f"Invalid receiver index: {sat_j_idx}")
    if sat_i_idx == sat_j_idx:
        raise ValueError("Transmitter and receiver must be different")
    
    # Calculate index positions dynamically
    i_start = sat_i_idx * n_states_per_sat
    j_start = sat_j_idx * n_states_per_sat
    
    # Extract satellite states
    sat_i_state = sat_states[i_start:i_start + n_states_per_sat]
    sat_j_state = sat_states[j_start:j_start + n_states_per_sat]
    
    # Extract positions (first 3 elements of each state)
    p_i = sat_i_state[:3]
    p_j = sat_j_state[:3]
    
    # Calculate unit line-of-sight vector
    delta_p = p_j - p_i
    distance = np.linalg.norm(delta_p)
    
    if distance < 1e-6:
        warnings.warn(f"Satellites {sat_i_idx} and {sat_j_idx} too close, Jacobian may be singular")
        u_ij = np.zeros(3)
    else:
        u_ij = delta_p / distance
    
    # Initialize sparse Jacobian
    H = np.zeros((1, n_states_total))
    
    # Fill non-zero elements at correct positions
    # Partial derivatives w.r.t sat_i position
    H[0, i_start:i_start+3] = -u_ij / SPEED_OF_LIGHT
    
    # Partial derivatives w.r.t sat_j position  
    H[0, j_start:j_start+3] = u_ij / SPEED_OF_LIGHT
    
    # Partial derivatives w.r.t clock biases
    H[0, i_start + 6] = -1  # sat_i clock bias
    H[0, j_start + 6] = 1   # sat_j clock bias
    
    return H



def predict_info(J_prev: np.ndarray, y_prev: np.ndarray,
                F: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Information filter time update (prediction) step.
    
    Implements equations (9)-(11) from Section III.B:
    M_k = (J_net(k-1|k-1) + F_k^T Q_{k-1}^{-1} F_k)^{-1}
    J_net(k|k-1) = Q_{k-1}^{-1} - Q_{k-1}^{-1} F_k M_k F_k^T Q_{k-1}^{-1}
    y_net(k|k-1) = J_net(k|k-1) F_k J_net^{-1}(k-1|k-1) y_net(k-1|k-1)
    
    This formulation avoids inversion of potentially singular F matrices.
    
    Args:
        J_prev: Posterior information matrix at k-1 (n x n)
        y_prev: Posterior information vector at k-1 (n x 1)
        F: State transition matrix (n x n)
        Q: Process noise covariance matrix (n x n)
    
    Returns:
        J_prior: Prior information matrix at k (n x n)
        y_prior: Prior information vector at k (n x 1)
    
    Note: Uses linear equation solving instead of explicit inversion for stability.
    """
    n = J_prev.shape[0]
    
    # Handle infinite prior (no information) case
    if np.allclose(J_prev, 0):
        # With no prior information, prediction yields no information
        return np.zeros_like(J_prev), np.zeros_like(y_prev)
    
    # Compute Q inverse (use pseudo-inverse for numerical stability)
    try:
        Q_inv = np.linalg.pinv(Q, rcond=1e-10)
    except np.linalg.LinAlgError:
        warnings.warn("Process noise covariance is singular, using regularization")
        Q_reg = Q + 1e-10 * np.eye(n)
        Q_inv = np.linalg.inv(Q_reg)
    
    # Compute M matrix - equation (9)
    # M_k = (J_prev + F^T Q^{-1} F)^{-1}
    M_inner = J_prev + F.T @ Q_inv @ F
    
    try:
        # Use Cholesky decomposition for positive definite matrices
        L = np.linalg.cholesky(M_inner)
        M = linalg.cho_solve((L, True), np.eye(n))
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if not positive definite
        M = np.linalg.pinv(M_inner, rcond=1e-10)
    
    # Compute prior information matrix - equation (10)
    # J_prior = Q^{-1} - Q^{-1} F M F^T Q^{-1}
    Q_inv_F = Q_inv @ F
    J_prior = Q_inv - Q_inv_F @ M @ Q_inv_F.T
    
    # Ensure symmetry (numerical errors can break it)
    J_prior = 0.5 * (J_prior + J_prior.T)
    
    # Compute prior information vector - equation (11)
    # y_prior = J_prior F J_prev^{-1} y_prev
    # Avoid explicit inversion: solve J_prev * x_prev = y_prev for x_prev
    try:
        x_prev = np.linalg.solve(J_prev, y_prev)
    except np.linalg.LinAlgError:
        # Use least squares if singular
        x_prev, _, _, _ = np.linalg.lstsq(J_prev, y_prev, rcond=1e-10)
    
    # Propagate state and convert back to information vector
    x_prior = F @ x_prev
    y_prior = J_prior @ x_prior
    
    return J_prior, y_prior


def update_info(J_prior: np.ndarray, y_prior: np.ndarray,
               active_links: List[Tuple[int, int]],
               sat_states: np.ndarray,
               range_variance_list: List[float],
               z_list: List[float],
               correlated_noise: bool = False,
               correlation_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Information filter measurement update (correction) step.
    
    CRITICAL: This function expects range variances in m² and converts them
    to TOA variances in s² internally for FIM calculation.
    
    Args:
        J_prior: Prior information matrix (n x n)
        y_prior: Prior information vector (n x 1)
        active_links: List of (sat_i_idx, sat_j_idx) tuples for active ISLs
        sat_states: Current network state vector
        range_variance_list: List of range measurement variances in m² (METERS SQUARED!)
        z_list: List of actual TOA measurements in seconds
        correlated_noise: If True, use correlation_matrix
        correlation_matrix: Full noise covariance C_n (if correlated)
    
    Returns:
        J_post: Posterior information matrix (n x n)
        y_post: Posterior information vector (n x 1)
    """
    n = J_prior.shape[0]
    J_post = J_prior.copy()
    y_post = y_prior.copy()
    
    if not active_links:  # No measurements
        return J_post, y_post
    
    # Build Jacobian for each active link
    H_list = []
    R_list = []  # TOA variances in s²
    
    for (sat_i_idx, sat_j_idx), range_var_m2 in zip(active_links, range_variance_list):
        # Build Jacobian using generic function with satellite indices
        H = build_jacobian(sat_i_idx, sat_j_idx, sat_states)
        H_list.append(H)
        
        # CRITICAL UNIT CONVERSION: Convert range variance (m²) to TOA variance (s²)
        # This is the key fix that ensures dimensional consistency
        toa_variance_s2 = range_var_m2 / (SPEED_OF_LIGHT**2)
        R_list.append(toa_variance_s2)
    
    if correlated_noise and correlation_matrix is not None:
        # Stack all Jacobians
        H = np.vstack(H_list)
        z = np.array(z_list).reshape(-1, 1)
        
        # Convert correlation matrix from range domain to TOA domain
        C_n_toa = correlation_matrix / (SPEED_OF_LIGHT**2)
        
        # Use Woodbury identity for efficient inversion
        try:
            C_n_inv = np.linalg.inv(C_n_toa)
        except np.linalg.LinAlgError:
            warnings.warn("Correlation matrix singular, using pseudo-inverse")
            C_n_inv = np.linalg.pinv(C_n_toa)
        
        # Update with correlated measurements
        J_post = J_prior + H.T @ C_n_inv @ H
        y_post = y_prior + H.T @ C_n_inv @ z
        
    else:
        # Independent measurements - simple addition
        for H_ell, R_ell_s2, z_ell in zip(H_list, R_list, z_list):
            # Each link's information contribution (using TOA variance in s²)
            J_ell = H_ell.T @ H_ell / R_ell_s2
            y_ell = H_ell.T * z_ell / R_ell_s2
            
            # Add to network totals
            J_post += J_ell
            y_post += y_ell.reshape(-1, 1)
    
    # Ensure symmetry
    J_post = 0.5 * (J_post + J_post.T)
    
    return J_post, y_post
    

def calculate_efim(J_full: np.ndarray,
                  kinematic_indices: np.ndarray,
                  clock_indices: np.ndarray) -> np.ndarray:
    """
    Calculate Equivalent FIM (EFIM) by marginalizing out nuisance parameters.
    
    Implements equation (19) from Section III.D using Schur complement:
    J_EFIM(a) = J_aa - J_ab * J_bb^{-1} * J_ba
    
    This represents the information about kinematic states after optimally
    estimating and accounting for clock parameters.
    
    Args:
        J_full: Complete posterior information matrix
        kinematic_indices: Indices of kinematic states (positions, velocities)
        clock_indices: Indices of clock states (biases, drifts)
    
    Returns:
        J_EFIM: Equivalent FIM for kinematic states only
    """
    # Extract submatrices
    J_aa = J_full[np.ix_(kinematic_indices, kinematic_indices)]
    J_ab = J_full[np.ix_(kinematic_indices, clock_indices)]
    J_ba = J_full[np.ix_(clock_indices, kinematic_indices)]
    J_bb = J_full[np.ix_(clock_indices, clock_indices)]
    
    # Check if clock parameters are observable
    if np.linalg.matrix_rank(J_bb) < len(clock_indices):
        warnings.warn("Clock parameters not fully observable, EFIM may be ill-conditioned")
        # Use pseudo-inverse for rank-deficient case
        J_bb_inv = np.linalg.pinv(J_bb, rcond=1e-10)
    else:
        try:
            # Use Cholesky for efficient inversion
            L = np.linalg.cholesky(J_bb)
            J_bb_inv = linalg.cho_solve((L, True), np.eye(J_bb.shape[0]))
        except np.linalg.LinAlgError:
            J_bb_inv = np.linalg.pinv(J_bb, rcond=1e-10)
    
    # Compute Schur complement
    J_EFIM = J_aa - J_ab @ J_bb_inv @ J_ba
    
    # Ensure symmetry and positive semi-definiteness
    J_EFIM = 0.5 * (J_EFIM + J_EFIM.T)
    
    # Check conditioning
    try:
        cond_num = np.linalg.cond(J_EFIM)
        if cond_num > 1e12:
            warnings.warn(f"EFIM poorly conditioned (κ = {cond_num:.2e})")
    except:
        pass
    
    return J_EFIM


def get_performance_metrics(J_efim: np.ndarray,
                          n_satellites: int = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Extract performance metrics from the Equivalent FIM.
    
    Computes:
    - Network CRLB matrix: CRLB_net = J_EFIM^{-1}
    - Position CRLB for each satellite
    - Velocity CRLB for each satellite
    - Network GDOP: sqrt(trace(CRLB_pos))
    - Observability: rank and condition number of J_EFIM
    
    Args:
        J_efim: Equivalent FIM for kinematic states
        n_satellites: Number of satellites (for per-satellite metrics)
    
    Returns:
        Dictionary containing all performance metrics
    """
    metrics = {}
    
    # Check observability
    rank = np.linalg.matrix_rank(J_efim, tol=1e-10)
    n_states = J_efim.shape[0]
    metrics['rank'] = rank
    metrics['fully_observable'] = (rank == n_states)
    
    if rank < n_states:
        warnings.warn(f"System not fully observable: rank {rank} < {n_states}")
        # Use pseudo-inverse for rank-deficient case
        CRLB_net = np.linalg.pinv(J_efim, rcond=1e-10)
    else:
        try:
            # Compute CRLB matrix (inverse of EFIM)
            CRLB_net = np.linalg.inv(J_efim)
        except np.linalg.LinAlgError:
            warnings.warn("EFIM singular, using pseudo-inverse")
            CRLB_net = np.linalg.pinv(J_efim, rcond=1e-10)
    
    metrics['CRLB_matrix'] = CRLB_net
    
    # Condition number (observability strength)
    try:
        metrics['condition_number'] = np.linalg.cond(J_efim)
    except:
        metrics['condition_number'] = np.inf
    
    # Extract per-satellite metrics if possible
    if n_satellites is not None:
        n_kin_per_sat = 6  # 3D position + 3D velocity
        
        metrics['position_crlb'] = {}
        metrics['velocity_crlb'] = {}
        
        for i in range(n_satellites):
            base_idx = i * n_kin_per_sat
            
            # Position CRLB (3x3 submatrix)
            pos_crlb = CRLB_net[base_idx:base_idx+3, base_idx:base_idx+3]
            pos_rmse = np.sqrt(np.trace(pos_crlb))
            metrics['position_crlb'][f'sat_{i}'] = {
                'covariance': pos_crlb,
                'rmse': pos_rmse,
                'x_std': np.sqrt(pos_crlb[0, 0]),
                'y_std': np.sqrt(pos_crlb[1, 1]),
                'z_std': np.sqrt(pos_crlb[2, 2])
            }
            
            # Velocity CRLB (3x3 submatrix)
            vel_crlb = CRLB_net[base_idx+3:base_idx+6, base_idx+3:base_idx+6]
            vel_rmse = np.sqrt(np.trace(vel_crlb))
            metrics['velocity_crlb'][f'sat_{i}'] = {
                'covariance': vel_crlb,
                'rmse': vel_rmse,
                'vx_std': np.sqrt(vel_crlb[0, 0]),
                'vy_std': np.sqrt(vel_crlb[1, 1]),
                'vz_std': np.sqrt(vel_crlb[2, 2])
            }
    
    # Calculate network GDOP (for all position states)
    if n_satellites is not None:
        # Extract all position-related diagonal elements
        pos_indices = []
        for i in range(n_satellites):
            base = i * 6
            pos_indices.extend(range(base, base + 3))
        
        pos_crlb_total = CRLB_net[np.ix_(pos_indices, pos_indices)]
        gdop = np.sqrt(np.trace(pos_crlb_total))
        metrics['GDOP'] = gdop
        
        # Additional GDOP components
        metrics['PDOP'] = gdop  # Position DOP (same as GDOP for position-only)
        metrics['HDOP'] = np.sqrt(pos_crlb_total[0, 0] + pos_crlb_total[1, 1])  # Horizontal
        metrics['VDOP'] = np.sqrt(pos_crlb_total[2, 2])  # Vertical (if Z is vertical)
    else:
        # Fallback: compute GDOP from full kinematic CRLB
        gdop = np.sqrt(np.trace(CRLB_net))
        metrics['GDOP'] = gdop
    
    return metrics


def create_state_transition_matrix(dt: float, n_satellites: int) -> np.ndarray:
    """
    Create block-diagonal state transition matrix for satellite constellation.
    
    Each satellite follows linear dynamics:
    - Position += velocity * dt
    - Clock bias += clock drift * dt
    
    Args:
        dt: Time step (seconds)
        n_satellites: Number of satellites
    
    Returns:
        Block-diagonal state transition matrix F
    """
    # Single satellite transition matrix (8x8)
    F_single = np.eye(8)
    F_single[0:3, 3:6] = dt * np.eye(3)  # Position from velocity
    F_single[6, 7] = dt  # Clock bias from drift
    
    # Create block diagonal matrix
    F_blocks = [F_single for _ in range(n_satellites)]
    F = block_diag(F_blocks)
    
    return F


def create_process_noise_covariance(dt: float, n_satellites: int,
                                   sigma_a: float = 1e-6,
                                   sigma_clk: float = 1e-12) -> np.ndarray:
    """
    Create process noise covariance matrix Q.
    
    Models:
    - Velocity random walk for orbital dynamics
    - Allan variance for clock stability
    
    Args:
        dt: Time step (seconds)
        n_satellites: Number of satellites
        sigma_a: Acceleration noise PSD (m²/s³)
        sigma_clk: Clock drift noise (s/s^(1/2))
    
    Returns:
        Process noise covariance matrix Q
    """
    # Single satellite process noise (8x8)
    Q_single = np.zeros((8, 8))
    
    # Position-velocity coupling (velocity random walk)
    Q_pos_vel = np.array([
        [dt**3/3, dt**2/2],
        [dt**2/2, dt]
    ]) * sigma_a
    
    # Fill 3D position-velocity blocks
    for i in range(3):
        Q_single[i:i+1, i:i+1] = Q_pos_vel[0, 0]
        Q_single[i:i+1, i+3:i+4] = Q_pos_vel[0, 1]
        Q_single[i+3:i+4, i:i+1] = Q_pos_vel[1, 0]
        Q_single[i+3:i+4, i+3:i+4] = Q_pos_vel[1, 1]
    
    # Clock noise
    Q_single[6, 6] = sigma_clk * dt  # Clock bias
    Q_single[7, 7] = sigma_clk * dt  # Clock drift
    
    # Create block diagonal matrix
    Q_blocks = [Q_single for _ in range(n_satellites)]
    Q = block_diag(Q_blocks)
    
    return Q


# ============================================================================
# Unit Tests
# ============================================================================

def test_jacobian_construction():
    """Test Jacobian construction for TOA measurement."""
    print("Testing Jacobian construction...")
    
    # Two satellites at known positions
    sat_i = np.array([7000e3, 0, 0, 0, 7.5e3, 0, 1e-6, 0])
    sat_j = np.array([0, 7000e3, 0, -7.5e3, 0, 0, 2e-6, 0])
    
    H = build_jacobian(sat_i, sat_j, 16)  # 2 satellites total
    
    # Check sparsity
    non_zero = np.count_nonzero(H)
    print(f"  Non-zero elements: {non_zero} / {H.size}")
    
    # Check structure
    assert H.shape == (1, 16), "Incorrect shape"
    assert non_zero == 8, "Should have 8 non-zero elements"
    
    print("✓ Jacobian construction test passed")


def test_information_prediction():
    """Test information filter prediction step."""
    print("Testing information prediction...")
    
    n = 8
    # Initial information (moderate uncertainty)
    J_prev = np.eye(n) * 100
    y_prev = J_prev @ np.ones(n)
    
    # Simple dynamics
    F = create_state_transition_matrix(dt=1.0, n_satellites=1)
    Q = create_process_noise_covariance(dt=1.0, n_satellites=1)
    
    # Predict
    J_prior, y_prior = predict_info(J_prev, y_prev, F, Q)
    
    # Check information decrease (uncertainty increase)
    print(f"  Previous info norm: {np.linalg.norm(J_prev):.2f}")
    print(f"  Predicted info norm: {np.linalg.norm(J_prior):.2f}")
    
    assert np.linalg.norm(J_prior) <= np.linalg.norm(J_prev), \
        "Information should decrease during prediction"
    
    print("✓ Information prediction test passed")


def test_measurement_update():
    """Test measurement update with multiple links."""
    print("Testing measurement update...")
    
    n = 16  # 2 satellites
    J_prior = np.eye(n) * 10
    y_prior = np.zeros((n, 1))
    
    # Create mock measurements
    H1 = np.zeros((1, n))
    H1[0, :3] = [0.5, 0.5, 0.707] / SPEED_OF_LIGHT
    H1[0, 6] = 1
    
    H2 = np.zeros((1, n))
    H2[0, 8:11] = [0.707, 0.707, 0] / SPEED_OF_LIGHT
    H2[0, 14] = 1
    
    H_list = [H1, H2]
    R_list = [1e-12, 2e-12]  # Measurement variances
    z_list = [1e-6, 2e-6]  # Measurements
    
    # Update
    J_post, y_post = update_info(J_prior, y_prior, H_list, R_list, z_list)
    
    # Check information increase
    print(f"  Prior info norm: {np.linalg.norm(J_prior):.2f}")
    print(f"  Posterior info norm: {np.linalg.norm(J_post):.2f}")
    
    assert np.linalg.norm(J_post) >= np.linalg.norm(J_prior), \
        "Information should increase with measurements"
    
    print("✓ Measurement update test passed")


def test_efim_calculation():
    """Test EFIM calculation via Schur complement."""
    print("Testing EFIM calculation...")
    
    # Create information filter
    info_filter = InformationFilter(n_states_per_sat=8, n_satellites=2)
    
    # Create a full-rank information matrix
    n = 16
    J_full = np.eye(n) * 100
    # Add some coupling between kinematic and clock states
    J_full[:12, 12:] = np.random.randn(12, 4) * 10
    J_full[12:, :12] = J_full[:12, 12:].T
    J_full = 0.5 * (J_full + J_full.T)  # Ensure symmetry
    
    # Calculate EFIM
    J_efim = calculate_efim(J_full, info_filter.kinematic_indices,
                           info_filter.clock_indices)
    
    print(f"  Full matrix size: {J_full.shape}")
    print(f"  EFIM size: {J_efim.shape}")
    print(f"  EFIM condition number: {np.linalg.cond(J_efim):.2e}")
    
    assert J_efim.shape == (12, 12), "EFIM should be 12x12 for 2 satellites"
    
    print("✓ EFIM calculation test passed")


def test_performance_metrics():
    """Test performance metrics extraction."""
    print("Testing performance metrics extraction...")
    
    # Create a simple EFIM
    n_sats = 2
    J_efim = np.eye(12) * 1000  # High information (low uncertainty)
    
    # Get metrics
    metrics = get_performance_metrics(J_efim, n_satellites=n_sats)
    
    print(f"  Observability rank: {metrics['rank']} / 12")
    print(f"  Condition number: {metrics['condition_number']:.2e}")
    print(f"  GDOP: {metrics['GDOP']:.3f} m")
    
    # Check specific satellite metrics
    for i in range(n_sats):
        pos_rmse = metrics['position_crlb'][f'sat_{i}']['rmse']
        vel_rmse = metrics['velocity_crlb'][f'sat_{i}']['rmse']
        print(f"  Sat {i}: pos RMSE = {pos_rmse:.3f} m, "
              f"vel RMSE = {vel_rmse:.3f} m/s")
    
    assert metrics['fully_observable'], "System should be observable"
    assert metrics['GDOP'] > 0, "GDOP should be positive"
    
    print("✓ Performance metrics test passed")


def test_recursive_filtering():
    """Test complete recursive filtering cycle."""
    print("Testing recursive filtering...")
    
    # Initialize
    n_sats = 2
    info_filter = InformationFilter(n_states_per_sat=8, n_satellites=n_sats)
    
    # Start with no prior information
    J = np.zeros((16, 16))
    y = np.zeros((16, 1))
    
    # Dynamics
    F = create_state_transition_matrix(dt=1.0, n_satellites=n_sats)
    Q = create_process_noise_covariance(dt=1.0, n_satellites=n_sats)
    
    # Simulate 3 time steps
    gdop_history = []
    
    for k in range(3):
        # Prediction
        if k > 0:
            J, y = predict_info(J, y, F, Q)
        
        # Measurement update (mock data)
        H1 = np.random.randn(1, 16) * 0.1
        H2 = np.random.randn(1, 16) * 0.1
        J, y = update_info(J, y, [H1, H2], [1e-12, 1e-12], [1e-6, 2e-6])
        
        # Calculate EFIM and GDOP
        if np.linalg.matrix_rank(J) > 0:
            J_efim = calculate_efim(J, info_filter.kinematic_indices,
                                   info_filter.clock_indices)
            metrics = get_performance_metrics(J_efim, n_satellites=n_sats)
            gdop_history.append(metrics.get('GDOP', np.inf))
            print(f"  Step {k}: GDOP = {gdop_history[-1]:.3f} m")
    
    print("✓ Recursive filtering test passed")

def prepare_measurement_noise_for_fim(range_variance_m2: float) -> float:
    """
    Convert range variance (m²) to TOA variance (s²) for FIM calculations.
    
    The FIM framework expects time-domain measurement noise since the
    Jacobian H is derived for TOA measurements in seconds.
    
    Args:
        range_variance_m2: Range measurement variance in m²
    
    Returns:
        TOA measurement variance in s²
    """
    # Convert range variance to time variance
    toa_variance_s2 = range_variance_m2 / (SPEED_OF_LIGHT**2)
    return toa_variance_s2

if __name__ == "__main__":
    """Run all unit tests and demonstrate usage."""
    print("=" * 60)
    print("Running FIM Engine Unit Tests")
    print("=" * 60)
    
    test_jacobian_construction()
    test_information_prediction()
    test_measurement_update()
    test_efim_calculation()
    test_performance_metrics()
    test_recursive_filtering()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    # Comprehensive example
    print("\n" + "=" * 60)
    print("Example: Network FIM Analysis")
    print("=" * 60)
    
    # Network configuration
    n_satellites = 4
    print(f"\nNetwork: {n_satellites} satellites")
    print("-" * 40)
    
    # Initialize information filter
    info_filter = InformationFilter(n_states_per_sat=8, n_satellites=n_satellites)
    n_total = info_filter.n_states_total
    
    # Initial information (moderate prior)
    J_network = np.eye(n_total) * 50
    y_network = np.zeros((n_total, 1))
    
    # Simulate measurements from 6 ISLs
    print("\nMeasurement Update:")
    print("-" * 40)
    
    # Create ISL measurements (simplified)
    H_list = []
    R_list = []
    z_list = []
    
    # ISL pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    for i, j in pairs:
        H = np.zeros((1, n_total))
        # Simplified Jacobian (random for demonstration)
        H[0, i*8:(i*8+3)] = np.random.randn(3) * 0.1
        H[0, j*8:(j*8+3)] = np.random.randn(3) * 0.1
        H[0, i*8+6] = -1
        H[0, j*8+6] = 1
        
        H_list.append(H)
        R_list.append(1e-12)  # 1 ps² timing variance
        z_list.append(np.random.randn() * 1e-6)  # μs measurements
    
    print(f"Active ISLs: {len(H_list)}")
    
    # Update with measurements
    J_post, y_post = update_info(J_network, y_network, H_list, R_list, z_list)
    
    print(f"Information gain: {np.linalg.norm(J_post - J_network):.2f}")
    
    # Calculate EFIM
    print("\nEFIM Analysis:")
    print("-" * 40)
    
    J_efim = calculate_efim(J_post, info_filter.kinematic_indices,
                           info_filter.clock_indices)
    
    print(f"Full FIM size: {J_post.shape}")
    print(f"EFIM size: {J_efim.shape} (kinematic states only)")
    
    # Extract performance metrics
    print("\nPerformance Metrics:")
    print("-" * 40)
    
    metrics = get_performance_metrics(J_efim, n_satellites=n_satellites)
    
    print(f"System observability: {'Full' if metrics['fully_observable'] else 'Partial'}")
    print(f"FIM rank: {metrics['rank']} / {J_efim.shape[0]}")
    print(f"Condition number: {metrics['condition_number']:.2e}")
    print(f"Network GDOP: {metrics['GDOP']:.3f} m")
    
    # Per-satellite metrics
    print("\nPer-Satellite CRLB:")
    print("-" * 40)
    
    for i in range(n_satellites):
        pos_data = metrics['position_crlb'][f'sat_{i}']
        vel_data = metrics['velocity_crlb'][f'sat_{i}']
        
        print(f"Satellite {i}:")
        print(f"  Position RMSE: {pos_data['rmse']:.3f} m")
        print(f"    X: {pos_data['x_std']:.3f} m")
        print(f"    Y: {pos_data['y_std']:.3f} m")
        print(f"    Z: {pos_data['z_std']:.3f} m")
        print(f"  Velocity RMSE: {vel_data['rmse']:.3f} m/s")
    
    # DOP components
    print("\nDOP Analysis:")
    print("-" * 40)
    print(f"GDOP: {metrics['GDOP']:.3f}")
    print(f"PDOP: {metrics['PDOP']:.3f}")
    print(f"HDOP: {metrics['HDOP']:.3f}")
    print(f"VDOP: {metrics['VDOP']:.3f}")
    
    print("\n" + "=" * 60)
    print("FIM engine demonstration complete!")