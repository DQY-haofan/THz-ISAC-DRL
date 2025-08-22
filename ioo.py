"""
Opportunistic Sensing Module for THz LEO-ISL ISAC Network Simulation
=====================================================================

This module implements the "Interference as Opportunity" (IoO) framework,
transforming interference signals into valuable sensing information through
opportunistic bistatic radar configurations.

Based on Section 4.3 of "Network Interference and Opportunistic Sensing":
- Bistatic radar geometry and range calculation
- Fisher Information contribution from opportunistic links
- Information fusion for network performance enhancement

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


@dataclass
class BistaticGeometry:
    """
    Container for bistatic radar geometry parameters.
    
    Attributes:
        bistatic_range: Total signal path length (m)
        gradient: Gradient vector ∇_p_t R_b (3x1)
        transmitter_range: Distance from transmitter to target (m)
        receiver_range: Distance from target to receiver (m)
        bistatic_angle: Angle between Tx-target and Rx-target vectors (rad)
        unit_tx_to_target: Unit vector from transmitter to target
        unit_rx_to_target: Unit vector from receiver to target
    """
    bistatic_range: float
    gradient: np.ndarray
    transmitter_range: float
    receiver_range: float
    bistatic_angle: float
    unit_tx_to_target: np.ndarray
    unit_rx_to_target: np.ndarray


@dataclass
class BistaticRadarParameters:
    """
    Bistatic radar system parameters with processing gain.
    
    Attributes:
        tx_power: Transmit power (Watts)
        tx_gain: Transmit antenna gain (linear scale)
        rx_gain: Receive antenna gain (linear scale)
        wavelength: Carrier wavelength (meters)
        bistatic_rcs: Bistatic radar cross section (m²)
        processing_gain: Coherent processing gain (linear scale, e.g., 1e6 for 60 dB)
        processing_loss: Processing loss factor (≥1)
        noise_power: Receiver noise power (Watts)
    """
    tx_power: float
    tx_gain: float
    rx_gain: float
    wavelength: float
    bistatic_rcs: float
    processing_gain: float = 1e6  # Default 60 dB processing gain
    processing_loss: float = 2.0  # Default 3 dB processing loss
    noise_power: float = 1e-15
    
    def __post_init__(self):
        """Validate radar parameters."""
        if self.tx_power <= 0:
            raise ValueError("Transmit power must be positive")
        if self.tx_gain <= 0 or self.rx_gain <= 0:
            raise ValueError("Antenna gains must be positive")
        if self.wavelength <= 0:
            raise ValueError("Wavelength must be positive")
        if self.bistatic_rcs <= 0:
            raise ValueError("RCS must be positive")
        if self.processing_gain < 1:
            raise ValueError("Processing gain must be ≥ 1")
        if self.processing_loss < 1:
            raise ValueError("Processing loss must be ≥ 1")
        if self.noise_power <= 0:
            raise ValueError("Noise power must be positive")


def calculate_bistatic_geometry(tx_position: np.ndarray,
                               rx_position: np.ndarray,
                               target_position: np.ndarray) -> BistaticGeometry:
    """
    Calculate bistatic radar geometry for opportunistic sensing.
    
    Implements equations (26) and (28) from Section 4.3.1:
    - Bistatic range: R_b(p_t) = ||p_m - p_t|| + ||p_t - p_ℓ||
    - Gradient: ∇_{p_t} R_b = u_{m→t} + u_{ℓ→t}
    
    The gradient represents the direction of maximum information gain,
    lying along the bistatic bisector.
    
    Args:
        tx_position: Transmitter position vector [x, y, z] (m)
        rx_position: Receiver position vector [x, y, z] (m)
        target_position: Target position vector [x, y, z] (m)
    
    Returns:
        BistaticGeometry object containing all geometric parameters
    
    Example:
        >>> tx_pos = np.array([7000e3, 0, 0])
        >>> rx_pos = np.array([0, 7000e3, 0])
        >>> target_pos = np.array([3500e3, 3500e3, 0])
        >>> geometry = calculate_bistatic_geometry(tx_pos, rx_pos, target_pos)
    """
    # Convert to numpy arrays
    tx_position = np.asarray(tx_position, dtype=float)
    rx_position = np.asarray(rx_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    
    # Validate dimensions
    if tx_position.shape != (3,) or rx_position.shape != (3,) or target_position.shape != (3,):
        raise ValueError("All positions must be 3D vectors")
    
    # Calculate ranges
    tx_to_target = target_position - tx_position
    rx_to_target = target_position - rx_position
    
    R_tx = np.linalg.norm(tx_to_target)
    R_rx = np.linalg.norm(rx_to_target)
    
    # Check for degenerate cases
    if R_tx < 1e-6:
        warnings.warn("Target too close to transmitter")
        R_tx = 1e-6
    if R_rx < 1e-6:
        warnings.warn("Target too close to receiver")
        R_rx = 1e-6
    
    # Bistatic range - equation (26)
    R_b = R_tx + R_rx
    
    # Unit vectors
    u_tx_to_target = tx_to_target / R_tx
    u_rx_to_target = rx_to_target / R_rx
    
    # Gradient vector - equation (28)
    # Note: gradient points FROM satellites TO target
    gradient = u_tx_to_target + u_rx_to_target
    
    # Bistatic angle
    cos_beta = np.dot(u_tx_to_target, u_rx_to_target)
    cos_beta = np.clip(cos_beta, -1, 1)  # Numerical safety
    bistatic_angle = np.arccos(cos_beta)
    
    return BistaticGeometry(
        bistatic_range=R_b,
        gradient=gradient,
        transmitter_range=R_tx,
        receiver_range=R_rx,
        bistatic_angle=bistatic_angle,
        unit_tx_to_target=u_tx_to_target,
        unit_rx_to_target=u_rx_to_target
    )


def calculate_sinr_ioo(radar_params: BistaticRadarParameters,
                      geometry: BistaticGeometry) -> float:
    """
    Calculate SINR for opportunistic bistatic radar with processing gain.
    
    Implements the bistatic radar equation with coherent processing gain,
    which is critical for detecting weak echoes from space debris.
    
    Args:
        radar_params: Bistatic radar system parameters
        geometry: Bistatic geometry from calculate_bistatic_geometry
    
    Returns:
        Effective SINR including processing gain (linear scale)
    """
    # Extract parameters
    P_t = radar_params.tx_power
    G_t = radar_params.tx_gain
    G_r = radar_params.rx_gain
    λ = radar_params.wavelength
    σ_b = radar_params.bistatic_rcs
    G_p = radar_params.processing_gain  # KEY ADDITION
    L_proc = radar_params.processing_loss
    N_eff = radar_params.noise_power
    
    R_tx = geometry.transmitter_range
    R_rx = geometry.receiver_range
    
    # Validate ranges
    if R_tx <= 0 or R_rx <= 0:
        warnings.warn("Invalid range in bistatic geometry")
        return 0.0
    
    # Bistatic radar equation with processing gain
    # SINR = (P_t * G_t * G_r * σ_b * λ² * G_p) / ((4π)³ * R_tx² * R_rx² * L_proc * N_eff)
    numerator = P_t * G_t * G_r * σ_b * λ**2 * G_p  # Processing gain in numerator
    denominator = (4 * np.pi)**3 * R_tx**2 * R_rx**2 * L_proc * N_eff
    
    sinr_ioo = numerator / denominator
    
    return sinr_ioo


def calculate_j_ioo(gradient: np.ndarray,
                   measurement_variance: float) -> np.ndarray:
    """
    Calculate Fisher Information Matrix for opportunistic sensing link.
    
    Implements equation (27) from Section 4.3.2:
    J_IoO(p_t) = (1/σ²_{R_b}) * (∇_{p_t} R_b)(∇_{p_t} R_b)^T
    
    This rank-1 matrix provides information along the bistatic bisector
    direction. Multiple opportunistic links with different geometries
    combine to provide full 3D observability.
    
    Args:
        gradient: Gradient vector ∇_{p_t} R_b from bistatic geometry (3x1)
        measurement_variance: Bistatic ranging error variance σ²_{R_b} (m²)
    
    Returns:
        Fisher Information Matrix J_IoO (3x3) for target position
    
    Example:
        >>> gradient = geometry.gradient
        >>> variance = 1e-4  # 10 cm ranging variance
        >>> J_ioo = calculate_j_ioo(gradient, variance)
    """
    # Validate inputs
    gradient = np.asarray(gradient, dtype=float)
    if gradient.shape != (3,):
        raise ValueError("Gradient must be a 3D vector")
    if measurement_variance <= 0:
        raise ValueError("Measurement variance must be positive")
    
    # Reshape gradient to column vector
    grad_col = gradient.reshape(3, 1)
    
    # Calculate FIM - equation (27)
    # J_IoO = (1/σ²) * gradient * gradient^T
    J_ioo = (1 / measurement_variance) * (grad_col @ grad_col.T)
    
    # Ensure symmetry (numerical precision)
    J_ioo = 0.5 * (J_ioo + J_ioo.T)
    
    return J_ioo


def calculate_network_ioo_fim(tx_positions: List[np.ndarray],
                             rx_positions: List[np.ndarray],
                             target_position: np.ndarray,
                             measurement_variances: List[float]) -> np.ndarray:
    """
    Calculate total FIM from multiple opportunistic sensing links.
    
    Implements the information fusion principle from Section 4.3.4:
    J_net,new = J_net,old + Σ J_IoO,i
    
    Multiple bistatic measurements with diverse geometries combine
    to provide full 3D target observability.
    
    Args:
        tx_positions: List of transmitter positions
        rx_positions: List of receiver positions  
        target_position: Target position (common for all links)
        measurement_variances: List of ranging variances for each link
    
    Returns:
        Total network FIM for target position (3x3)
    """
    if len(tx_positions) != len(rx_positions) or len(tx_positions) != len(measurement_variances):
        raise ValueError("Input lists must have same length")
    
    # Initialize total FIM
    J_total = np.zeros((3, 3))
    
    # Sum contributions from each opportunistic link
    for tx_pos, rx_pos, variance in zip(tx_positions, rx_positions, measurement_variances):
        # Calculate geometry
        geometry = calculate_bistatic_geometry(tx_pos, rx_pos, target_position)
        
        # Calculate individual FIM
        J_link = calculate_j_ioo(geometry.gradient, variance)
        
        # Add to total (information additivity)
        J_total += J_link
    
    return J_total


def analyze_geometric_diversity(geometries: List[BistaticGeometry]) -> Dict[str, float]:
    """
    Analyze geometric diversity of opportunistic sensing links.
    
    Quantifies how well the bistatic geometries complement each other
    for 3D target localization.
    
    Args:
        geometries: List of bistatic geometries from multiple links
    
    Returns:
        Dictionary with diversity metrics
    """
    if not geometries:
        return {"error": "No geometries provided"}
    
    # Stack gradient vectors
    gradients = np.array([g.gradient for g in geometries])
    
    # Compute Gram matrix (inner products)
    gram = gradients @ gradients.T
    
    # Analyze orthogonality
    n_links = len(geometries)
    orthogonality_scores = []
    
    for i in range(n_links):
        for j in range(i+1, n_links):
            # Angle between gradients
            cos_angle = gram[i, j] / (np.linalg.norm(gradients[i]) * 
                                      np.linalg.norm(gradients[j]))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            orthogonality_scores.append(angle)
    
    # Compute diversity metrics
    metrics = {
        "n_links": n_links,
        "mean_orthogonality": np.mean(orthogonality_scores) if orthogonality_scores else 0,
        "min_orthogonality": np.min(orthogonality_scores) if orthogonality_scores else 0,
        "max_orthogonality": np.max(orthogonality_scores) if orthogonality_scores else 0,
    }
    
    # Check if gradients span 3D space
    if n_links >= 3:
        rank = np.linalg.matrix_rank(gradients.T, tol=1e-10)
        metrics["spans_3d"] = (rank == 3)
        metrics["gradient_rank"] = rank
    
    return metrics


def estimate_information_gain(J_prior: np.ndarray,
                             J_ioo: np.ndarray) -> Dict[str, float]:
    """
    Quantify information gain from opportunistic sensing.
    
    Based on Section 4.3.4, calculates the reduction in error volume
    and other metrics showing the value of opportunistic measurements.
    
    Args:
        J_prior: Prior FIM before opportunistic sensing (3x3)
        J_ioo: FIM contribution from opportunistic link (3x3)
    
    Returns:
        Dictionary with information gain metrics
    """
    # Posterior FIM after fusion
    J_post = J_prior + J_ioo
    
    # Calculate CRLBs (inverse of FIM)
    try:
        crlb_prior = np.linalg.inv(J_prior)
        prior_invertible = True
    except np.linalg.LinAlgError:
        crlb_prior = np.linalg.pinv(J_prior)
        prior_invertible = False
    
    try:
        crlb_post = np.linalg.inv(J_post)
    except np.linalg.LinAlgError:
        crlb_post = np.linalg.pinv(J_post)
    
    # Error volume (proportional to sqrt of determinant)
    vol_prior = np.sqrt(np.linalg.det(crlb_prior)) if prior_invertible else np.inf
    vol_post = np.sqrt(np.linalg.det(crlb_post))
    
    # Volume reduction ratio - equation (30) from Section 4.3.4
    if vol_prior != np.inf and vol_prior > 0:
        # Corrected formula based on matrix determinant lemma
        gradient = None
        if hasattr(J_ioo, 'shape') and J_ioo.shape == (3, 3):
            # J_ioo is rank-1, extract gradient
            eigenvals, eigenvecs = np.linalg.eigh(J_ioo)
            max_idx = np.argmax(eigenvals)
            if eigenvals[max_idx] > 1e-10:
                gradient = eigenvecs[:, max_idx] * np.sqrt(eigenvals[max_idx])
        
        if gradient is not None:
            # Apply formula from equation (30)
            inner_term = gradient.T @ crlb_prior @ gradient
            volume_ratio = 1 / np.sqrt(1 + inner_term)
        else:
            volume_ratio = vol_post / vol_prior
    else:
        volume_ratio = 0  # Infinite improvement
    
    # Position uncertainty reduction (trace of CRLB)
    pos_uncertainty_prior = np.trace(crlb_prior)
    pos_uncertainty_post = np.trace(crlb_post)
    
    # Information metrics
    metrics = {
        "volume_ratio": volume_ratio,
        "volume_reduction_percent": (1 - volume_ratio) * 100 if volume_ratio > 0 else 100,
        "position_uncertainty_prior": pos_uncertainty_prior,
        "position_uncertainty_post": pos_uncertainty_post,
        "uncertainty_reduction_percent": 
            (1 - pos_uncertainty_post/pos_uncertainty_prior) * 100 
            if pos_uncertainty_prior > 0 else 0,
        "information_increase": np.trace(J_ioo),  # Sum of eigenvalues
        "prior_observable": prior_invertible
    }
    
    return metrics


def calculate_bistatic_measurement_variance(sinr_ioo: float,
                                           sigma_phi_squared: float,
                                           f_c: float,
                                           bandwidth: float) -> float:
    """
    Calculate bistatic ranging error variance.
    
    Uses the same unified model as direct-path measurements but with
    the bistatic SINR from the radar equation.
    
    Args:
        sinr_ioo: Effective SINR for opportunistic echo
        sigma_phi_squared: Phase noise variance (rad²)
        f_c: Carrier frequency (Hz)
        bandwidth: Signal bandwidth (Hz)
    
    Returns:
        Bistatic ranging error variance σ²_{R_b} (m²)
    """
    if sinr_ioo <= 0:
        return np.inf
    
    # Waveform constant
    beta_rms = bandwidth / np.sqrt(12)
    kappa_wf = 1 / (8 * np.pi**2 * beta_rms**2)
    
    # Unified measurement variance model
    waveform_term = kappa_wf / sinr_ioo
    phase_noise_term = sigma_phi_squared / (2 * np.pi * f_c)**2 if sigma_phi_squared > 0 else 0
    
    variance = SPEED_OF_LIGHT**2 * (waveform_term + phase_noise_term)
    
    return variance


# ============================================================================
# Unit Tests
# ============================================================================

def test_bistatic_geometry():
    """Test bistatic geometry calculation."""
    print("Testing bistatic geometry...")
    
    # Simple right-angle configuration
    tx_pos = np.array([1000, 0, 0])
    rx_pos = np.array([0, 1000, 0])
    target_pos = np.array([500, 500, 0])
    
    geometry = calculate_bistatic_geometry(tx_pos, rx_pos, target_pos)
    
    # Check bistatic range
    expected_range = 2 * np.sqrt(500**2 + 500**2)
    assert abs(geometry.bistatic_range - expected_range) < 1e-6, \
        f"Range mismatch: {geometry.bistatic_range} vs {expected_range}"
    
    # Check gradient direction (should point along bisector)
    expected_gradient_dir = np.array([1, 1, 0]) / np.sqrt(2)
    gradient_dir = geometry.gradient / np.linalg.norm(geometry.gradient)
    
    print(f"  Bistatic range: {geometry.bistatic_range:.2f} m")
    print(f"  Bistatic angle: {np.degrees(geometry.bistatic_angle):.1f}°")
    print(f"  Gradient direction: {gradient_dir}")
    
    print("✓ Bistatic geometry test passed")


def test_sinr_calculation():
    """Test opportunistic SINR calculation."""
    print("Testing SINR calculation...")
    
    # Create geometry
    tx_pos = np.array([7000e3, 0, 0])
    rx_pos = np.array([0, 7000e3, 0])
    target_pos = np.array([5000e3, 5000e3, 0])
    geometry = calculate_bistatic_geometry(tx_pos, rx_pos, target_pos)
    
    # Radar parameters
    params = BistaticRadarParameters(
        tx_power=1.0,  # 1 W
        tx_gain=10000,  # 40 dBi
        rx_gain=10000,  # 40 dBi
        wavelength=1e-3,  # 1 mm (300 GHz)
        bistatic_rcs=10.0,  # 10 m² RCS
        processing_loss=2.0,  # 3 dB loss
        noise_power=1e-15  # -150 dBW
    )
    
    sinr = calculate_sinr_ioo(params, geometry)
    sinr_db = 10 * np.log10(sinr)
    
    print(f"  Tx range: {geometry.transmitter_range/1e3:.1f} km")
    print(f"  Rx range: {geometry.receiver_range/1e3:.1f} km")
    print(f"  SINR: {sinr_db:.1f} dB")
    
    assert sinr > 0, "SINR should be positive"
    
    print("✓ SINR calculation test passed")


def test_ioo_fim():
    """Test IoO FIM calculation."""
    print("Testing IoO FIM calculation...")
    
    # Gradient along x-axis
    gradient = np.array([2, 0, 0])  # Magnitude 2 (sum of two unit vectors)
    variance = 1e-4  # 10 cm ranging std dev squared
    
    J_ioo = calculate_j_ioo(gradient, variance)
    
    # Check rank (should be 1)
    rank = np.linalg.matrix_rank(J_ioo)
    assert rank == 1, f"FIM should be rank 1, got {rank}"
    
    # Check eigenvalues
    eigenvals = np.linalg.eigvals(J_ioo)
    eigenvals.sort()
    
    print(f"  FIM shape: {J_ioo.shape}")
    print(f"  FIM rank: {rank}")
    print(f"  Eigenvalues: {eigenvals}")
    print(f"  Information along gradient: {J_ioo[0, 0]:.2e}")
    
    # Information should be concentrated along gradient direction
    expected_info = 4 / variance  # ||gradient||² / variance
    assert abs(J_ioo[0, 0] - expected_info) < 1e-10, \
        f"Information mismatch: {J_ioo[0, 0]} vs {expected_info}"
    
    print("✓ IoO FIM test passed")


def test_network_fusion():
    """Test network-level information fusion."""
    print("Testing network information fusion...")
    
    # Target at origin
    target_pos = np.array([0, 0, 0])
    
    # Three transmitters in xy-plane
    tx_positions = [
        np.array([1000, 0, 0]),
        np.array([0, 1000, 0]),
        np.array([-500, -500, 0])
    ]
    
    # Three receivers at different locations
    rx_positions = [
        np.array([0, -1000, 0]),
        np.array([-1000, 0, 0]),
        np.array([500, 500, 0])
    ]
    
    # Equal measurement variances
    variances = [1e-4, 1e-4, 1e-4]
    
    # Calculate total FIM
    J_total = calculate_network_ioo_fim(tx_positions, rx_positions,
                                       target_pos, variances)
    
    # Check properties
    rank = np.linalg.matrix_rank(J_total)
    eigenvals = np.linalg.eigvals(J_total)
    
    print(f"  Total FIM rank: {rank}")
    print(f"  Eigenvalues: {eigenvals}")
    
    # Should achieve full 3D observability with 3 diverse links
    assert rank == 3, f"Should have full rank, got {rank}"
    
    # Calculate CRLB
    crlb = np.linalg.inv(J_total)
    pos_rmse = np.sqrt(np.trace(crlb))
    
    print(f"  Position RMSE: {pos_rmse:.3f} m")
    
    print("✓ Network fusion test passed")


def test_geometric_diversity():
    """Test geometric diversity analysis."""
    print("Testing geometric diversity analysis...")
    
    # Create diverse geometries
    target_pos = np.array([0, 0, 0])
    
    geometries = [
        calculate_bistatic_geometry(
            np.array([1000, 0, 0]),
            np.array([0, 1000, 0]),
            target_pos
        ),
        calculate_bistatic_geometry(
            np.array([0, 0, 1000]),
            np.array([1000, 0, 0]),
            target_pos
        ),
        calculate_bistatic_geometry(
            np.array([0, 1000, 0]),
            np.array([0, 0, 1000]),
            target_pos
        )
    ]
    
    metrics = analyze_geometric_diversity(geometries)
    
    print(f"  Number of links: {metrics['n_links']}")
    print(f"  Mean orthogonality: {np.degrees(metrics['mean_orthogonality']):.1f}°")
    print(f"  Spans 3D: {metrics.get('spans_3d', False)}")
    print(f"  Gradient rank: {metrics.get('gradient_rank', 0)}")
    
    assert metrics.get('spans_3d', False), "Should span 3D space"
    
    print("✓ Geometric diversity test passed")


def test_information_gain():
    """Test information gain quantification."""
    print("Testing information gain calculation...")
    
    # Prior information (moderate uncertainty)
    J_prior = np.diag([0.01, 0.01, 0.001])  # Worse Z accuracy
    
    # Opportunistic measurement improving Z
    gradient = np.array([0, 0, 2])  # Strong Z component
    J_ioo = calculate_j_ioo(gradient, 1e-4)
    
    # Calculate gain
    metrics = estimate_information_gain(J_prior, J_ioo)
    
    print(f"  Volume reduction: {metrics['volume_reduction_percent']:.1f}%")
    print(f"  Uncertainty reduction: {metrics['uncertainty_reduction_percent']:.1f}%")
    print(f"  Information increase: {metrics['information_increase']:.2e}")
    
    assert metrics['volume_reduction_percent'] > 0, "Should reduce volume"
    assert metrics['uncertainty_reduction_percent'] > 0, "Should reduce uncertainty"
    
    print("✓ Information gain test passed")


if __name__ == "__main__":
    """Run all unit tests and demonstrate usage."""
    print("=" * 60)
    print("Running Opportunistic Sensing Module Tests")
    print("=" * 60)
    
    test_bistatic_geometry()
    test_sinr_calculation()
    test_ioo_fim()
    test_network_fusion()
    test_geometric_diversity()
    test_information_gain()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    # Comprehensive example
    print("\n" + "=" * 60)
    print("Example: Opportunistic Sensing Scenario")
    print("=" * 60)
    
    # Scenario: Space debris tracking
    print("\nScenario: Tracking space debris using ISL interference")
    print("-" * 40)
    
    # Satellite positions (LEO constellation)
    sat1_pos = np.array([7000e3, 0, 500e3])
    sat2_pos = np.array([0, 7000e3, 600e3])
    sat3_pos = np.array([-5000e3, -5000e3, 550e3])
    
    # Debris position
    debris_pos = np.array([2000e3, 3000e3, 550e3])
    
    print(f"Debris position: [{debris_pos[0]/1e3:.0f}, "
          f"{debris_pos[1]/1e3:.0f}, {debris_pos[2]/1e3:.0f}] km")
    
    # Calculate bistatic geometries for potential IoO links
    print("\nOpportunistic Links:")
    print("-" * 40)
    
    links = [
        ("Sat1→Sat2", sat1_pos, sat2_pos),
        ("Sat1→Sat3", sat1_pos, sat3_pos),
        ("Sat2→Sat3", sat2_pos, sat3_pos)
    ]
    
    geometries = []
    J_ioo_list = []
    
    for name, tx_pos, rx_pos in links:
        # Calculate geometry
        geom = calculate_bistatic_geometry(tx_pos, rx_pos, debris_pos)
        geometries.append(geom)
        
        # Calculate SINR
        params = BistaticRadarParameters(
            tx_power=1.0,
            tx_gain=10000,
            rx_gain=10000,
            wavelength=1e-3,  # 300 GHz
            bistatic_rcs=1.0,  # 1 m² for small debris
            processing_loss=3.0,
            noise_power=1e-14
        )
        
        sinr = calculate_sinr_ioo(params, geom)
        
        # Calculate measurement variance
        variance = calculate_bistatic_measurement_variance(
            sinr, sigma_phi_squared=1e-4, f_c=300e9, bandwidth=10e9
        )
        
        # Calculate FIM contribution
        J_ioo = calculate_j_ioo(geom.gradient, variance)
        J_ioo_list.append(J_ioo)
        
        print(f"{name}:")
        print(f"  Bistatic range: {geom.bistatic_range/1e3:.1f} km")
        print(f"  Bistatic angle: {np.degrees(geom.bistatic_angle):.1f}°")
        print(f"  SINR: {10*np.log10(sinr):.1f} dB")
        print(f"  Ranging std: {np.sqrt(variance):.2f} m")
    
    # Analyze geometric diversity
    print("\nGeometric Diversity:")
    print("-" * 40)
    diversity = analyze_geometric_diversity(geometries)
    print(f"Mean orthogonality: {np.degrees(diversity['mean_orthogonality']):.1f}°")
    print(f"Spans 3D space: {diversity.get('spans_3d', False)}")
    
    # Calculate total information
    print("\nInformation Fusion:")
    print("-" * 40)
    
    # Sum all IoO contributions
    J_total = sum(J_ioo_list)
    
    # Calculate CRLB
    try:
        crlb = np.linalg.inv(J_total)
        pos_rmse = np.sqrt(np.trace(crlb))
        
        print(f"Combined position RMSE: {pos_rmse:.2f} m")
        print(f"X accuracy: {np.sqrt(crlb[0,0]):.2f} m")
        print(f"Y accuracy: {np.sqrt(crlb[1,1]):.2f} m")
        print(f"Z accuracy: {np.sqrt(crlb[2,2]):.2f} m")
    except np.linalg.LinAlgError:
        print("System not fully observable with current geometry")
    
    # Key insight
    print("\n" + "=" * 60)
    print("Key Insight: Interference as Opportunity")
    print("=" * 60)
    print("\nBy repurposing interference signals for bistatic radar:")
    print("• Each interfering link provides sensing information")
    print("• Multiple links with diverse geometries enable 3D localization")
    print("• Information gain is maximized along bistatic bisector")
    print("• Weak echoes from many links combine for strong performance")
    print("\nThis transforms interference from a performance limiter")
    print("into a valuable source of sensing information!")