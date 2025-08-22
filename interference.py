"""
Interference Coefficient Module for THz LEO-ISL ISAC Network Simulation
========================================================================

This module implements the geometric-stochastic interference model for
networked THz inter-satellite links. It provides closed-form expressions
for interference coefficients that capture the interplay between deterministic
network geometry and stochastic platform dynamics.

Based on Section 4.2 of "Network Interference and Opportunistic Sensing",
specifically equations (24) and (25) for heterogeneous and homogeneous networks.

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings


@dataclass
class LinkParameters:
    """
    Parameters characterizing a communication link.
    
    Attributes:
        power: Transmit power (Watts)
        gain_tx: Transmit antenna gain (linear scale)
        gain_rx: Receive antenna gain (linear scale)
        beamwidth: 3dB beamwidth (radians)
        sigma_e: Pointing error standard deviation (radians)
        distance: Link distance (meters)
    """
    power: float
    gain_tx: float
    gain_rx: float
    beamwidth: float
    sigma_e: float
    distance: float
    
    def __post_init__(self):
        """Validate link parameters."""
        if self.power <= 0:
            raise ValueError("Transmit power must be positive")
        if self.gain_tx <= 0 or self.gain_rx <= 0:
            raise ValueError("Antenna gains must be positive")
        if self.beamwidth <= 0:
            raise ValueError("Beamwidth must be positive")
        if self.sigma_e < 0:
            raise ValueError("Pointing error must be non-negative")
        if self.distance <= 0:
            raise ValueError("Distance must be positive")


def calculate_alpha_lm(target_link: LinkParameters,
                      interferer_tx: LinkParameters,
                      distance_lm: float,
                      theta_lm: float) -> float:
    """
    Calculate the general interference coefficient for heterogeneous networks.
    
    Implements equation (24) from Section 4.2.3:
    α_ℓm = (P_m G_{T,m})/(P_ℓ G_{T,ℓ}) * (d_ℓ/d_{ℓm})^2 * 
           (1 + 4σ²_{e,ℓ}/θ²_{B,ℓ})/(1 + 4σ²_{e,m}/θ²_{B,m}) * 
           exp(-(θ_{ℓm}^2)/(θ²_{B,m}/2 + 2σ²_{e,m}))
    
    This general form accounts for different hardware specifications across
    satellites in the constellation.
    
    Args:
        target_link: Parameters of the desired communication link ℓ
        interferer_tx: Parameters of the interfering transmitter m
        distance_lm: Distance between interferer tx and target rx (meters)
        theta_lm: Angular separation between nominal pointing directions (radians)
    
    Returns:
        Interference coefficient α_ℓm (dimensionless ratio)
    
    Example:
        >>> target = LinkParameters(power=1.0, gain_tx=1000, gain_rx=1000,
        ...                        beamwidth=2e-3, sigma_e=1e-6, distance=2000e3)
        >>> interferer = LinkParameters(power=2.0, gain_tx=2000, gain_rx=1000,
        ...                           beamwidth=1e-3, sigma_e=2e-6, distance=1500e3)
        >>> alpha = calculate_alpha_lm(target, interferer, 3000e3, 5e-3)
    """
    # Validate inputs
    if distance_lm <= 0:
        raise ValueError("Interferer-to-receiver distance must be positive")
    if theta_lm < 0:
        raise ValueError("Angular separation must be non-negative")
    
    # Power and antenna gain ratio
    power_gain_ratio = (interferer_tx.power * interferer_tx.gain_tx) / \
                      (target_link.power * target_link.gain_tx)
    
    # Path loss ratio (inverse square law)
    path_loss_ratio = (target_link.distance / distance_lm)**2
    
    # Pointing error broadening factor for target link
    broadening_target = 1 + 4 * (target_link.sigma_e / target_link.beamwidth)**2
    
    # Pointing error broadening factor for interferer
    broadening_interferer = 1 + 4 * (interferer_tx.sigma_e / interferer_tx.beamwidth)**2
    
    # Broadening ratio (accounts for different platform stabilities)
    broadening_ratio = broadening_target / broadening_interferer
    
    # Effective beam broadening for interference path
    # Denominator in exponential: θ²_B,m/2 + 2σ²_e,m
    effective_width_squared = (interferer_tx.beamwidth**2 / 2) + \
                             (2 * interferer_tx.sigma_e**2)
    
    # Angular attenuation factor (Gaussian beam pattern)
    angular_attenuation = np.exp(-theta_lm**2 / effective_width_squared)
    
    # Complete interference coefficient
    alpha_lm = power_gain_ratio * path_loss_ratio * broadening_ratio * angular_attenuation
    
    return alpha_lm


def calculate_alpha_lm_homogeneous(d_l: float, d_lm: float, 
                                  theta_lm: float, theta_B: float,
                                  sigma_e: float) -> float:
    """
    Calculate the simplified interference coefficient for homogeneous networks.
    
    Implements equation (25) from Section 4.2.3:
    α_ℓm = (d_ℓ/d_{ℓm})^2 * exp(-(θ_{ℓm}^2)/(θ²_B/2 + 2σ²_e))
    
    This simplified form applies when all satellites have identical:
    - Transmit power (P_m = P_ℓ)
    - Antenna characteristics (G_T, G_R, θ_B)
    - Platform stability (σ_e)
    
    Args:
        d_l: Distance of target link (meters)
        d_lm: Distance from interferer to target receiver (meters)
        theta_lm: Angular separation between pointing directions (radians)
        theta_B: Common 3dB beamwidth (radians)
        sigma_e: Common pointing error std dev (radians)
    
    Returns:
        Interference coefficient α_ℓm (dimensionless ratio)
    
    Example:
        >>> alpha = calculate_alpha_lm_homogeneous(
        ...     d_l=2000e3,      # 2000 km target link
        ...     d_lm=3000e3,     # 3000 km interference path
        ...     theta_lm=5e-3,   # 5 mrad angular separation
        ...     theta_B=2e-3,    # 2 mrad beamwidth
        ...     sigma_e=1e-6     # 1 μrad pointing error
        ... )
    """
    # Input validation
    if d_l <= 0 or d_lm <= 0:
        raise ValueError("Distances must be positive")
    if theta_lm < 0:
        raise ValueError("Angular separation must be non-negative")
    if theta_B <= 0:
        raise ValueError("Beamwidth must be positive")
    if sigma_e < 0:
        raise ValueError("Pointing error must be non-negative")
    
    # Path loss ratio
    path_loss_ratio = (d_l / d_lm)**2
    
    # Effective beam broadening
    effective_width_squared = (theta_B**2 / 2) + (2 * sigma_e**2)
    
    # Angular attenuation
    angular_attenuation = np.exp(-theta_lm**2 / effective_width_squared)
    
    # Simplified interference coefficient
    alpha_lm = path_loss_ratio * angular_attenuation
    
    return alpha_lm


def calculate_normalized_interference_coeff(alpha_lm: float, 
                                          snr0_l: float) -> float:
    """
    Calculate the normalized interference coefficient for SINR computation.
    
    Implements the normalization: α̃_ℓm = SNR_{0,ℓ} * α_ℓm
    
    This normalization is used in the network SINR formula (Section 2.3):
    SINR_eff,ℓ = exp(-σ²_φ,ℓ) / (SNR⁻¹_{0,ℓ} + Γ_eff,ℓ + Σ α̃_ℓm)
    
    The tilde notation indicates that the interference coefficient has been
    scaled to be directly comparable with the inverse SNR term.
    
    Args:
        alpha_lm: Interference coefficient (power ratio)
        snr0_l: Pre-impairment SNR of target link (linear scale)
    
    Returns:
        Normalized interference coefficient α̃_ℓm
    
    Example:
        >>> alpha = 0.01  # -20 dB interference
        >>> snr0 = 100    # 20 dB SNR
        >>> alpha_tilde = calculate_normalized_interference_coeff(alpha, snr0)
        >>> # Result: 1.0 (interference equals noise in normalized units)
    """
    if alpha_lm < 0:
        raise ValueError("Interference coefficient must be non-negative")
    if snr0_l <= 0:
        raise ValueError("SNR must be positive")
    
    alpha_tilde_lm = snr0_l * alpha_lm
    
    return alpha_tilde_lm


def calculate_network_interference(target_link_params: LinkParameters,
                                  interferer_list: List[Tuple[LinkParameters, float, float]],
                                  normalized: bool = False,
                                  snr0: Optional[float] = None) -> float:
    """
    Calculate total network interference from multiple interfering links.
    
    Aggregates interference coefficients from all active interfering transmitters
    in the network, implementing the summation Σ_{m≠ℓ} α_ℓm.
    
    Args:
        target_link_params: Parameters of the target/victim link
        interferer_list: List of tuples (interferer_params, distance_lm, theta_lm)
        normalized: If True, return normalized coefficients (requires snr0)
        snr0: Pre-impairment SNR for normalization (required if normalized=True)
    
    Returns:
        Total interference coefficient (sum of all α_ℓm or α̃_ℓm)
    
    Example:
        >>> target = LinkParameters(power=1.0, gain_tx=1000, gain_rx=1000,
        ...                        beamwidth=2e-3, sigma_e=1e-6, distance=2000e3)
        >>> interferers = [
        ...     (LinkParameters(...), 3000e3, 5e-3),  # Interferer 1
        ...     (LinkParameters(...), 2500e3, 8e-3),  # Interferer 2
        ... ]
        >>> total_interference = calculate_network_interference(target, interferers)
    """
    if normalized and snr0 is None:
        raise ValueError("SNR required for normalized interference calculation")
    
    total_interference = 0.0
    
    for interferer_params, distance_lm, theta_lm in interferer_list:
        # Calculate individual interference coefficient
        alpha = calculate_alpha_lm(target_link_params, interferer_params,
                                  distance_lm, theta_lm)
        
        # Normalize if requested
        if normalized:
            alpha = calculate_normalized_interference_coeff(alpha, snr0)
        
        total_interference += alpha
    
    return total_interference


def effective_beam_broadening(theta_B: float, sigma_e: float) -> float:
    """
    Calculate the effective beam broadening due to pointing jitter.
    
    The effective angular width combines inherent beam divergence with
    platform-induced spreading: θ_eff = sqrt(θ²_B/2 + 2σ²_e)
    
    This represents the characteristic angular scale over which
    interference power decays.
    
    Args:
        theta_B: 3dB beamwidth (radians)
        sigma_e: Pointing error standard deviation (radians)
    
    Returns:
        Effective angular width (radians)
    """
    if theta_B <= 0:
        raise ValueError("Beamwidth must be positive")
    if sigma_e < 0:
        raise ValueError("Pointing error must be non-negative")
    
    effective_width = np.sqrt(theta_B**2 / 2 + 2 * sigma_e**2)
    
    return effective_width


def interference_operating_regime(theta_B: float, sigma_e: float) -> str:
    """
    Determine the interference operating regime based on beam and jitter.
    
    Classifies the system into one of three regimes:
    - Static: σ_e << θ_B (platform stability excellent)
    - Transition: σ_e ~ θ_B (comparable scales)
    - Dynamic: σ_e >> θ_B (jitter dominates)
    
    Args:
        theta_B: 3dB beamwidth (radians)
        sigma_e: Pointing error standard deviation (radians)
    
    Returns:
        String describing the operating regime
    """
    ratio = sigma_e / theta_B
    
    if ratio < 0.1:
        return "Static (antenna-limited)"
    elif ratio < 1.0:
        return "Transition"
    else:
        return "Dynamic (jitter-limited)"


# ============================================================================
# Unit Tests
# ============================================================================

def test_homogeneous_consistency():
    """Test that general and homogeneous formulas agree for identical parameters."""
    print("Testing homogeneous consistency...")
    
    # Common parameters
    power = 1.0
    gain = 1000.0
    beamwidth = 2e-3  # 2 mrad
    sigma_e = 1e-6    # 1 μrad
    d_l = 2000e3      # 2000 km
    d_lm = 3000e3     # 3000 km
    theta_lm = 5e-3   # 5 mrad
    
    # Create identical link parameters
    target = LinkParameters(power, gain, gain, beamwidth, sigma_e, d_l)
    interferer = LinkParameters(power, gain, gain, beamwidth, sigma_e, d_l)
    
    # Calculate with general formula
    alpha_general = calculate_alpha_lm(target, interferer, d_lm, theta_lm)
    
    # Calculate with homogeneous formula
    alpha_homo = calculate_alpha_lm_homogeneous(d_l, d_lm, theta_lm, 
                                               beamwidth, sigma_e)
    
    # Check consistency
    relative_error = abs(alpha_general - alpha_homo) / alpha_homo
    assert relative_error < 1e-10, f"Formulas inconsistent: {relative_error}"
    
    print(f"  General formula: α = {alpha_general:.6e}")
    print(f"  Homogeneous formula: α = {alpha_homo:.6e}")
    print(f"  Relative error: {relative_error:.2e}")
    print("✓ Consistency test passed")


def test_path_loss_scaling():
    """Test that interference scales correctly with distance."""
    print("Testing path loss scaling...")
    
    # Base case
    d_l = 2000e3
    d_lm1 = 2000e3
    alpha1 = calculate_alpha_lm_homogeneous(d_l, d_lm1, 0, 2e-3, 1e-6)
    
    # Double the interference path distance
    d_lm2 = 4000e3
    alpha2 = calculate_alpha_lm_homogeneous(d_l, d_lm2, 0, 2e-3, 1e-6)
    
    # Should scale as (d_lm1/d_lm2)^2 = 1/4
    expected_ratio = (d_lm1 / d_lm2)**2
    actual_ratio = alpha2 / alpha1
    
    assert abs(actual_ratio - expected_ratio) < 1e-10, \
        f"Path loss scaling incorrect: {actual_ratio} vs {expected_ratio}"
    
    print(f"  α(2000km): {alpha1:.6e}")
    print(f"  α(4000km): {alpha2:.6e}")
    print(f"  Ratio: {actual_ratio:.4f} (expected {expected_ratio:.4f})")
    print("✓ Path loss scaling test passed")


def test_angular_attenuation():
    """Test angular attenuation behavior."""
    print("Testing angular attenuation...")
    
    theta_B = 2e-3  # 2 mrad beamwidth
    sigma_e = 1e-6  # 1 μrad jitter
    
    # On-axis interference (θ_ℓm = 0)
    alpha_0 = calculate_alpha_lm_homogeneous(2000e3, 2000e3, 0, theta_B, sigma_e)
    
    # One beamwidth off-axis
    alpha_1 = calculate_alpha_lm_homogeneous(2000e3, 2000e3, theta_B, 
                                            theta_B, sigma_e)
    
    # Two beamwidths off-axis
    alpha_2 = calculate_alpha_lm_homogeneous(2000e3, 2000e3, 2*theta_B,
                                            theta_B, sigma_e)
    
    print(f"  α(0): {alpha_0:.6e}")
    print(f"  α(θ_B): {alpha_1:.6e} ({10*np.log10(alpha_1/alpha_0):.1f} dB)")
    print(f"  α(2θ_B): {alpha_2:.6e} ({10*np.log10(alpha_2/alpha_0):.1f} dB)")
    
    # Check monotonic decrease
    assert alpha_0 > alpha_1 > alpha_2, "Angular attenuation not monotonic"
    
    print("✓ Angular attenuation test passed")


def test_normalization():
    """Test interference coefficient normalization."""
    print("Testing normalization...")
    
    alpha = 0.01  # -20 dB interference
    snr0 = 100    # 20 dB SNR
    
    alpha_tilde = calculate_normalized_interference_coeff(alpha, snr0)
    
    print(f"  α = {alpha:.3f} ({10*np.log10(alpha):.1f} dB)")
    print(f"  SNR₀ = {snr0:.0f} ({10*np.log10(snr0):.1f} dB)")
    print(f"  α̃ = {alpha_tilde:.1f}")
    
    # In this case, normalized interference equals 1 (same as noise)
    assert abs(alpha_tilde - 1.0) < 1e-10, "Normalization incorrect"
    
    print("✓ Normalization test passed")


def test_operating_regimes():
    """Test regime classification."""
    print("Testing operating regime classification...")
    
    # Static regime
    regime1 = interference_operating_regime(theta_B=2e-3, sigma_e=1e-7)
    print(f"  θ_B=2mrad, σ_e=0.1μrad: {regime1}")
    assert "Static" in regime1
    
    # Transition regime
    regime2 = interference_operating_regime(theta_B=2e-3, sigma_e=1e-3)
    print(f"  θ_B=2mrad, σ_e=1mrad: {regime2}")
    assert "Transition" in regime2
    
    # Dynamic regime
    regime3 = interference_operating_regime(theta_B=2e-3, sigma_e=5e-3)
    print(f"  θ_B=2mrad, σ_e=5mrad: {regime3}")
    assert "Dynamic" in regime3
    
    print("✓ Regime classification test passed")


def test_network_aggregation():
    """Test network interference aggregation."""
    print("Testing network interference aggregation...")
    
    # Target link
    target = LinkParameters(power=1.0, gain_tx=1000, gain_rx=1000,
                          beamwidth=2e-3, sigma_e=1e-6, distance=2000e3)
    
    # Multiple interferers
    interferer1 = LinkParameters(power=1.0, gain_tx=1000, gain_rx=1000,
                                beamwidth=2e-3, sigma_e=1e-6, distance=2000e3)
    interferer2 = LinkParameters(power=2.0, gain_tx=1500, gain_rx=1000,
                                beamwidth=1.5e-3, sigma_e=2e-6, distance=1500e3)
    
    interferer_list = [
        (interferer1, 3000e3, 5e-3),   # 3000 km, 5 mrad offset
        (interferer2, 2500e3, 8e-3),   # 2500 km, 8 mrad offset
    ]
    
    # Calculate total interference
    total = calculate_network_interference(target, interferer_list)
    
    # Calculate individual contributions
    alpha1 = calculate_alpha_lm(target, interferer1, 3000e3, 5e-3)
    alpha2 = calculate_alpha_lm(target, interferer2, 2500e3, 8e-3)
    
    print(f"  Interferer 1: α = {alpha1:.6e}")
    print(f"  Interferer 2: α = {alpha2:.6e}")
    print(f"  Total: α = {total:.6e}")
    print(f"  Sum check: {alpha1 + alpha2:.6e}")
    
    assert abs(total - (alpha1 + alpha2)) < 1e-10, "Aggregation error"
    
    print("✓ Network aggregation test passed")


def test_effective_broadening():
    """Test effective beam broadening calculation."""
    print("Testing effective beam broadening...")
    
    theta_B = 2e-3  # 2 mrad
    
    # Case 1: No jitter
    eff1 = effective_beam_broadening(theta_B, 0)
    print(f"  No jitter: θ_eff = {eff1*1e3:.3f} mrad")
    
    # Case 2: Small jitter
    eff2 = effective_beam_broadening(theta_B, 0.5e-3)
    print(f"  Small jitter (0.5 mrad): θ_eff = {eff2*1e3:.3f} mrad")
    
    # Case 3: Large jitter
    eff3 = effective_beam_broadening(theta_B, 5e-3)
    print(f"  Large jitter (5 mrad): θ_eff = {eff3*1e3:.3f} mrad")
    
    assert eff1 < eff2 < eff3, "Broadening not monotonic with jitter"
    
    print("✓ Effective broadening test passed")


if __name__ == "__main__":
    """Run all unit tests and demonstrate usage."""
    print("=" * 60)
    print("Running Interference Module Unit Tests")
    print("=" * 60)
    
    test_homogeneous_consistency()
    test_path_loss_scaling()
    test_angular_attenuation()
    test_normalization()
    test_operating_regimes()
    test_network_aggregation()
    test_effective_broadening()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    # Demonstration of key insights
    print("\n" + "=" * 60)
    print("Demonstration: Key Physical Insights")
    print("=" * 60)
    
    # Scenario parameters
    print("\nScenario: 300 GHz THz ISL Network")
    print("-" * 40)
    d_target = 2000e3     # 2000 km
    d_interferer = 3000e3 # 3000 km
    theta_B = 2e-3        # 2 mrad beamwidth
    
    print(f"Target link distance: {d_target/1e3:.0f} km")
    print(f"Interferer distance: {d_interferer/1e3:.0f} km")
    print(f"Beamwidth: {theta_B*1e3:.1f} mrad")
    
    # Compare different jitter levels
    print("\nInterference vs. Angular Separation:")
    print("-" * 40)
    
    for sigma_e_urad in [0.1, 1.0, 10.0]:
        sigma_e = sigma_e_urad * 1e-6
        print(f"\nPointing jitter: {sigma_e_urad:.1f} μrad")
        print(f"Operating regime: {interference_operating_regime(theta_B, sigma_e)}")
        
        for theta_lm_mrad in [0, 1, 2, 5, 10]:
            theta_lm = theta_lm_mrad * 1e-3
            alpha = calculate_alpha_lm_homogeneous(d_target, d_interferer,
                                                  theta_lm, theta_B, sigma_e)
            alpha_db = 10 * np.log10(alpha) if alpha > 0 else -np.inf
            print(f"  θ = {theta_lm_mrad:2d} mrad: α = {alpha_db:6.1f} dB")
    
    # Key insight about effective broadening
    print("\n" + "=" * 60)
    print("Key Insight: Effective Beam Broadening")
    print("=" * 60)
    
    print("\nThe denominator θ²_B/2 + 2σ²_e reveals that:")
    print("- Pointing jitter effectively broadens the antenna pattern")
    print("- When σ_e << θ_B: Interference limited by antenna pattern")
    print("- When σ_e >> θ_B: Interference limited by platform stability")
    print("- Design trade-off: Narrow beams vs. pointing requirements")
    
    # Practical example
    print("\nPractical Example:")
    target = LinkParameters(power=1.0, gain_tx=10000, gain_rx=10000,
                          beamwidth=2e-3, sigma_e=1e-6, distance=2000e3)
    interferer = LinkParameters(power=1.0, gain_tx=10000, gain_rx=10000,
                              beamwidth=2e-3, sigma_e=1e-6, distance=2000e3)
    
    alpha = calculate_alpha_lm(target, interferer, 3000e3, 5e-3)
    print(f"Interference coefficient: {10*np.log10(alpha):.1f} dB")
    
    # With 20 dB SNR
    snr0 = 100
    alpha_tilde = calculate_normalized_interference_coeff(alpha, snr0)
    print(f"With SNR₀ = 20 dB, normalized α̃ = {alpha_tilde:.3f}")
    
    if alpha_tilde < 0.1:
        print("→ Interference is negligible (< -10 dB relative to noise)")
    elif alpha_tilde < 1.0:
        print("→ Interference is below noise level but non-negligible")
    else:
        print("→ Interference exceeds noise level - dominant impairment!")