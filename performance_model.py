"""
Unified Performance Model Module for THz LEO-ISL ISAC Network Simulation
=========================================================================

This module serves as the central hub connecting all physical models to final
performance metrics. All distance calculations use SI units (meters).

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import Union, List, Optional, Dict, Tuple
from dataclasses import dataclass
import warnings

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


@dataclass
class PerformanceMetrics:
    """
    Container for comprehensive performance metrics.
    
    Attributes:
        sinr_eff: Effective SINR (linear scale)
        sinr_eff_db: Effective SINR in dB
        range_variance_m2: Range measurement error variance (m²)
        ranging_rmse: Ranging RMSE (meters)
        hardware_penalty_db: Performance loss due to hardware (dB)
        interference_penalty_db: Performance loss due to interference (dB)
    """
    sinr_eff: float
    sinr_eff_db: float
    range_variance_m2: float
    ranging_rmse: float
    hardware_penalty_db: float
    interference_penalty_db: float


def calculate_effective_sinr(snr0: float,
                            gamma_eff: float = 0.0,
                            sigma_phi_squared: float = 0.0,
                            normalized_interference: Union[float, List[float], np.ndarray] = 0.0,
                            hardware_on: bool = True,
                            interference_on: bool = True,
                            phase_noise_on: bool = True) -> float:
    """
    Calculate effective SINR incorporating all impairments.
    
    Implements the unified noise-normalized formula from Section III eq. (18):
    SINR_eff,ℓ = (SNR_0,ℓ · exp(-σ²_φ,ℓ)) / (1 + SNR_0,ℓ · Γ_eff,ℓ + Σ α̃_ℓm)
    
    Args:
        snr0: Pre-impairment SNR (linear scale, not dB)
        gamma_eff: Effective hardware quality factor (default 0 = ideal)
        sigma_phi_squared: Phase noise variance in rad² (default 0 = no phase noise)
        normalized_interference: Normalized interference coefficients α̃_ℓm
        hardware_on: Enable hardware impairments (default True)
        interference_on: Enable interference effects (default True)
        phase_noise_on: Enable phase noise effects (default True)
    
    Returns:
        Effective SINR (linear scale, not dB)
    """
    # Input validation
    if snr0 < 0:
        raise ValueError("SNR must be non-negative")
    if gamma_eff < 0 or gamma_eff > 1:
        warnings.warn("Hardware quality factor should be in [0, 1]")
    if sigma_phi_squared < 0:
        raise ValueError("Phase noise variance must be non-negative")
    
    # Apply configuration switches
    if not hardware_on:
        gamma_eff = 0.0
    if not phase_noise_on:
        sigma_phi_squared = 0.0
    
    # Calculate coherent power loss due to phase noise
    coherent_loss = np.exp(-sigma_phi_squared) if phase_noise_on else 1.0
    
    # Calculate total normalized interference
    if interference_on and normalized_interference is not None:
        if isinstance(normalized_interference, (list, np.ndarray)):
            interference_sum = np.sum(normalized_interference)
        else:
            interference_sum = float(normalized_interference)
    else:
        interference_sum = 0.0
    
    if interference_sum < 0:
        raise ValueError("Interference sum must be non-negative")
    
    # Calculate effective SINR using unified formula
    numerator = snr0 * coherent_loss
    denominator = 1 + snr0 * gamma_eff + interference_sum
    
    sinr_eff = numerator / denominator
    
    return sinr_eff


def calculate_range_variance_m2(sinr_eff: float,
                                sigma_phi_squared: float,
                                f_c: float,
                                kappa_wf: Optional[float] = None,
                                bandwidth: Optional[float] = None) -> float:
    """
    Calculate range measurement error variance in meters squared.
    
    THIS FUNCTION RETURNS VARIANCE IN m², NOT s².
    
    Args:
        sinr_eff: Effective SINR (linear scale)
        sigma_phi_squared: Phase noise variance (rad²)
        f_c: Carrier frequency (Hz)
        kappa_wf: Waveform constant κ_WF = 1/(8π²β²)
        bandwidth: Signal bandwidth (Hz), used if kappa_wf not provided
    
    Returns:
        Range measurement error variance in m² (METERS SQUARED)
    """
    # Input validation
    if sinr_eff <= 0:
        return np.inf  # Return infinity for invalid SINR
    if sigma_phi_squared < 0:
        raise ValueError("Phase noise variance must be non-negative")
    if f_c <= 0:
        raise ValueError("Carrier frequency must be positive")
    
    # Determine waveform constant
    if kappa_wf is None:
        if bandwidth is None:
            raise ValueError("Either kappa_wf or bandwidth must be provided")
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
        beta_rms = bandwidth / np.sqrt(12)
        kappa_wf = 1 / (8 * np.pi**2 * beta_rms**2)
    elif kappa_wf <= 0:
        raise ValueError("Waveform constant must be positive")
    
    # Calculate time variance components
    waveform_term = kappa_wf / sinr_eff
    phase_noise_term = sigma_phi_squared / (2 * np.pi * f_c)**2 if sigma_phi_squared > 0 else 0.0
    
    # Total time variance (s²)
    sigma_squared_time = waveform_term + phase_noise_term
    
    # Convert to range variance (m²) - THIS IS THE KEY CONVERSION
    sigma_squared_range_m2 = SPEED_OF_LIGHT**2 * sigma_squared_time
    
    return sigma_squared_range_m2


def calculate_ranging_rmse(range_variance_m2: float) -> float:
    """
    Convert range measurement variance to ranging RMSE.
    
    Args:
        range_variance_m2: Range measurement error variance in m² (METERS SQUARED)
    
    Returns:
        Ranging RMSE in meters
    """
    if range_variance_m2 < 0:
        raise ValueError("Range variance must be non-negative")
    if range_variance_m2 == np.inf:
        return np.inf
    
    # RMSE = sqrt(σ²_range) where σ²_range is in m²
    ranging_rmse_m = np.sqrt(range_variance_m2)
    
    return ranging_rmse_m


def analyze_performance_breakdown(snr0: float,
                                 gamma_eff: float,
                                 sigma_phi_squared: float,
                                 normalized_interference: Union[float, List[float]],
                                 f_c: float,
                                 bandwidth: float) -> PerformanceMetrics:
    """
    Comprehensive performance analysis with breakdown by impairment source.
    
    Args:
        snr0: Pre-impairment SNR (linear scale)
        gamma_eff: Hardware quality factor
        sigma_phi_squared: Phase noise variance (rad²)
        normalized_interference: Normalized interference coefficients
        f_c: Carrier frequency (Hz)
        bandwidth: Signal bandwidth (Hz)
    
    Returns:
        PerformanceMetrics object with comprehensive analysis
    """
    # Calculate SINR with all impairments
    sinr_full = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_squared, normalized_interference,
        hardware_on=True, interference_on=True, phase_noise_on=True
    )
    
    # Calculate SINR without hardware impairments (for penalty calculation)
    sinr_no_hw = calculate_effective_sinr(
        snr0, 0, sigma_phi_squared, normalized_interference,
        hardware_on=False, interference_on=True, phase_noise_on=True
    )
    
    # Calculate SINR without interference (for penalty calculation)
    sinr_no_int = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_squared, 0,
        hardware_on=True, interference_on=False, phase_noise_on=True
    )
    
    # Calculate range variance and ranging RMSE
    range_var = calculate_range_variance(
        sinr_full, sigma_phi_squared, f_c, bandwidth=bandwidth
    )
    ranging_rmse = calculate_ranging_rmse(range_var)
    
    # Calculate penalties in dB
    hardware_penalty = 10 * np.log10(sinr_no_hw / sinr_full) if sinr_full > 0 else np.inf
    interference_penalty = 10 * np.log10(sinr_no_int / sinr_full) if sinr_full > 0 else np.inf
    
    return PerformanceMetrics(
        sinr_eff=sinr_full,
        sinr_eff_db=10 * np.log10(sinr_full) if sinr_full > 0 else -np.inf,
        range_variance_m2=range_var,
        ranging_rmse=ranging_rmse,
        hardware_penalty_db=hardware_penalty,
        interference_penalty_db=interference_penalty
    )


def identify_limiting_regime(snr0: float,
                            gamma_eff: float,
                            normalized_interference_sum: float) -> str:
    """
    Identify the dominant performance-limiting factor.
    
    Args:
        snr0: Pre-impairment SNR (linear scale)
        gamma_eff: Hardware quality factor
        normalized_interference_sum: Sum of normalized interference
    
    Returns:
        String describing the limiting regime
    """
    # Calculate denominator terms
    noise_term = 1.0
    hardware_term = snr0 * gamma_eff
    interference_term = normalized_interference_sum
    
    # Find dominant term
    terms = {
        "Noise-limited": noise_term,
        "Hardware-limited": hardware_term,
        "Interference-limited": interference_term
    }
    
    dominant = max(terms, key=terms.get)
    
    # Add quantitative assessment
    total = noise_term + hardware_term + interference_term
    dominance_percentage = 100 * terms[dominant] / total
    
    return f"{dominant} ({dominance_percentage:.1f}% of degradation)"


def calculate_capacity_upper_bound(sinr_eff: float) -> float:
    """
    Calculate Shannon capacity upper bound.
    
    C = log₂(1 + SINR_eff) bits/symbol
    
    Args:
        sinr_eff: Effective SINR (linear scale)
    
    Returns:
        Capacity in bits/symbol
    """
    if sinr_eff < 0:
        raise ValueError("SINR must be non-negative")
    
    capacity = np.log2(1 + sinr_eff)
    return capacity


def calculate_hardware_ceiling(gamma_eff: float, 
                              sigma_phi_squared: float) -> Dict[str, float]:
    """
    Calculate hardware-imposed performance ceilings.
    
    As SNR → ∞, performance saturates at hardware-determined limits.
    
    Args:
        gamma_eff: Hardware quality factor
        sigma_phi_squared: Phase noise variance (rad²)
    
    Returns:
        Dictionary with ceiling values for SINR and capacity
    """
    if gamma_eff <= 0:
        return {"sinr_ceiling": np.inf, "capacity_ceiling": np.inf}
    
    # SINR ceiling from Section 2.3
    sinr_ceiling = np.exp(-sigma_phi_squared) / gamma_eff
    
    # Capacity ceiling
    capacity_ceiling = np.log2(1 + sinr_ceiling)
    
    return {
        "sinr_ceiling": sinr_ceiling,
        "sinr_ceiling_db": 10 * np.log10(sinr_ceiling),
        "capacity_ceiling": capacity_ceiling
    }


# ============================================================================
# Unit Tests
# ============================================================================

def test_effective_sinr():
    """Test effective SINR calculation with various configurations."""
    print("Testing effective SINR calculation...")
    
    # Test parameters
    snr0 = 100  # 20 dB
    gamma_eff = 0.01
    sigma_phi_sq = 1e-4
    interference = [0.5, 0.3, 0.1]
    
    # Test with all impairments
    sinr_full = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_sq, interference
    )
    print(f"  Full impairments: {10*np.log10(sinr_full):.2f} dB")
    
    # Test with hardware off
    sinr_no_hw = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_sq, interference, hardware_on=False
    )
    print(f"  Hardware off: {10*np.log10(sinr_no_hw):.2f} dB")
    
    # Test with interference off
    sinr_no_int = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_sq, interference, interference_on=False
    )
    print(f"  Interference off: {10*np.log10(sinr_no_int):.2f} dB")
    
    # Test with phase noise off
    sinr_no_pn = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_sq, interference, phase_noise_on=False
    )
    print(f"  Phase noise off: {10*np.log10(sinr_no_pn):.2f} dB")
    
    # Verify hierarchy
    assert sinr_full < sinr_no_hw, "Hardware should reduce SINR"
    assert sinr_full < sinr_no_int, "Interference should reduce SINR"
    assert sinr_full < sinr_no_pn, "Phase noise should reduce SINR"
    
    print("✓ Effective SINR tests passed")


def test_range_variance():
    """Test range variance calculation."""
    print("Testing range variance calculation...")
    
    # Test parameters
    sinr_eff = 50  # ~17 dB
    sigma_phi_sq = 1e-4
    f_c = 300e9  # 300 GHz
    bandwidth = 10e9  # 10 GHz
    
    # Calculate variance
    var = calculate_range_variance(
        sinr_eff, sigma_phi_sq, f_c, bandwidth=bandwidth
    )
    
    # Convert to ranging RMSE
    rmse = calculate_ranging_rmse(var)
    
    print(f"  Range variance: {var:.3e} m²")
    print(f"  Ranging RMSE: {rmse*1e3:.3f} mm")
    
    # Test error floor
    var_high_sinr = calculate_range_variance(
        1e6, sigma_phi_sq, f_c, bandwidth=bandwidth
    )
    rmse_floor = calculate_ranging_rmse(var_high_sinr)
    
    print(f"  Error floor (high SINR): {rmse_floor*1e3:.3f} mm")
    
    assert rmse > rmse_floor, "Should approach but not exceed floor"
    
    print("✓ Range variance tests passed")


def test_performance_breakdown():
    """Test comprehensive performance analysis."""
    print("Testing performance breakdown...")
    
    # System parameters
    metrics = analyze_performance_breakdown(
        snr0=100,  # 20 dB
        gamma_eff=0.01,
        sigma_phi_squared=1e-4,
        normalized_interference=[0.5, 0.3, 0.1],
        f_c=300e9,
        bandwidth=10e9
    )
    
    print(f"  Effective SINR: {metrics.sinr_eff_db:.1f} dB")
    print(f"  Ranging RMSE: {metrics.ranging_rmse*1e3:.2f} mm")
    print(f"  Hardware penalty: {metrics.hardware_penalty_db:.1f} dB")
    print(f"  Interference penalty: {metrics.interference_penalty_db:.1f} dB")
    
    assert metrics.hardware_penalty_db > 0, "Should have hardware penalty"
    assert metrics.interference_penalty_db > 0, "Should have interference penalty"
    
    print("✓ Performance breakdown tests passed")


def test_limiting_regime():
    """Test regime identification."""
    print("Testing limiting regime identification...")
    
    # Noise-limited case
    regime1 = identify_limiting_regime(snr0=1, gamma_eff=0.01, 
                                      normalized_interference_sum=0.1)
    print(f"  Low SNR: {regime1}")
    assert "Noise" in regime1
    
    # Hardware-limited case
    regime2 = identify_limiting_regime(snr0=1000, gamma_eff=0.05,
                                      normalized_interference_sum=0.1)
    print(f"  High SNR, poor hardware: {regime2}")
    assert "Hardware" in regime2
    
    # Interference-limited case
    regime3 = identify_limiting_regime(snr0=100, gamma_eff=0.001,
                                      normalized_interference_sum=50)
    print(f"  Strong interference: {regime3}")
    assert "Interference" in regime3
    
    print("✓ Regime identification tests passed")


def test_hardware_ceiling():
    """Test hardware-imposed performance ceilings."""
    print("Testing hardware ceiling calculation...")
    
    # Calculate ceilings for different hardware qualities
    for gamma, name in [(0.005, "State-of-the-Art"), 
                        (0.01, "High-Performance"),
                        (0.05, "Low-Cost")]:
        ceiling = calculate_hardware_ceiling(gamma, 1e-4)
        print(f"  {name}: SINR ceiling = {ceiling['sinr_ceiling_db']:.1f} dB, "
              f"Capacity = {ceiling['capacity_ceiling']:.2f} bits/symbol")
    
    # Verify ordering
    ceiling_good = calculate_hardware_ceiling(0.01, 1e-4)
    ceiling_bad = calculate_hardware_ceiling(0.05, 1e-4)
    
    assert ceiling_good["sinr_ceiling"] > ceiling_bad["sinr_ceiling"], \
        "Better hardware should have higher ceiling"
    
    print("✓ Hardware ceiling tests passed")


def test_unity_switches():
    """Test that switches properly disable effects."""
    print("Testing configuration switches...")
    
    snr0 = 100
    gamma = 0.05
    sigma_phi = 1e-3
    interference = 10.0
    
    # All effects on (baseline)
    sinr_all = calculate_effective_sinr(snr0, gamma, sigma_phi, interference)
    
    # All effects off (should equal SNR0)
    sinr_none = calculate_effective_sinr(
        snr0, gamma, sigma_phi, interference,
        hardware_on=False, interference_on=False, phase_noise_on=False
    )
    
    print(f"  All effects: {10*np.log10(sinr_all):.2f} dB")
    print(f"  No effects: {10*np.log10(sinr_none):.2f} dB")
    print(f"  Original SNR: {10*np.log10(snr0):.2f} dB")
    
    assert abs(sinr_none - snr0) < 1e-10, "Should equal original SNR with all off"
    
    print("✓ Configuration switch tests passed")


if __name__ == "__main__":
    """Run all unit tests."""
    print("=" * 60)
    print("Running Performance Model Unit Tests (SI Units)")
    print("=" * 60)
    
    test_effective_sinr()
    test_range_variance()
    test_performance_breakdown()
    test_limiting_regime()
    test_hardware_ceiling()
    test_unity_switches()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)