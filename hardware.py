"""
Hardware and Channel Module for THz LEO-ISL ISAC Network Simulation
====================================================================

This module provides hardware impairment characterization and channel modeling
for THz inter-satellite links. It maps hardware quality levels to specific
impairment parameters and computes channel gains incorporating path loss and
pointing error effects.

Based on the theoretical framework from:
- Section III.B.2 of IEEE THz LEO-ISL paper (Hardware Quality Factor)
- Section 2.3 of System Model (Hardware Impairments)
- Section 4.2 of Network Interference (Pointing Error Effects)

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import Dict, Union, Optional
from dataclasses import dataclass
import warnings

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
BOLTZMANN_CONSTANT = 1.38064852e-23  # J/K


@dataclass
class HardwareProfile:
    """
    Hardware impairment profile for THz transceivers.
    
    Attributes:
        gamma_eff: Effective hardware quality factor (dimensionless)
        sigma_phi_squared: Phase noise variance (rad²)
        sigma_e: Pointing error standard deviation (radians)
        gamma_pa: PA nonlinearity factor
        gamma_lo: LO phase noise factor
        gamma_adc: ADC quantization factor
        description: Profile description
    """
    gamma_eff: float
    sigma_phi_squared: float
    sigma_e: float
    gamma_pa: float
    gamma_lo: float
    gamma_adc: float
    description: str
    
    def __post_init__(self):
        """Validate hardware parameters."""
        if self.gamma_eff <= 0 or self.gamma_eff > 1:
            raise ValueError("Hardware quality factor must be in (0, 1]")
        if self.sigma_phi_squared < 0:
            raise ValueError("Phase noise variance must be non-negative")
        if self.sigma_e < 0:
            raise ValueError("Pointing error must be non-negative")


# Hardware profiles based on Table I and Section IV-A of the IEEE paper
HARDWARE_PROFILES = {
    "State-of-the-Art": HardwareProfile(
        gamma_eff=0.005,
        sigma_phi_squared=1e-5,  # ~10 kHz linewidth, integrated variance
        sigma_e=0.5e-6,  # 0.5 μrad pointing jitter (ultra-stable platform)
        gamma_pa=0.0045,  # Advanced linearization
        gamma_lo=2e-7,
        gamma_adc=1e-4,
        description="State-of-the-art InP/InGaAs with advanced linearization"
    ),
    "High-Performance": HardwareProfile(
        gamma_eff=0.01,
        sigma_phi_squared=1e-4,  # ~100 kHz linewidth
        sigma_e=1e-6,  # 1 μrad pointing jitter
        gamma_pa=0.0112,  # InP DHBT/HEMT, 10.6% EVM
        gamma_lo=4.3e-7,  # 20.9 fs jitter
        gamma_adc=1.7e-4,
        description="InP DHBT/HEMT PAs with ultra-low jitter PLLs"
    ),
    "SWaP-Efficient": HardwareProfile(
        gamma_eff=0.045,
        sigma_phi_squared=5e-4,  # ~500 kHz linewidth
        sigma_e=2e-6,  # 2 μrad pointing jitter
        gamma_pa=0.0438,  # Silicon CMOS/SiGe, 20.93% EVM
        gamma_lo=4.8e-6,  # 70 fs jitter
        gamma_adc=6.5e-4,
        description="Silicon CMOS/SiGe with digital pre-distortion"
    ),
    "Low-Cost": HardwareProfile(
        gamma_eff=0.05,
        sigma_phi_squared=1e-3,  # ~1 MHz linewidth
        sigma_e=5e-6,  # 5 μrad pointing jitter
        gamma_pa=0.0475,
        gamma_lo=5e-6,
        gamma_adc=1e-3,
        description="Commercial off-the-shelf components"
    )
}


def get_hardware_params(level: str) -> HardwareProfile:
    """
    Get hardware impairment parameters for a given hardware quality level.
    
    Based on Section III.B.2 and Table I of the IEEE paper, mapping
    hardware quality levels to specific impairment parameters.
    
    Args:
        level: Hardware quality level, one of:
               - "State-of-the-Art": Γ_eff = 0.005
               - "High-Performance": Γ_eff = 0.01 (InP technology)
               - "SWaP-Efficient": Γ_eff = 0.045 (Silicon CMOS/SiGe)
               - "Low-Cost": Γ_eff = 0.05 (COTS components)
    
    Returns:
        HardwareProfile containing all impairment parameters
    
    Raises:
        ValueError: If level is not recognized
    """
    if level not in HARDWARE_PROFILES:
        available = ", ".join(HARDWARE_PROFILES.keys())
        raise ValueError(f"Unknown hardware level '{level}'. "
                        f"Available levels: {available}")
    
    return HARDWARE_PROFILES[level]


def calculate_fspl(distance_m: float, frequency_hz: float) -> float:
    """
    Calculate free-space path loss (FSPL).
    
    Args:
        distance_m: Link distance in meters
        frequency_hz: Carrier frequency in Hz
    
    Returns:
        Path loss as a power ratio (not in dB)
    """
    if distance_m <= 0:
        raise ValueError("Distance must be positive")
    if frequency_hz <= 0:
        raise ValueError("Frequency must be positive")
    
    # FSPL = (c / (4π d f))²
    wavelength = SPEED_OF_LIGHT / frequency_hz
    fspl = (wavelength / (4 * np.pi * distance_m))**2
    
    return fspl


def calculate_pointing_loss(sigma_e: float, theta_b: float) -> float:
    """
    Calculate average pointing error loss for Gaussian beam and jitter.
    
    Based on Section 4.2 of Network Interference document, equation (19):
    For nominal alignment (θ_ℓm = 0), the average power reduction factor is
    1 / (1 + 4σ_e²/θ_B²)
    
    Args:
        sigma_e: Pointing error standard deviation (radians)
        theta_b: 3dB beamwidth (radians)
    
    Returns:
        Average power reduction factor (≤ 1)
    """
    if sigma_e < 0:
        raise ValueError("Pointing error must be non-negative")
    if theta_b <= 0:
        raise ValueError("Beamwidth must be positive")
    
    # Average pointing loss factor
    loss_factor = 1.0 / (1.0 + 4 * (sigma_e / theta_b)**2)
    
    return loss_factor


def calculate_beamwidth(frequency_hz: float, antenna_diameter_m: float) -> float:
    """
    Calculate antenna 3dB beamwidth for a circular aperture.
    
    Using the approximation θ_3dB ≈ 1.02 λ/D for circular apertures.
    
    Args:
        frequency_hz: Carrier frequency in Hz
        antenna_diameter_m: Antenna diameter in meters
    
    Returns:
        3dB beamwidth in radians
    """
    if frequency_hz <= 0:
        raise ValueError("Frequency must be positive")
    if antenna_diameter_m <= 0:
        raise ValueError("Antenna diameter must be positive")
    
    wavelength = SPEED_OF_LIGHT / frequency_hz
    theta_3db = 1.02 * wavelength / antenna_diameter_m
    
    return theta_3db


def calculate_antenna_gain(frequency_hz: float, antenna_diameter_m: float,
                          efficiency: float = 0.55) -> float:
    """
    Calculate antenna gain for a circular aperture.
    
    Args:
        frequency_hz: Carrier frequency in Hz
        antenna_diameter_m: Antenna diameter in meters
        efficiency: Antenna efficiency (default 0.55 for typical reflector)
    
    Returns:
        Antenna gain as a power ratio (not in dB)
    """
    if efficiency <= 0 or efficiency > 1:
        raise ValueError("Antenna efficiency must be in (0, 1]")
    
    wavelength = SPEED_OF_LIGHT / frequency_hz
    area = np.pi * (antenna_diameter_m / 2)**2
    gain = efficiency * (4 * np.pi * area) / wavelength**2
    
    return gain


def calculate_channel_gain(d: float, f_c: float, G_T: float, G_R: float,
                          sigma_e: float, theta_B: float) -> float:
    """
    Calculate average channel power gain including path loss and pointing errors.
    
    Implements the channel model from Section 2.2 with pointing error effects
    from Section 4.2. The average channel gain incorporates:
    1. Free-space path loss (Friis equation)
    2. Average pointing error loss for both transmit and receive
    
    Args:
        d: Link distance in meters
        f_c: Carrier frequency in Hz
        G_T: Transmit antenna gain (power ratio)
        G_R: Receive antenna gain (power ratio)
        sigma_e: Pointing error standard deviation (radians)
        theta_B: 3dB beamwidth (radians)
    
    Returns:
        Average channel power gain |g|² including all effects
    
    Example:
        >>> d = 2000e3  # 2000 km
        >>> f_c = 300e9  # 300 GHz
        >>> G_T = calculate_antenna_gain(f_c, 0.5)  # 0.5m antenna
        >>> G_R = G_T
        >>> theta_B = calculate_beamwidth(f_c, 0.5)
        >>> sigma_e = 1e-6  # 1 μrad
        >>> gain = calculate_channel_gain(d, f_c, G_T, G_R, sigma_e, theta_B)
    """
    # Input validation
    if d <= 0:
        raise ValueError("Distance must be positive")
    if f_c <= 0:
        raise ValueError("Frequency must be positive")
    if G_T <= 0 or G_R <= 0:
        raise ValueError("Antenna gains must be positive")
    if sigma_e < 0:
        raise ValueError("Pointing error must be non-negative")
    if theta_B <= 0:
        raise ValueError("Beamwidth must be positive")
    
    # Free-space path loss
    fspl = calculate_fspl(d, f_c)
    
    # Average pointing loss (assuming same for Tx and Rx)
    pointing_loss = calculate_pointing_loss(sigma_e, theta_B)
    
    # Total average channel power gain
    # |g|² = G_T * G_R * FSPL * pointing_loss
    # Note: pointing_loss already accounts for both Tx and Rx in the average
    channel_gain = G_T * G_R * fspl * pointing_loss
    
    return channel_gain


def calculate_effective_sinr(snr0: float, gamma_eff: float, 
                            sigma_phi_squared: float,
                            interference_sum: float = 0.0) -> float:
    """
    Calculate effective SINR incorporating hardware impairments.
    
    Based on equation (13) from Section 2.3:
    SINR_eff = (SNR_0 * exp(-σ²_φ)) / (1 + SNR_0 * Γ_eff + Σα_ℓm)
    
    Args:
        snr0: Pre-impairment SNR (linear scale)
        gamma_eff: Hardware quality factor
        sigma_phi_squared: Phase noise variance (rad²)
        interference_sum: Sum of interference coefficients (default 0)
    
    Returns:
        Effective SINR (linear scale)
    """
    if snr0 < 0:
        raise ValueError("SNR must be non-negative")
    if gamma_eff <= 0:
        raise ValueError("Hardware quality factor must be positive")
    if sigma_phi_squared < 0:
        raise ValueError("Phase noise variance must be non-negative")
    if interference_sum < 0:
        raise ValueError("Interference sum must be non-negative")
    
    # Coherent power loss due to phase noise
    coherent_loss = np.exp(-sigma_phi_squared)
    
    # Effective SINR with hardware impairments
    sinr_eff = (snr0 * coherent_loss) / (1 + snr0 * gamma_eff + interference_sum)
    
    return sinr_eff


def calculate_noise_power(bandwidth_hz: float, 
                         noise_figure_db: float = 3.0,
                         temperature_k: float = 290.0) -> float:
    """
    Calculate thermal noise power.
    
    Args:
        bandwidth_hz: Signal bandwidth in Hz
        noise_figure_db: Receiver noise figure in dB (default 3 dB)
        temperature_k: System noise temperature in Kelvin (default 290 K)
    
    Returns:
        Noise power in Watts
    """
    if bandwidth_hz <= 0:
        raise ValueError("Bandwidth must be positive")
    if temperature_k <= 0:
        raise ValueError("Temperature must be positive")
    
    # Convert noise figure from dB
    noise_figure = 10**(noise_figure_db / 10)
    
    # Thermal noise power: N0 = k_B * T * B * F
    noise_power = BOLTZMANN_CONSTANT * temperature_k * bandwidth_hz * noise_figure
    
    return noise_power


def db_to_linear(value_db: float) -> float:
    """Convert dB to linear scale."""
    return 10**(value_db / 10)


def linear_to_db(value_linear: float) -> float:
    """Convert linear scale to dB."""
    if value_linear <= 0:
        return -np.inf
    return 10 * np.log10(value_linear)


# ============================================================================
# Unit Tests
# ============================================================================

def test_hardware_profiles():
    """Test hardware profile retrieval."""
    print("Testing hardware profiles...")
    
    # Test all defined profiles
    for level in ["State-of-the-Art", "High-Performance", 
                  "SWaP-Efficient", "Low-Cost"]:
        profile = get_hardware_params(level)
        assert profile.gamma_eff > 0, f"Invalid gamma_eff for {level}"
        assert profile.sigma_phi_squared >= 0, f"Invalid phase noise for {level}"
        print(f"  {level}: Γ_eff = {profile.gamma_eff:.3f}, "
              f"σ²_φ = {profile.sigma_phi_squared:.1e}")
    
    # Test invalid profile
    try:
        get_hardware_params("InvalidLevel")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ Hardware profile tests passed")


def test_fspl():
    """Test free-space path loss calculation."""
    print("Testing FSPL calculation...")
    
    # Test case: 2000 km at 300 GHz
    d = 2000e3  # meters
    f = 300e9   # Hz
    
    fspl = calculate_fspl(d, f)
    fspl_db = linear_to_db(fspl)
    
    # Expected FSPL (approximate)
    # FSPL_dB = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
    expected_db = 20*np.log10(d) + 20*np.log10(f) + 20*np.log10(4*np.pi/SPEED_OF_LIGHT)
    
    assert abs(fspl_db - expected_db) < 0.1, f"FSPL mismatch: {fspl_db} vs {expected_db}"
    
    print(f"  FSPL at 2000 km, 300 GHz: {fspl_db:.1f} dB")
    print("✓ FSPL tests passed")


def test_pointing_loss():
    """Test pointing error loss calculation."""
    print("Testing pointing loss calculation...")
    
    # Test case: 1 μrad jitter, 2 mrad beamwidth
    sigma_e = 1e-6  # radians
    theta_b = 2e-3  # radians
    
    loss = calculate_pointing_loss(sigma_e, theta_b)
    loss_db = linear_to_db(loss)
    
    print(f"  Pointing loss (σ_e=1μrad, θ_B=2mrad): {loss_db:.3f} dB")
    
    # Test extreme case: large jitter
    sigma_e_large = 1e-3  # 1 mrad jitter
    loss_large = calculate_pointing_loss(sigma_e_large, theta_b)
    loss_large_db = linear_to_db(loss_large)
    
    print(f"  Pointing loss (σ_e=1mrad, θ_B=2mrad): {loss_large_db:.3f} dB")
    
    assert loss > loss_large, "Larger jitter should cause more loss"
    
    print("✓ Pointing loss tests passed")


def test_channel_gain():
    """Test complete channel gain calculation."""
    print("Testing channel gain calculation...")
    
    # Test scenario
    d = 2000e3  # 2000 km
    f_c = 300e9  # 300 GHz
    D_ant = 0.5  # 0.5 m antenna
    
    # Calculate antenna parameters
    G_T = calculate_antenna_gain(f_c, D_ant)
    G_R = G_T
    theta_B = calculate_beamwidth(f_c, D_ant)
    
    # Hardware parameters
    sigma_e = 1e-6  # 1 μrad
    
    # Calculate channel gain
    gain = calculate_channel_gain(d, f_c, G_T, G_R, sigma_e, theta_B)
    gain_db = linear_to_db(gain)
    
    print(f"  Antenna gain: {linear_to_db(G_T):.1f} dBi")
    print(f"  Beamwidth: {theta_B*1e3:.2f} mrad")
    print(f"  Total channel gain: {gain_db:.1f} dB")
    
    # Verify reasonable values
    assert -200 < gain_db < -100, f"Unreasonable channel gain: {gain_db} dB"
    
    print("✓ Channel gain tests passed")


def test_effective_sinr():
    """Test effective SINR calculation with impairments."""
    print("Testing effective SINR calculation...")
    
    # Test parameters
    snr0 = 100  # 20 dB pre-impairment SNR
    gamma_eff = 0.01  # High-performance hardware
    sigma_phi_squared = 1e-4
    
    # Without interference
    sinr_eff = calculate_effective_sinr(snr0, gamma_eff, sigma_phi_squared, 0)
    sinr_eff_db = linear_to_db(sinr_eff)
    
    print(f"  Pre-impairment SNR: {linear_to_db(snr0):.1f} dB")
    print(f"  Effective SINR (no interference): {sinr_eff_db:.1f} dB")
    
    # With interference
    interference = 0.1  # -10 dB interference
    sinr_with_int = calculate_effective_sinr(snr0, gamma_eff, sigma_phi_squared, 
                                            interference)
    sinr_with_int_db = linear_to_db(sinr_with_int)
    
    print(f"  Effective SINR (with interference): {sinr_with_int_db:.1f} dB")
    
    assert sinr_with_int < sinr_eff, "Interference should reduce SINR"
    
    print("✓ Effective SINR tests passed")


if __name__ == "__main__":
    """Run all unit tests."""
    print("=" * 60)
    print("Running Hardware Module Unit Tests")
    print("=" * 60)
    
    test_hardware_profiles()
    test_fspl()
    test_pointing_loss()
    test_channel_gain()
    test_effective_sinr()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    # Example usage demonstration
    print("\n" + "=" * 60)
    print("Example: Complete Link Budget Calculation")
    print("=" * 60)
    
    # System parameters
    distance = 2000e3  # 2000 km
    frequency = 300e9  # 300 GHz
    tx_power_dbm = 30  # 30 dBm = 1 W
    antenna_diameter = 0.5  # 0.5 m
    bandwidth = 10e9  # 10 GHz
    
    # Get hardware profile
    hw_profile = get_hardware_params("High-Performance")
    
    # Calculate link parameters
    G_ant = calculate_antenna_gain(frequency, antenna_diameter)
    theta_3db = calculate_beamwidth(frequency, antenna_diameter)
    channel_gain = calculate_channel_gain(distance, frequency, G_ant, G_ant,
                                         hw_profile.sigma_e, theta_3db)
    
    # Calculate SNR
    tx_power = 10**((tx_power_dbm - 30) / 10)  # Convert dBm to Watts
    noise_power = calculate_noise_power(bandwidth)
    snr0 = (tx_power * channel_gain) / noise_power
    
    # Calculate effective SINR
    sinr_eff = calculate_effective_sinr(snr0, hw_profile.gamma_eff,
                                       hw_profile.sigma_phi_squared)
    
    print(f"Link distance: {distance/1e3:.0f} km")
    print(f"Frequency: {frequency/1e9:.0f} GHz")
    print(f"Tx power: {tx_power_dbm:.0f} dBm")
    print(f"Antenna diameter: {antenna_diameter:.1f} m")
    print(f"Hardware profile: {hw_profile.description}")
    print("-" * 40)
    print(f"Antenna gain: {linear_to_db(G_ant):.1f} dBi")
    print(f"Beamwidth: {theta_3db*1e3:.2f} mrad")
    print(f"Channel gain: {linear_to_db(channel_gain):.1f} dB")
    print(f"Noise power: {linear_to_db(noise_power)+30:.1f} dBm")
    print(f"Pre-impairment SNR: {linear_to_db(snr0):.1f} dB")
    print(f"Effective SINR: {linear_to_db(sinr_eff):.1f} dB")
    print(f"Hardware penalty: {linear_to_db(snr0) - linear_to_db(sinr_eff):.1f} dB")