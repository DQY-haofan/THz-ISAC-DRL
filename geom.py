"""
Geometry and Orbital Module for THz LEO-ISL ISAC Network Simulation
====================================================================

This module provides the fundamental geometric and orbital mechanics functionality
for simulating a LEO satellite constellation with inter-satellite links (ISLs).
All calculations use SI units (meters, seconds, radians).

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import warnings

# Physical constants (SI units - meters, seconds, radians)
EARTH_RADIUS = 6371000.0  # Earth radius in meters
EARTH_MU = 3.986004418e14  # Earth gravitational parameter (m³/s²)
SPEED_OF_LIGHT = 299792458.0  # Speed of light (m/s)


@dataclass
class OrbitalElements:
    """Keplerian orbital elements for satellite orbit description."""
    a: float  # Semi-major axis (m)
    e: float  # Eccentricity
    i: float  # Inclination (rad)
    raan: float  # Right ascension of ascending node (rad)
    omega: float  # Argument of periapsis (rad)
    nu: float  # True anomaly (rad)
    
    def __post_init__(self):
        """Validate orbital elements."""
        if self.a <= 0:
            raise ValueError("Semi-major axis must be positive")
        if not 0 <= self.e < 1:
            raise ValueError("Eccentricity must be in [0, 1) for closed orbits")
        if not 0 <= self.i <= np.pi:
            raise ValueError("Inclination must be in [0, π]")


@dataclass
class StateVector:
    """
    Satellite state vector containing kinematic and temporal parameters.
    
    Based on the system model from Section 2.1, each satellite state includes:
    - 3D position vector (m)
    - 3D velocity vector (m/s)
    - Clock bias (s)
    - Clock drift (s/s)
    """
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    clock_bias: float = 0.0  # Clock bias in seconds
    clock_drift: float = 0.0  # Clock drift in s/s
    timestamp: float = 0.0  # Time of state vector (s)
    
    def __post_init__(self):
        """Ensure numpy arrays and validate dimensions."""
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3D vector")
        if self.velocity.shape != (3,):
            raise ValueError("Velocity must be a 3D vector")
    
    def to_array(self) -> np.ndarray:
        """Convert state vector to 8x1 numpy array [p, v, b, b_dot]."""
        return np.concatenate([
            self.position,
            self.velocity,
            [self.clock_bias],
            [self.clock_drift]
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, timestamp: float = 0.0):
        """Create StateVector from 8x1 numpy array."""
        if arr.shape != (8,):
            raise ValueError("Array must have shape (8,)")
        return cls(
            position=arr[:3],
            velocity=arr[3:6],
            clock_bias=arr[6],
            clock_drift=arr[7],
            timestamp=timestamp
        )


class Satellite:
    """
    Represents a single satellite with orbital dynamics and state propagation.
    All internal calculations use SI units.
    """
    
    def __init__(self, 
                 sat_id: str,
                 initial_state: Optional[StateVector] = None,
                 orbital_elements: Optional[OrbitalElements] = None,
                 propagation_model: str = "keplerian"):
        """
        Initialize a satellite.
        
        Args:
            sat_id: Unique identifier for the satellite
            initial_state: Initial state vector (if provided)
            orbital_elements: Initial orbital elements (if provided)
            propagation_model: Type of propagation ("keplerian" or "linear")
        """
        self.sat_id = sat_id
        self.propagation_model = propagation_model
        
        # Initialize state
        if initial_state is not None:
            self.current_state = initial_state
        elif orbital_elements is not None:
            self.current_state = self._elements_to_state(orbital_elements)
        else:
            raise ValueError("Must provide either initial_state or orbital_elements")
        
        # Store initial state for reference
        self.initial_state = StateVector(
            position=self.current_state.position.copy(),
            velocity=self.current_state.velocity.copy(),
            clock_bias=self.current_state.clock_bias,
            clock_drift=self.current_state.clock_drift,
            timestamp=self.current_state.timestamp
        )
    
    def _elements_to_state(self, elements: OrbitalElements) -> StateVector:
        """Convert orbital elements to state vector (SI units)."""
        # Position in perifocal coordinates
        r_mag = elements.a * (1 - elements.e**2) / (1 + elements.e * np.cos(elements.nu))
        r_pqw = r_mag * np.array([
            np.cos(elements.nu),
            np.sin(elements.nu),
            0
        ])
        
        # Velocity in perifocal coordinates
        h = np.sqrt(EARTH_MU * elements.a * (1 - elements.e**2))
        v_pqw = (EARTH_MU / h) * np.array([
            -np.sin(elements.nu),
            elements.e + np.cos(elements.nu),
            0
        ])
        
        # Rotation matrices
        R3_W = self._rotation_z(-elements.raan)
        R1_i = self._rotation_x(-elements.i)
        R3_w = self._rotation_z(-elements.omega)
        
        # Transform to ECI frame
        transform = R3_W @ R1_i @ R3_w
        position = transform @ r_pqw
        velocity = transform @ v_pqw
        
        return StateVector(position=position, velocity=velocity)
    
    def _rotation_x(self, angle: float) -> np.ndarray:
        """Rotation matrix about x-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    def _rotation_z(self, angle: float) -> np.ndarray:
        """Rotation matrix about z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def propagate(self, t: float, dt: Optional[float] = None) -> StateVector:
        """
        Propagate satellite state to time t.
        
        Args:
            t: Target time (seconds from epoch)
            dt: Time step for numerical integration (if needed)
        
        Returns:
            Updated state vector at time t
        """
        if self.propagation_model == "linear":
            return self._propagate_linear(t)
        elif self.propagation_model == "keplerian":
            return self._propagate_keplerian(t)
        else:
            raise ValueError(f"Unknown propagation model: {self.propagation_model}")
    
    def _propagate_linear(self, t: float) -> StateVector:
        """
        Linear state propagation using state transition matrix.
        All calculations in SI units.
        """
        dt = t - self.current_state.timestamp
        
        # Construct state transition matrix (8x8)
        F = np.eye(8)
        F[0:3, 3:6] = dt * np.eye(3)  # Position += velocity * dt
        F[6, 7] = dt  # Clock bias += clock drift * dt
        
        # Current state as array
        x_current = self.current_state.to_array()
        
        # Propagate (without process noise for deterministic propagation)
        x_new = F @ x_current
        
        # Update state
        self.current_state = StateVector.from_array(x_new, timestamp=t)
        return self.current_state
    
    def _propagate_keplerian(self, t: float) -> StateVector:
        """
        Keplerian two-body orbital propagation.
        Uses analytical solution for unperturbed two-body motion.
        All calculations in SI units.
        """
        dt = t - self.initial_state.timestamp
        
        # Current position and velocity (m and m/s)
        r0 = self.initial_state.position
        v0 = self.initial_state.velocity
        
        # Solve Kepler's equation (simplified - using linear mean motion)
        r0_mag = np.linalg.norm(r0)
        v0_mag = np.linalg.norm(v0)
        
        # Orbital period and mean motion
        a = 1 / (2/r0_mag - v0_mag**2/EARTH_MU)  # Semi-major axis (m)
        n = np.sqrt(EARTH_MU / a**3)  # Mean motion (rad/s)
        
        # Simple circular orbit approximation for quick implementation
        if a > 0:  # Elliptical orbit
            # Rotate position and velocity vectors
            theta = n * dt
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Simplified 2D rotation in orbital plane
            r_new = r0 * cos_theta + (v0 / n) * sin_theta
            v_new = -r0 * n * sin_theta + v0 * cos_theta
        else:
            # Fallback to linear propagation for hyperbolic orbits
            r_new = r0 + v0 * dt
            v_new = v0
        
        # Update clock states
        clock_bias_new = self.current_state.clock_bias + self.current_state.clock_drift * dt
        
        self.current_state = StateVector(
            position=r_new,
            velocity=v_new,
            clock_bias=clock_bias_new,
            clock_drift=self.current_state.clock_drift,
            timestamp=t
        )
        
        return self.current_state


class Constellation:
    """
    Manages a collection of satellites forming a constellation.
    All calculations use SI units.
    """
    
    def __init__(self, satellites: Optional[List[Satellite]] = None):
        """
        Initialize constellation.
        
        Args:
            satellites: List of Satellite objects
        """
        self.satellites: Dict[str, Satellite] = {}
        if satellites:
            for sat in satellites:
                self.add_satellite(sat)
        
        self.current_time = 0.0
    
    def add_satellite(self, satellite: Satellite):
        """Add a satellite to the constellation."""
        if satellite.sat_id in self.satellites:
            warnings.warn(f"Satellite {satellite.sat_id} already exists, replacing...")
        self.satellites[satellite.sat_id] = satellite
    
    def remove_satellite(self, sat_id: str):
        """Remove a satellite from the constellation."""
        if sat_id in self.satellites:
            del self.satellites[sat_id]
        else:
            warnings.warn(f"Satellite {sat_id} not found in constellation")
    
    def propagate_all(self, t: float) -> Dict[str, StateVector]:
        """
        Propagate all satellites to time t.
        
        Args:
            t: Target time (seconds from epoch)
        
        Returns:
            Dictionary mapping satellite IDs to their state vectors
        """
        states = {}
        for sat_id, satellite in self.satellites.items():
            states[sat_id] = satellite.propagate(t)
        self.current_time = t
        return states
    
    def get_state_matrix(self) -> np.ndarray:
        """
        Get the complete network state vector as defined in Section 2.1.
        
        Returns:
            8N_v x 1 state vector for all satellites
        """
        n_sats = len(self.satellites)
        state_matrix = np.zeros((8 * n_sats, 1))
        
        for i, (_, satellite) in enumerate(self.satellites.items()):
            state_matrix[8*i:8*(i+1), 0] = satellite.current_state.to_array()
        
        return state_matrix


def calculate_geometry(satellite_i: Satellite, 
                      satellite_j: Satellite) -> Dict[str, np.ndarray]:
    """
    Calculate geometric relationships between two satellites.
    
    Args:
        satellite_i: First satellite
        satellite_j: Second satellite
    
    Returns:
        Dictionary containing:
        - 'distance': Euclidean distance (meters)
        - 'unit_vector': Unit line-of-sight vector from i to j
        - 'relative_position': Position vector from i to j (meters)
        - 'relative_velocity': Velocity vector from i to j (m/s)
    """
    # Extract state vectors (in meters from StateVector)
    p_i = satellite_i.current_state.position  # meters
    p_j = satellite_j.current_state.position  # meters
    v_i = satellite_i.current_state.velocity  # m/s
    v_j = satellite_j.current_state.velocity  # m/s
    
    # Relative kinematics
    relative_position = p_j - p_i  # meters
    relative_velocity = v_j - v_i  # m/s
    
    # Distance and unit vector
    distance = np.linalg.norm(relative_position)  # meters
    
    if distance > 0:
        unit_vector = relative_position / distance
    else:
        unit_vector = np.zeros(3)
        warnings.warn("Satellites at same position, unit vector undefined")
    
    return {
        'distance': distance,  # meters
        'unit_vector': unit_vector,
        'relative_position': relative_position,  # meters
        'relative_velocity': relative_velocity  # m/s
    }


def check_earth_obstruction(sat_i_pos: np.ndarray, 
                           sat_j_pos: np.ndarray,
                           earth_radius: float = EARTH_RADIUS) -> bool:
    """
    Check if Earth obstructs the line-of-sight between two satellites.
    
    Args:
        sat_i_pos: Position of first satellite (meters)
        sat_j_pos: Position of second satellite (meters)
        earth_radius: Earth radius (meters, default 6371000.0)
    
    Returns:
        True if line-of-sight is clear, False if obstructed
    """
    # Vector from sat_i to sat_j
    d = sat_j_pos - sat_i_pos
    
    # Closest point on line segment to Earth center
    t = -np.dot(sat_i_pos, d) / np.dot(d, d)
    t = np.clip(t, 0, 1)
    
    closest_point = sat_i_pos + t * d
    distance_to_earth = np.linalg.norm(closest_point)
    
    # Add margin for atmosphere (100 km = 100000 m)
    return distance_to_earth > (earth_radius + 100000)  # 100 km atmosphere in meters


def get_visibility_graph(constellation: Constellation,
                        t: float,
                        max_range: float = 5000000.0,  # 5000 km in meters
                        check_obstruction: bool = True) -> Set[Tuple[str, str]]:
    """
    Generate the network topology (active ISLs) at time t.
    
    Args:
        constellation: Constellation object
        t: Time instant (seconds)
        max_range: Maximum communication range (meters, default 5000 km)
        check_obstruction: Whether to check for Earth obstruction
    
    Returns:
        Set of tuples (sat_i_id, sat_j_id) representing active links
    """
    # Propagate all satellites to time t
    constellation.propagate_all(t)
    
    active_links = set()
    sat_ids = list(constellation.satellites.keys())
    
    # Check all satellite pairs
    for i in range(len(sat_ids)):
        for j in range(i + 1, len(sat_ids)):
            sat_i = constellation.satellites[sat_ids[i]]
            sat_j = constellation.satellites[sat_ids[j]]
            
            # Calculate geometry
            geometry = calculate_geometry(sat_i, sat_j)
            distance = geometry['distance']  # meters
            
            # Check range constraint
            if distance > max_range:
                continue
            
            # Check Earth obstruction
            if check_obstruction:
                if not check_earth_obstruction(
                    sat_i.current_state.position,  # meters
                    sat_j.current_state.position   # meters
                ):
                    continue
            
            # Add bidirectional link
            active_links.add((sat_ids[i], sat_ids[j]))
    
    return active_links


def calculate_jacobian_toa(satellite_i: Satellite,
                          satellite_j: Satellite) -> np.ndarray:
    """
    Calculate the measurement Jacobian for TOA measurement between two satellites.
    
    Based on Section 3.1, computes H_ℓ for a link ℓ between satellites i and j.
    
    Args:
        satellite_i: Transmitting satellite
        satellite_j: Receiving satellite
    
    Returns:
        1x16 Jacobian matrix for the two-satellite state vector
    """
    # Get geometry
    geometry = calculate_geometry(satellite_i, satellite_j)
    unit_vector = geometry['unit_vector']
    
    # Initialize Jacobian (1x16 for two satellites with 8 states each)
    H = np.zeros((1, 16))
    
    # Partial derivatives w.r.t positions (converted to time units)
    H[0, 0:3] = -unit_vector / SPEED_OF_LIGHT  # ∂h/∂p_i
    H[0, 8:11] = unit_vector / SPEED_OF_LIGHT  # ∂h/∂p_j
    
    # Partial derivatives w.r.t clock biases
    H[0, 6] = -1  # ∂h/∂b_i
    H[0, 14] = 1  # ∂h/∂b_j
    
    # Velocities and clock drifts don't appear in TOA measurement
    
    return H


# ============================================================================
# Unit Tests
# ============================================================================

def test_state_vector():
    """Test StateVector class functionality."""
    print("Testing StateVector...")
    
    # Test initialization (now in meters)
    pos = np.array([7000000.0, 0.0, 0.0])  # 7000 km in meters
    vel = np.array([0.0, 7500.0, 0.0])  # 7.5 km/s in m/s
    state = StateVector(position=pos, velocity=vel, clock_bias=1e-6)
    
    assert state.position.shape == (3,), "Position shape incorrect"
    assert state.velocity.shape == (3,), "Velocity shape incorrect"
    
    # Test array conversion
    arr = state.to_array()
    assert arr.shape == (8,), "State array shape incorrect"
    
    # Test from_array
    state2 = StateVector.from_array(arr)
    np.testing.assert_array_almost_equal(state2.position, pos)
    
    print("✓ StateVector tests passed")


def test_satellite_propagation():
    """Test satellite propagation."""
    print("Testing Satellite propagation...")
    
    # Create satellite in circular orbit (LEO at 700 km altitude)
    initial_state = StateVector(
        position=np.array([7071000.0, 0.0, 0.0]),  # Earth radius + 700 km altitude in meters
        velocity=np.array([0.0, 7546.0, 0.0])       # Circular orbit velocity in m/s
    )
    
    sat = Satellite("SAT1", initial_state=initial_state, propagation_model="linear")
    
    # Propagate for 100 seconds
    new_state = sat.propagate(100.0)
    
    # Check that satellite has moved
    assert not np.array_equal(new_state.position, initial_state.position), \
        "Satellite didn't move"
    
    print(f"  Initial position: {initial_state.position/1000} km")
    print(f"  Final position: {new_state.position/1000} km")
    print("✓ Satellite propagation tests passed")


def test_geometry_calculation():
    """Test geometric calculations between satellites."""
    print("Testing geometry calculations...")
    
    # Create two satellites (positions in meters)
    state1 = StateVector(
        position=np.array([7000000.0, 0.0, 0.0]),
        velocity=np.array([0.0, 7500.0, 0.0])
    )
    state2 = StateVector(
        position=np.array([0.0, 7000000.0, 0.0]),
        velocity=np.array([-7500.0, 0.0, 0.0])
    )
    
    sat1 = Satellite("SAT1", initial_state=state1)
    sat2 = Satellite("SAT2", initial_state=state2)
    
    # Calculate geometry
    geom = calculate_geometry(sat1, sat2)
    
    # Expected distance: sqrt(2) * 7000000
    expected_distance = np.sqrt(2) * 7000000
    assert abs(geom['distance'] - expected_distance) < 10, \
        f"Distance calculation error: {geom['distance']} vs {expected_distance}"
    
    print(f"  Distance: {geom['distance']/1000:.2f} km")
    print(f"  Unit vector: {geom['unit_vector']}")
    print("✓ Geometry calculation tests passed")


def test_visibility_graph():
    """Test network topology generation."""
    print("Testing visibility graph generation...")
    
    # Create a simple 3-satellite constellation (positions in meters)
    sats = []
    for i, angle in enumerate([0, 120, 240]):
        angle_rad = np.deg2rad(angle)
        pos = 7000000 * np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        vel = 7500 * np.array([-np.sin(angle_rad), np.cos(angle_rad), 0])
        state = StateVector(position=pos, velocity=vel)
        sats.append(Satellite(f"SAT{i+1}", initial_state=state))
    
    constellation = Constellation(sats)
    
    # Get visibility graph
    links = get_visibility_graph(constellation, t=0.0, max_range=15000000.0)
    
    print(f"  Active links: {links}")
    print(f"  Number of links: {len(links)}")
    
    # Should have 3 links in a triangle
    assert len(links) == 3, f"Expected 3 links, got {len(links)}"
    
    print("✓ Visibility graph tests passed")


def test_earth_obstruction():
    """Test Earth obstruction checking."""
    print("Testing Earth obstruction...")
    
    # Case 1: Clear line of sight (both satellites high above Earth, in meters)
    sat1_pos = np.array([8000000.0, 0.0, 0.0])
    sat2_pos = np.array([0.0, 8000000.0, 0.0])
    assert check_earth_obstruction(sat1_pos, sat2_pos), \
        "False obstruction detected for clear LOS"
    
    # Case 2: Obstructed (line passes through Earth)
    sat1_pos = np.array([7000000.0, 0.0, 0.0])
    sat2_pos = np.array([-7000000.0, 0.0, 0.0])
    assert not check_earth_obstruction(sat1_pos, sat2_pos), \
        "Failed to detect obstruction"
    
    print("✓ Earth obstruction tests passed")


if __name__ == "__main__":
    """Run all unit tests."""
    print("=" * 60)
    print("Running Geometry Module Unit Tests (SI Units)")
    print("=" * 60)
    
    test_state_vector()
    test_satellite_propagation()
    test_geometry_calculation()
    test_visibility_graph()
    test_earth_obstruction()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)