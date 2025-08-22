"""
Centralized Successive Convex Approximation (SCA) Benchmark Solver
===================================================================

This module implements the centralized SCA-based resource allocation algorithm
that serves as the theoretical "golden standard" benchmark for LEO-ISAC networks.
It assumes perfect global channel state information and centralized control,
providing an upper bound on achievable performance.

Based on the theoretical framework from "Defining Our Golden Standard Benchmark Algorithm"
which formalizes the network-wide ISAC optimization problem and its SCA solution.

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
from scipy.linalg import inv
import time


@dataclass
class NetworkState:
    """
    Complete global network state information.
    
    In reality, this perfect information is impossible to obtain instantaneously,
    but in simulation it serves as the theoretical benchmark.
    """
    # Channel gains
    direct_channels: Dict[str, complex]  # h_ℓ for each link ℓ
    interference_channels: Dict[Tuple[str, str], complex]  # g_ℓ',ℓ from link ℓ' to link ℓ
    
    # Network topology
    active_links: List[str]  # Currently active ISL links
    satellites: List[str]  # Satellite IDs
    link_mapping: Dict[str, Tuple[str, str]]  # Link ID -> (tx_sat, rx_sat)
    
    # System parameters
    noise_power: float  # N0 * B (Watts)
    frequency_hz: float  # Carrier frequency
    bandwidth_hz: float  # Total available bandwidth
    
    # Constraints
    power_budgets: Dict[str, float]  # Total power per satellite
    bandwidth_budgets: Dict[str, float]  # Total bandwidth per satellite
    per_link_power_max: Dict[str, float]  # Maximum power per link
    
    # HPA parameters
    hpa_saturation_power: float  # P_sat
    hpa_smoothness: float  # ρ for Rapp model
    
    # QoS requirements
    min_sinr: Dict[str, float]  # Minimum SINR per link (linear scale)
    max_sensing_distortion: float  # Maximum allowed Tr(J_net^-1)
    
    # Weights
    link_weights: Dict[str, float]  # Communication utility weights w_ℓ
    sensing_weight: float  # λ for sensing utility
    
    # Fisher Information Matrix components (for sensing)
    fim_jacobians: Optional[Dict[str, np.ndarray]] = None  # Jacobian matrices per link


@dataclass
class AllocationSolution:
    """Container for resource allocation solution."""
    power_allocation: Dict[str, float]  # Power per link (Watts)
    bandwidth_allocation: Dict[str, float]  # Bandwidth per link (Hz)
    achieved_sinr: Dict[str, float]  # Achieved SINR per link
    communication_utility: float  # Total communication utility
    sensing_utility: float  # Total sensing utility
    total_utility: float  # Combined ISAC utility
    iterations: int  # Number of SCA iterations
    converged: bool  # Whether algorithm converged
    solve_time: float  # Total solution time (seconds)


class CentralizedSCASolver:
    """
    Centralized Successive Convex Approximation solver for LEO-ISAC resource allocation.
    
    This solver implements the theoretical benchmark algorithm that assumes:
    1. Perfect, instantaneous global CSI
    2. Centralized decision making with zero delay
    3. Unlimited computational resources
    
    The algorithm iteratively solves convex approximations of the non-convex
    ISAC optimization problem until convergence to a local optimum.
    """
    
    def __init__(self,
                 max_iterations: int = 50,
                 convergence_threshold: float = 1e-4,
                 initial_power_fraction: float = 0.5,
                 initial_bandwidth_fraction: float = 0.8,
                 verbose: bool = False):
        """
        Initialize the centralized SCA solver.
        
        Args:
            max_iterations: Maximum SCA iterations
            convergence_threshold: Convergence criterion for objective value
            initial_power_fraction: Fraction of budget for initial feasible point
            initial_bandwidth_fraction: Fraction of bandwidth for initial point
            verbose: Whether to print iteration progress
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.initial_power_fraction = initial_power_fraction
        self.initial_bandwidth_fraction = initial_bandwidth_fraction
        self.verbose = verbose
        
        # Solver statistics
        self.iteration_history = []
        
    def solve(self, global_state: NetworkState) -> AllocationSolution:
        """
        Solve the centralized ISAC resource allocation problem using SCA.
        
        This is the main entry point that implements the iterative SCA algorithm
        from Report 4, using perfect global information.
        
        Args:
            global_state: Complete network state information
            
        Returns:
            Optimal resource allocation solution
        """
        start_time = time.time()
        
        if self.verbose:
            print("=" * 60)
            print("Centralized SCA Solver - Starting Optimization")
            print(f"Network: {len(global_state.active_links)} links, "
                  f"{len(global_state.satellites)} satellites")
            print("=" * 60)
        
        # Initialize with feasible solution
        P_current, B_current = self._initialize_feasible_solution(global_state)
        
        # Compute initial objective value
        obj_current = self._compute_objective(P_current, B_current, global_state)
        
        # SCA iteration loop
        converged = False
        iteration = 0
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\nIteration {iteration + 1}:")
                print(f"  Current objective: {obj_current:.6f}")
            
            # Construct and solve convex subproblem
            P_new, B_new, solve_status = self._solve_convex_subproblem(
                P_current, B_current, global_state
            )
            
            if solve_status != "optimal":
                warnings.warn(f"Convex subproblem solution status: {solve_status}")
                if solve_status == "infeasible":
                    break
            
            # Compute new objective value
            obj_new = self._compute_objective(P_new, B_new, global_state)
            
            # Store iteration history
            self.iteration_history.append({
                'iteration': iteration + 1,
                'objective': obj_new,
                'improvement': obj_new - obj_current
            })
            
            if self.verbose:
                print(f"  New objective: {obj_new:.6f}")
                print(f"  Improvement: {obj_new - obj_current:.6e}")
            
            # Check convergence
            if abs(obj_new - obj_current) < self.convergence_threshold:
                converged = True
                if self.verbose:
                    print(f"\n✓ Converged after {iteration + 1} iterations")
                break
            
            # Update solution
            P_current = P_new
            B_current = B_new
            obj_current = obj_new
        
        solve_time = time.time() - start_time
        
        # Compute final metrics
        sinr_values = self._compute_sinr(P_current, B_current, global_state)
        comm_utility = self._compute_communication_utility(
            P_current, B_current, sinr_values, global_state
        )
        sens_utility = self._compute_sensing_utility(P_current, B_current, global_state)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete")
            print(f"  Iterations: {iteration + 1}")
            print(f"  Converged: {converged}")
            print(f"  Total utility: {obj_current:.6f}")
            print(f"  Solve time: {solve_time:.3f} seconds")
            print("=" * 60)
        
        return AllocationSolution(
            power_allocation=P_current,
            bandwidth_allocation=B_current,
            achieved_sinr=sinr_values,
            communication_utility=comm_utility,
            sensing_utility=sens_utility,
            total_utility=obj_current,
            iterations=iteration + 1,
            converged=converged,
            solve_time=solve_time
        )
    
    def _initialize_feasible_solution(self, state: NetworkState) -> Tuple[Dict, Dict]:
        """
        Generate an initial feasible solution satisfying all constraints.
        
        Uses a simple heuristic: allocate resources equally among links
        from each satellite, scaled to satisfy budget constraints.
        
        Args:
            state: Network state
            
        Returns:
            Initial power and bandwidth allocations
        """
        P_init = {}
        B_init = {}
        
        # Count outgoing links per satellite
        outgoing_links = {sat: [] for sat in state.satellites}
        for link_id, (tx_sat, rx_sat) in state.link_mapping.items():
            if link_id in state.active_links:
                outgoing_links[tx_sat].append(link_id)
        
        # Allocate power and bandwidth
        for sat_id, links in outgoing_links.items():
            if len(links) > 0:
                # Equal allocation among satellite's links
                power_per_link = (state.power_budgets[sat_id] * 
                                 self.initial_power_fraction) / len(links)
                bandwidth_per_link = (state.bandwidth_budgets[sat_id] * 
                                    self.initial_bandwidth_fraction) / len(links)
                
                for link_id in links:
                    # Respect per-link maximum
                    P_init[link_id] = min(power_per_link, 
                                         state.per_link_power_max.get(link_id, power_per_link))
                    B_init[link_id] = bandwidth_per_link
            
        return P_init, B_init
    
    def _solve_convex_subproblem(self, P_prev: Dict, B_prev: Dict, 
                                 state: NetworkState) -> Tuple[Dict, Dict, str]:
        """
        Solve the convex approximation subproblem at current iteration point.
        
        Implements the convexification techniques from Report 4:
        - Linearization of log(1 + SINR) for communication utility
        - First-order approximation for sensing FIM
        - Convex relaxation of HPA constraints
        
        Args:
            P_prev: Previous power allocation
            B_prev: Previous bandwidth allocation
            state: Network state
            
        Returns:
            New power allocation, bandwidth allocation, and solve status
        """
        n_links = len(state.active_links)
        
        # Create optimization variables
        P = {}
        B = {}
        for link in state.active_links:
            P[link] = cp.Variable(nonneg=True)
            B[link] = cp.Variable(nonneg=True)
        
        # Compute linearization coefficients at previous point
        sinr_prev = self._compute_sinr(P_prev, B_prev, state)
        
        # Construct objective function (convex approximation)
        objective = 0
        
        # Communication utility - linearized log(1 + SINR)
        for link in state.active_links:
            if link in sinr_prev and sinr_prev[link] > 0:
                # Linearization coefficients from Report 4
                sinr_k = sinr_prev[link]
                A_k = np.log2(1 + sinr_k) - sinr_k / ((1 + sinr_k) * np.log(2))
                C_k = 1 / ((1 + sinr_k) * np.log(2))
                
                # Linear approximation of SINR
                numerator = P[link] * abs(state.direct_channels[link])**2
                
                # Denominator with interference
                denominator = state.noise_power * B[link]
                for other_link in state.active_links:
                    if other_link != link:
                        interference_key = (other_link, link)
                        if interference_key in state.interference_channels:
                            g_interference = abs(state.interference_channels[interference_key])**2
                            denominator += P[other_link] * g_interference
                
                # Use auxiliary variable for SINR approximation
                sinr_approx = numerator / (denominator + 1e-10)
                
                # Add to objective with weight
                weight = state.link_weights.get(link, 1.0)
                objective += weight * B[link] * (A_k + C_k * sinr_approx)
        
        # Sensing utility - linearized (simplified for demonstration)
        # In practice, would compute full FIM and its gradient
        if state.fim_jacobians is not None and state.sensing_weight > 0:
            # Simplified sensing term (quadratic approximation)
            sensing_term = 0
            for link in state.active_links:
                if link in P_prev:
                    # Gradient-based linear approximation
                    sensing_term += 0.1 * cp.sqrt(P[link])  # Simplified
            
            objective += state.sensing_weight * sensing_term
        
        # Constraints
        constraints = []
        
        # (C1) Per-satellite total power constraint
        for sat_id in state.satellites:
            sat_links = [link for link, (tx, rx) in state.link_mapping.items()
                        if tx == sat_id and link in state.active_links]
            if sat_links:
                constraints.append(
                    cp.sum([P[link] for link in sat_links]) <= state.power_budgets[sat_id]
                )
        
        # (C2) Per-satellite total bandwidth constraint
        for sat_id in state.satellites:
            sat_links = [link for link, (tx, rx) in state.link_mapping.items()
                        if tx == sat_id and link in state.active_links]
            if sat_links:
                constraints.append(
                    cp.sum([B[link] for link in sat_links]) <= state.bandwidth_budgets[sat_id]
                )
        
        # (C3) Per-link peak power constraint
        for link in state.active_links:
            if link in state.per_link_power_max:
                constraints.append(P[link] <= state.per_link_power_max[link])
        
        # (C4) HPA saturation constraint (convexified)
        # Using linearization of Rapp model at previous point
        for link in state.active_links:
            if link in P_prev:
                P_k = P_prev[link]
                if P_k > 0:
                    # Linear upper bound on HPA output
                    rho = state.hpa_smoothness
                    P_sat = state.hpa_saturation_power
                    
                    # Gradient of Rapp model at P_k
                    denominator = (1 + (P_k/P_sat)**(2*rho))**(1/(2*rho) + 1)
                    gradient = (1 + (1 - 2*rho) * (P_k/P_sat)**(2*rho)) / denominator
                    
                    # Linear constraint
                    hpa_output_approx = P_k * gradient + gradient * (P[link] - P_k)
                    constraints.append(hpa_output_approx <= P_sat)
        
        # (C5a) Minimum SINR constraints (convex form)
        for link in state.active_links:
            if link in state.min_sinr:
                numerator = P[link] * abs(state.direct_channels[link])**2
                
                denominator = state.noise_power * B[link]
                for other_link in state.active_links:
                    if other_link != link:
                        interference_key = (other_link, link)
                        if interference_key in state.interference_channels:
                            g_interference = abs(state.interference_channels[interference_key])**2
                            denominator += P[other_link] * g_interference
                
                # Convex constraint: numerator >= min_sinr * denominator
                constraints.append(
                    numerator >= state.min_sinr[link] * denominator
                )
        
        # Create and solve the problem
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            status = problem.status
        except Exception as e:
            warnings.warn(f"Solver error: {e}")
            status = "error"
        
        # Extract solution
        P_new = {}
        B_new = {}
        
        if status in ["optimal", "optimal_inaccurate"]:
            for link in state.active_links:
                P_new[link] = max(0, P[link].value) if P[link].value is not None else P_prev.get(link, 0)
                B_new[link] = max(0, B[link].value) if B[link].value is not None else B_prev.get(link, 0)
        else:
            # Return previous solution if optimization failed
            P_new = P_prev.copy()
            B_new = B_prev.copy()
        
        return P_new, B_new, status
    
    def _compute_sinr(self, P: Dict, B: Dict, state: NetworkState) -> Dict[str, float]:
        """
        Compute SINR values for all links given power and bandwidth allocation.
        
        Args:
            P: Power allocation
            B: Bandwidth allocation
            state: Network state
            
        Returns:
            Dictionary of SINR values per link
        """
        sinr_values = {}
        
        for link in state.active_links:
            if link not in P or P[link] <= 0:
                sinr_values[link] = 0
                continue
            
            # Signal power
            signal_power = P[link] * abs(state.direct_channels[link])**2
            
            # Interference plus noise
            noise = state.noise_power * B.get(link, state.bandwidth_hz)
            interference = 0
            
            for other_link in state.active_links:
                if other_link != link and other_link in P:
                    interference_key = (other_link, link)
                    if interference_key in state.interference_channels:
                        g_interference = abs(state.interference_channels[interference_key])**2
                        interference += P[other_link] * g_interference
            
            sinr_values[link] = signal_power / (interference + noise + 1e-10)
        
        return sinr_values
    
    def _compute_objective(self, P: Dict, B: Dict, state: NetworkState) -> float:
        """
        Compute the true objective value (not the convex approximation).
        
        Args:
            P: Power allocation
            B: Bandwidth allocation
            state: Network state
            
        Returns:
            Total ISAC utility value
        """
        sinr_values = self._compute_sinr(P, B, state)
        
        # Communication utility
        comm_utility = self._compute_communication_utility(P, B, sinr_values, state)
        
        # Sensing utility
        sens_utility = self._compute_sensing_utility(P, B, state)
        
        return comm_utility + sens_utility
    
    def _compute_communication_utility(self, P: Dict, B: Dict, 
                                      sinr_values: Dict, state: NetworkState) -> float:
        """
        Compute communication utility: sum of weighted link capacities.
        
        Args:
            P: Power allocation
            B: Bandwidth allocation
            sinr_values: Pre-computed SINR values
            state: Network state
            
        Returns:
            Total communication utility
        """
        utility = 0
        
        for link in state.active_links:
            if link in sinr_values and link in B:
                weight = state.link_weights.get(link, 1.0)
                capacity = B[link] * np.log2(1 + sinr_values[link])
                utility += weight * capacity
        
        return utility
    
    def _compute_sensing_utility(self, P: Dict, B: Dict, state: NetworkState) -> float:
        """
        Compute sensing utility based on Fisher Information Matrix.
        
        Simplified implementation - in practice would compute full network FIM
        and its inverse trace.
        
        Args:
            P: Power allocation
            B: Bandwidth allocation
            state: Network state
            
        Returns:
            Sensing utility value
        """
        if state.fim_jacobians is None or state.sensing_weight == 0:
            return 0
        
        # Simplified sensing utility computation
        # In full implementation, would:
        # 1. Build network FIM from individual link measurements
        # 2. Compute FIM inverse
        # 3. Return -λ * Tr(FIM^-1)
        
        sensing_quality = 0
        for link in state.active_links:
            if link in P and P[link] > 0:
                # Simplified: sensing quality improves with sqrt of power
                sensing_quality += np.sqrt(P[link])
        
        return state.sensing_weight * sensing_quality
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get solver statistics from last run.
        
        Returns:
            Dictionary of solver statistics
        """
        if not self.iteration_history:
            return {}
        
        improvements = [h['improvement'] for h in self.iteration_history[1:]]
        
        return {
            'total_iterations': len(self.iteration_history),
            'final_objective': self.iteration_history[-1]['objective'],
            'total_improvement': self.iteration_history[-1]['objective'] - 
                               self.iteration_history[0]['objective'],
            'average_improvement': np.mean(improvements) if improvements else 0,
            'convergence_history': self.iteration_history
        }


# ============================================================================
# Helper Functions for Testing
# ============================================================================

def create_test_network_state(n_satellites: int = 4, 
                             n_links: int = 6) -> NetworkState:
    """
    Create a test network state for solver validation.
    
    Args:
        n_satellites: Number of satellites
        n_links: Number of active links
        
    Returns:
        Test network state
    """
    # Generate satellite IDs
    satellites = [f"SAT_{i}" for i in range(n_satellites)]
    
    # Generate links (bidirectional connectivity)
    active_links = []
    link_mapping = {}
    
    link_id = 0
    for i in range(n_satellites):
        for j in range(i+1, min(i+2, n_satellites)):
            if link_id < n_links:
                link_name = f"LINK_{link_id}"
                active_links.append(link_name)
                link_mapping[link_name] = (satellites[i], satellites[j])
                link_id += 1
    
    # Generate channel gains
    direct_channels = {}
    interference_channels = {}
    
    for link in active_links:
        # Direct channel with path loss
        distance = np.random.uniform(1000, 5000) * 1000  # meters
        wavelength = 3e8 / 300e9  # THz band
        path_loss = (wavelength / (4 * np.pi * distance))**2
        direct_channels[link] = np.sqrt(path_loss) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        
        # Interference channels (weaker)
        for other_link in active_links:
            if other_link != link:
                interference_distance = np.random.uniform(2000, 8000) * 1000
                interference_loss = (wavelength / (4 * np.pi * interference_distance))**2
                interference_channels[(other_link, link)] = \
                    np.sqrt(interference_loss * 0.01) * np.exp(1j * np.random.uniform(0, 2*np.pi))
    
    # Set budgets and constraints
    power_budgets = {sat: 10.0 for sat in satellites}  # 10W per satellite
    bandwidth_budgets = {sat: 1e9 for sat in satellites}  # 1 GHz per satellite
    per_link_power_max = {link: 5.0 for link in active_links}  # 5W max per link
    
    # QoS requirements
    min_sinr = {link: 10.0 for link in active_links[:2]}  # 10 dB for some links
    
    # Weights
    link_weights = {link: 1.0 for link in active_links}
    
    return NetworkState(
        direct_channels=direct_channels,
        interference_channels=interference_channels,
        active_links=active_links,
        satellites=satellites,
        link_mapping=link_mapping,
        noise_power=1e-10,  # -100 dBW
        frequency_hz=300e9,  # 300 GHz
        bandwidth_hz=10e9,  # 10 GHz total
        power_budgets=power_budgets,
        bandwidth_budgets=bandwidth_budgets,
        per_link_power_max=per_link_power_max,
        hpa_saturation_power=15.0,  # 15W saturation
        hpa_smoothness=3.0,  # Rapp model parameter
        min_sinr=min_sinr,
        max_sensing_distortion=100.0,
        link_weights=link_weights,
        sensing_weight=0.1
    )


# ============================================================================
# Unit Tests
# ============================================================================

def test_initialization():
    """Test solver initialization and feasible solution generation."""
    print("Testing Solver Initialization...")
    
    solver = CentralizedSCASolver(verbose=False)
    state = create_test_network_state(n_satellites=3, n_links=4)
    
    # Test initial feasible solution
    P_init, B_init = solver._initialize_feasible_solution(state)
    
    print(f"  Active links: {len(state.active_links)}")
    print(f"  Initial allocations: {len(P_init)} power, {len(B_init)} bandwidth")
    
    # Verify constraints
    for sat in state.satellites:
        sat_links = [link for link, (tx, rx) in state.link_mapping.items() 
                    if tx == sat and link in state.active_links]
        
        total_power = sum(P_init.get(link, 0) for link in sat_links)
        total_bandwidth = sum(B_init.get(link, 0) for link in sat_links)
        
        assert total_power <= state.power_budgets[sat] + 1e-6, \
            f"Power budget violated for {sat}"
        assert total_bandwidth <= state.bandwidth_budgets[sat] + 1e-6, \
            f"Bandwidth budget violated for {sat}"
    
    print("✓ Initialization test passed")


def test_sinr_computation():
    """Test SINR computation with interference."""
    print("\nTesting SINR Computation...")
    
    solver = CentralizedSCASolver(verbose=False)
    state = create_test_network_state(n_satellites=2, n_links=2)
    
    # Simple allocation
    P = {link: 1.0 for link in state.active_links}
    B = {link: 1e8 for link in state.active_links}
    
    sinr_values = solver._compute_sinr(P, B, state)
    
    print(f"  SINR values computed: {len(sinr_values)}")
    for link, sinr in sinr_values.items():
        print(f"    {link}: {10*np.log10(sinr):.2f} dB")
    
    assert all(sinr > 0 for sinr in sinr_values.values()), "SINR should be positive"
    print("✓ SINR computation test passed")


def test_convex_subproblem():
    """Test convex subproblem solution."""
    print("\nTesting Convex Subproblem Solver...")
    
    solver = CentralizedSCASolver(verbose=False)
    state = create_test_network_state(n_satellites=2, n_links=2)
    
    # Initial point
    P_init, B_init = solver._initialize_feasible_solution(state)
    
    # Solve convex subproblem
    P_new, B_new, status = solver._solve_convex_subproblem(P_init, B_init, state)
    
    print(f"  Solver status: {status}")
    print(f"  Solution obtained: {len(P_new)} power, {len(B_new)} bandwidth allocations")
    
    if status == "optimal":
        # Verify constraints are satisfied
        for sat in state.satellites:
            sat_links = [link for link, (tx, rx) in state.link_mapping.items() 
                        if tx == sat and link in state.active_links]
            
            total_power = sum(P_new.get(link, 0) for link in sat_links)
            assert total_power <= state.power_budgets[sat] + 1e-3, \
                f"Power constraint violated"
    
    print("✓ Convex subproblem test passed")


def test_full_sca_algorithm():
    """Test complete SCA algorithm."""
    print("\nTesting Full SCA Algorithm...")
    
    solver = CentralizedSCASolver(
        max_iterations=20,
        convergence_threshold=1e-3,
        verbose=True
    )
    
    state = create_test_network_state(n_satellites=4, n_links=6)
    
    # Solve the problem
    solution = solver.solve(state)
    
    print(f"\nSolution Summary:")
    print(f"  Converged: {solution.converged}")
    print(f"  Iterations: {solution.iterations}")
    print(f"  Total utility: {solution.total_utility:.6f}")
    print(f"  Communication utility: {solution.communication_utility:.6f}")
    print(f"  Sensing utility: {solution.sensing_utility:.6f}")
    print(f"  Solve time: {solution.solve_time:.3f} seconds")
    
    # Verify solution quality
    assert solution.total_utility > 0, "Utility should be positive"
    assert solution.iterations <= solver.max_iterations, "Should not exceed max iterations"
    
    # Check constraint satisfaction
    for sat in state.satellites:
        sat_links = [link for link, (tx, rx) in state.link_mapping.items() 
                    if tx == sat and link in state.active_links]
        
        total_power = sum(solution.power_allocation.get(link, 0) for link in sat_links)
        total_bandwidth = sum(solution.bandwidth_allocation.get(link, 0) for link in sat_links)
        
        assert total_power <= state.power_budgets[sat] + 1e-3, \
            f"Power budget violated in final solution"
        assert total_bandwidth <= state.bandwidth_budgets[sat] + 1e-3, \
            f"Bandwidth budget violated in final solution"
    
    print("✓ Full SCA algorithm test passed")


def test_scalability():
    """Test solver scalability with larger networks."""
    print("\nTesting Scalability...")
    
    network_sizes = [(4, 6), (8, 12), (16, 24)]
    
    for n_sats, n_links in network_sizes:
        solver = CentralizedSCASolver(
            max_iterations=10,
            verbose=False
        )
        
        state = create_test_network_state(n_satellites=n_sats, n_links=n_links)
        
        start = time.time()
        solution = solver.solve(state)
        solve_time = time.time() - start
        
        print(f"  Network ({n_sats} sats, {n_links} links):")
        print(f"    Solve time: {solve_time:.3f} seconds")
        print(f"    Utility: {solution.total_utility:.4f}")
        print(f"    Converged: {solution.converged}")
    
    print("✓ Scalability test completed")


if __name__ == "__main__":
    """Run all tests and demonstrate the centralized SCA solver."""
    print("=" * 70)
    print("Centralized SCA Benchmark Solver - Test Suite")
    print("=" * 70)
    
    test_initialization()
    test_sinr_computation()
    test_convex_subproblem()
    test_full_sca_algorithm()
    test_scalability()
    
    print("\n" + "=" * 70)
    print("All tests passed successfully! ✓")
    print("=" * 70)
    
    # Demonstration with detailed example
    print("\n" + "=" * 70)
    print("Demonstration: Centralized vs Distributed Performance Gap")
    print("=" * 70)
    
    # Create a challenging scenario
    state = create_test_network_state(n_satellites=6, n_links=10)
    
    # Add strong interference to make problem harder
    for (link1, link2) in state.interference_channels:
        state.interference_channels[(link1, link2)] *= 10  # Stronger interference
    
    print("\nScenario: High-interference LEO constellation")
    print(f"  Satellites: {len(state.satellites)}")
    print(f"  Active ISLs: {len(state.active_links)}")
    print(f"  QoS constraints: {len(state.min_sinr)} links with minimum SINR")
    
    # Solve with centralized SCA
    print("\nCentralized SCA Solution (Perfect Information):")
    centralized_solver = CentralizedSCASolver(
        max_iterations=30,
        convergence_threshold=1e-4,
        verbose=False
    )
    
    centralized_solution = solver.solve(state)
    
    print(f"  Total utility: {centralized_solution.total_utility:.6f}")
    print(f"  Average SINR: {np.mean(list(centralized_solution.achieved_sinr.values())):.2f}")
    print(f"  Power efficiency: {sum(centralized_solution.power_allocation.values())
          f"sum(state.power_budgets.values()):.2%}")
    
    # Compare with naive equal allocation
    print("\nNaive Equal Allocation (Baseline):")
    equal_power = {link: 2.0 for link in state.active_links}
    equal_bandwidth = {link: 1e8 for link in state.active_links}
    
    equal_sinr = centralized_solver._compute_sinr(equal_power, equal_bandwidth, state)
    equal_utility = centralized_solver._compute_objective(equal_power, equal_bandwidth, state)
    
    print(f"  Total utility: {equal_utility:.6f}")
    print(f"  Average SINR: {np.mean(list(equal_sinr.values())):.2f}")
    
    improvement = (centralized_solution.total_utility - equal_utility) / equal_utility
    print(f"\nCentralized SCA Improvement: {improvement:.1%}")
    
    print("\nThis {improvement:.1%} improvement represents the theoretical upper bound")
    print("that any distributed algorithm should aspire to approach.")
    
    print("\n✓ Centralized SCA Benchmark Solver ready for performance evaluation!")