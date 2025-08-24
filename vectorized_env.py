"""
Vectorized Environment Wrapper for Parallel Training
=====================================================

This module implements a SubprocVecEnv wrapper that runs multiple instances
of LEO_ISAC_Env in parallel using multiprocessing, dramatically improving
data collection throughput for MARL training.

Author: THz ISAC Research Team
Date: December 2024
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from typing import List, Dict, Tuple, Optional, Any, Union
import cloudpickle
import warnings
import traceback
from copy import deepcopy

# Import the environment and configurations
from leo_isac_env import LEO_ISAC_Env, ConstellationConfig, HardwareConfig, ISACConfig


def worker_process(remote_conn, parent_conn, env_config: Dict):
    """
    Worker function that runs in a subprocess.
    
    This function:
    1. Creates an environment instance
    2. Listens for commands from the parent process
    3. Executes environment methods
    4. Sends results back through the pipe
    
    Args:
        remote_conn: Remote end of the pipe (used by worker)
        parent_conn: Parent end of the pipe (closed by worker)
        env_config: Configuration dictionary for creating the environment
    """
    # Close parent's end of the pipe in the worker
    parent_conn.close()
    
    try:
        # Create environment instance with provided config
        const_config = ConstellationConfig(**env_config.get('constellation', {}))
        hw_config = HardwareConfig(**env_config.get('hardware', {}))
        isac_config = ISACConfig(**env_config.get('isac', {}))
        
        env = LEO_ISAC_Env(const_config, hw_config, isac_config)
        
        # Get agent IDs for reference
        agent_ids = env.agent_ids
        
    except Exception as e:
        remote_conn.send(('error', f"Failed to create environment: {str(e)}"))
        remote_conn.close()
        return
    
    # Main worker loop
    while True:
        try:
            # Receive command from parent
            cmd, data = remote_conn.recv()
            
            if cmd == 'reset':
                # Reset environment
                obs_dict = env.reset()
                # Convert observation dict to array
                obs_array = dict_to_array_obs(obs_dict, agent_ids)
                remote_conn.send(('success', obs_array))
                
            elif cmd == 'step':
                # Convert action array to dict format expected by env
                action_dict = array_to_dict_action(data, agent_ids, env)
                
                # Execute step
                next_obs_dict, rewards_dict, done, info = env.step(action_dict)
                
                # Convert observations and rewards to arrays
                next_obs_array = dict_to_array_obs(next_obs_dict, agent_ids)
                rewards_array = dict_to_array_rewards(rewards_dict, agent_ids)
                
                remote_conn.send(('success', (next_obs_array, rewards_array, done, info)))
                
            elif cmd == 'close':
                # Clean up and exit
                env.close()
                remote_conn.close()
                break
                
            elif cmd == 'get_attr':
                # Get environment attribute
                attr_name = data
                value = getattr(env, attr_name)
                remote_conn.send(('success', value))
                
            elif cmd == 'set_attr':
                # Set environment attribute
                attr_name, value = data
                setattr(env, attr_name, value)
                remote_conn.send(('success', None))
                
            else:
                remote_conn.send(('error', f"Unknown command: {cmd}"))
                
        except Exception as e:
            # Send error information back to parent
            error_msg = f"Worker error: {str(e)}\n{traceback.format_exc()}"
            remote_conn.send(('error', error_msg))


def dict_to_array_obs(obs_dict: Dict[str, np.ndarray], agent_ids: List[str]) -> np.ndarray:
    """
    Convert observation dictionary to numpy array.
    
    Args:
        obs_dict: Dictionary mapping agent_id to observation
        agent_ids: Ordered list of agent IDs
        
    Returns:
        Array of shape (num_agents, obs_dim)
    """
    obs_list = [obs_dict[agent_id] for agent_id in agent_ids]
    return np.array(obs_list)


def dict_to_array_rewards(rewards_dict: Dict[str, float], agent_ids: List[str]) -> np.ndarray:
    """
    Convert rewards dictionary to numpy array.
    
    Args:
        rewards_dict: Dictionary mapping agent_id to reward
        agent_ids: Ordered list of agent IDs
        
    Returns:
        Array of shape (num_agents,)
    """
    rewards_list = [rewards_dict[agent_id] for agent_id in agent_ids]
    return np.array(rewards_list)


def array_to_dict_action(actions_array: np.ndarray, agent_ids: List[str], env) -> Dict:
    """
    Convert action array to dictionary format expected by LEO_ISAC_Env.
    
    Args:
        actions_array: Array of shape (num_agents, action_dim)
        agent_ids: Ordered list of agent IDs
        env: Environment instance (for accessing link_registry)
        
    Returns:
        Dictionary with format expected by env.step()
    """
    action_dict = {}
    
    for i, agent_id in enumerate(agent_ids):
        # Get links where this agent is transmitter
        agent_links = [
            lid for lid, (tx, rx) in env.link_registry.items()
            if tx == agent_id
        ]
        
        # Create power allocation dictionary
        power_alloc = {}
        for j, link_id in enumerate(agent_links[:4]):  # Max 4 links per agent
            if j < len(actions_array[i]):
                power_alloc[link_id] = float(actions_array[i][j])
        
        action_dict[agent_id] = {
            'power_allocation': power_alloc,
            'beam_selection': {}
        }
    
    return action_dict


class SubprocVecEnv:
    """
    Vectorized environment using subprocess workers.
    
    This class manages multiple environment instances running in parallel
    subprocesses, providing a batched interface for efficient training.
    """
    
    def __init__(self, env_configs: List[Dict], start_method: Optional[str] = None):
        """
        Initialize the vectorized environment.
        
        Args:
            env_configs: List of configuration dictionaries, one per environment
            start_method: Multiprocessing start method ('spawn', 'fork', 'forkserver')
        """
        self.num_envs = len(env_configs)
        
        if self.num_envs == 0:
            raise ValueError("Must provide at least one environment configuration")
        
        # Set multiprocessing start method if specified
        if start_method is not None:
            mp.set_start_method(start_method, force=True)
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        # Start worker processes
        self.processes = []
        for i, (work_remote, remote, config) in enumerate(
            zip(self.work_remotes, self.remotes, env_configs)
        ):
            # Make deep copy of config to avoid sharing between processes
            config_copy = deepcopy(config)
            
            # Create and start process
            process = Process(
                target=worker_process,
                args=(work_remote, remote, config_copy),
                daemon=True
            )
            process.start()
            self.processes.append(process)
            
            # Close worker's end of pipe in parent
            work_remote.close()
        
        # Get environment properties from first environment
        self.remotes[0].send(('get_attr', 'agent_ids'))
        status, result = self.remotes[0].recv()
        if status == 'success':
            self._agent_ids = result
            self.num_agents = len(self._agent_ids)
        else:
            raise RuntimeError(f"Failed to get agent_ids: {result}")
        
        # Get observation and action dimensions
        self.remotes[0].send(('reset', None))
        status, first_obs = self.remotes[0].recv()
        if status == 'success':
            self.obs_dim = first_obs.shape[1]  # (num_agents, obs_dim)
        else:
            raise RuntimeError(f"Failed to get observation dimension: {first_obs}")
        
        self.action_dim = 4  # Default max links per agent
        
        self.closed = False
    
    def reset(self) -> np.ndarray:
        """
        Reset all environments in parallel.
        
        Returns:
            Stacked observations of shape (num_envs, num_agents, obs_dim)
        """
        if self.closed:
            raise RuntimeError("Vectorized environment is closed")
        
        # Send reset command to all workers
        for remote in self.remotes:
            remote.send(('reset', None))
        
        # Collect results
        observations = []
        for remote in self.remotes:
            status, obs = remote.recv()
            if status == 'error':
                raise RuntimeError(f"Reset failed: {obs}")
            observations.append(obs)
        
        # Stack into single array
        return np.array(observations)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments in parallel.
        
        Args:
            actions: Actions array of shape (num_envs, num_agents, action_dim)
            
        Returns:
            Tuple of:
            - next_observations: (num_envs, num_agents, obs_dim)
            - rewards: (num_envs, num_agents)
            - dones: (num_envs,)
            - infos: List of info dictionaries
        """
        if self.closed:
            raise RuntimeError("Vectorized environment is closed")
        
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions.shape[0]}")
        
        # Send actions to all workers
        for i, remote in enumerate(self.remotes):
            remote.send(('step', actions[i]))
        
        # Collect results
        next_observations = []
        rewards = []
        dones = []
        infos = []
        
        for remote in self.remotes:
            status, result = remote.recv()
            if status == 'error':
                raise RuntimeError(f"Step failed: {result}")
            
            next_obs, reward, done, info = result
            next_observations.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        # Stack results
        next_observations = np.array(next_observations)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        return next_observations, rewards, dones, infos
    
    def close(self):
        """
        Close all worker processes and clean up resources.
        """
        if self.closed:
            return
        
        # Send close command to all workers
        for remote in self.remotes:
            remote.send(('close', None))
        
        # Wait for processes to terminate
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        
        # Close pipes
        for remote in self.remotes:
            remote.close()
        
        self.closed = True
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        if not self.closed:
            self.close()
    
    @property
    def agent_ids(self) -> List[str]:
        """Get list of agent IDs."""
        return self._agent_ids
    
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None):
        """
        Set attribute in specified environments.
        
        Args:
            attr_name: Name of attribute to set
            value: Value to set
            indices: Environment indices (None for all)
        """
        if indices is None:
            indices = range(self.num_envs)
        
        for i in indices:
            self.remotes[i].send(('set_attr', (attr_name, value)))
        
        for i in indices:
            status, _ = self.remotes[i].recv()
            if status == 'error':
                raise RuntimeError(f"Failed to set attribute {attr_name}")
    
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """
        Get attribute from specified environments.
        
        Args:
            attr_name: Name of attribute to get
            indices: Environment indices (None for all)
            
        Returns:
            List of attribute values
        """
        if indices is None:
            indices = range(self.num_envs)
        
        for i in indices:
            self.remotes[i].send(('get_attr', attr_name))
        
        results = []
        for i in indices:
            status, value = self.remotes[i].recv()
            if status == 'error':
                raise RuntimeError(f"Failed to get attribute {attr_name}: {value}")
            results.append(value)
        
        return results


# Test function
def test_vectorized_env():
    """Test the vectorized environment wrapper."""
    print("Testing SubprocVecEnv...")
    
    # Create test configurations
    num_envs = 4
    configs = []
    for i in range(num_envs):
        config = {
            'constellation': {
                'n_satellites': 3,
                'altitude_km': 550.0
            },
            'hardware': {
                'frequency_ghz': 300.0,
                'bandwidth_ghz': 10.0
            },
            'isac': {
                'episode_length': 50
            }
        }
        configs.append(config)
    
    # Create vectorized environment
    vec_env = SubprocVecEnv(configs)
    
    try:
        # Test reset
        obs = vec_env.reset()
        print(f"✔ Reset successful, obs shape: {obs.shape}")
        
        # Test step with random actions
        actions = np.random.rand(num_envs, vec_env.num_agents, vec_env.action_dim) * 0.1
        next_obs, rewards, dones, infos = vec_env.step(actions)
        
        print(f"✔ Step successful")
        print(f"  Next obs shape: {next_obs.shape}")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Dones shape: {dones.shape}")
        print(f"  Number of infos: {len(infos)}")
        
        # Run a few more steps
        for _ in range(5):
            actions = np.random.rand(num_envs, vec_env.num_agents, vec_env.action_dim) * 0.1
            next_obs, rewards, dones, infos = vec_env.step(actions)
        
        print("✔ Multiple steps successful")
        
    finally:
        # Clean up
        vec_env.close()
        print("✔ Environment closed successfully")


if __name__ == "__main__":
    test_vectorized_env()