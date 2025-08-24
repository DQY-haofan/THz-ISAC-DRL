"""
Main Training and Evaluation Script for LEO-ISAC Multi-Agent Reinforcement Learning
====================================================================================

This is the central orchestrator that manages the complete training pipeline,
including environment setup, agent training, periodic evaluation against benchmarks,
and comprehensive logging of experimental results.

Integrates all modules: LEO_ISAC_Env, MADDPG_Agent, CentralizedSCASolver, and baselines.

Author: THz ISAC Research Team
Date: August 2025
"""
import sys
try:
    import google.colab
    IN_COLAB = True
    from tqdm.notebook import tqdm  # Use notebook version for Colab
    print("✓ Running in Google Colab - using notebook progress bars")
except ImportError:
    IN_COLAB = False
    from tqdm import tqdm  # Use standard tqdm
    print("✓ Running in standard environment - using terminal progress bars")
import time
import argparse
import yaml
import os
import json
import csv
from vectorized_env import SubprocVecEnv

import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import warnings
from collections import defaultdict
from typing import Dict, Tuple
warnings.filterwarnings('ignore')

# Import custom modules
from leo_isac_env import LEO_ISAC_Env, ConstellationConfig, HardwareConfig, ISACConfig
from maddpg_agent import MADDPG_Agent
from benchmark_sca import CentralizedSCASolver, NetworkState

# Optional: Import for TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Using CSV logging only.")


class ExperimentConfig:
    """Configuration container for experiment parameters."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from configuration dictionary."""
        # Environment configuration
        self.env_config = config_dict.get('environment', {})
        self.n_satellites = self.env_config.get('n_satellites', 4)
        self.altitude_km = self.env_config.get('altitude_km', 550)
        self.frequency_ghz = self.env_config.get('frequency_ghz', 300)
        self.bandwidth_ghz = self.env_config.get('bandwidth_ghz', 10)
        self.tx_power_dbm = self.env_config.get('tx_power_dbm', 30)
        
        # ISAC configuration
        self.isac_config = config_dict.get('isac', {})
        self.w_comm = self.isac_config.get('w_comm', 1.0)
        self.w_sens = self.isac_config.get('w_sens', 0.5)
        self.w_penalty = self.isac_config.get('w_penalty', 10.0)
        
        # Agent configuration
        self.agent_config = config_dict.get('agent', {})
        self.actor_lr = self.agent_config.get('actor_lr', 1e-4)
        self.critic_lr = self.agent_config.get('critic_lr', 1e-3)
        self.tau = self.agent_config.get('tau', 0.005)
        self.gamma = self.agent_config.get('gamma', 0.99)
        self.buffer_capacity = self.agent_config.get('buffer_capacity', 1000000)
        self.batch_size = self.agent_config.get('batch_size', 256)
        self.use_per = self.agent_config.get('use_prioritized_replay', True)
        self.use_difference_rewards = self.agent_config.get('use_difference_rewards', True)
        
        # Training configuration
        self.training_config = config_dict.get('training', {})
        self.max_episodes = self.training_config.get('max_episodes', 1000)
        self.max_steps = self.training_config.get('max_steps_per_episode', 100)
        self.warmup_episodes = self.training_config.get('warmup_episodes', 50)
        self.update_frequency = self.training_config.get('update_frequency', 1)
        self.noise_decay_rate = self.training_config.get('noise_decay_rate', 0.995)
        self.min_noise = self.training_config.get('min_noise', 0.01)
        
        # Evaluation configuration
        self.eval_config = config_dict.get('evaluation', {})
        self.eval_frequency = self.eval_config.get('frequency', 50)
        self.eval_episodes = self.eval_config.get('episodes', 10)
        self.benchmark_sca = self.eval_config.get('benchmark_sca', True)
        self.compare_baselines = self.eval_config.get('compare_baselines', True)
        
        # Logging configuration
        self.logging_config = config_dict.get('logging', {})
        self.log_dir = self.logging_config.get('log_dir', './logs')
        self.save_frequency = self.logging_config.get('save_frequency', 100)
        self.use_tensorboard = self.logging_config.get('use_tensorboard', True) and TENSORBOARD_AVAILABLE
        self.verbose = self.logging_config.get('verbose', True)
        
        # Device configuration
        self.device = config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Create configuration from command line arguments."""
        config_dict = {
            'environment': {
                'n_satellites': args.n_satellites,
                'altitude_km': args.altitude_km,
                'frequency_ghz': args.frequency_ghz,
            },
            'training': {
                'max_episodes': args.max_episodes,
                'max_steps_per_episode': args.max_steps,
            },
            'logging': {
                'log_dir': args.log_dir,
                'verbose': args.verbose,
            }
        }
        return cls(config_dict)


class BaselineAgent:
    """Simple baseline agents for performance comparison."""
    
    def __init__(self, agent_type: str = 'random'):
        """
        Initialize baseline agent.
        
        Args:
            agent_type: Type of baseline ('random', 'equal', 'greedy')
        """
        self.agent_type = agent_type
    
    def select_actions(self, env: LEO_ISAC_Env, observations: Dict) -> np.ndarray:
        """
        Select actions based on baseline policy.
        
        Args:
            env: Environment instance
            observations: Current observations
            
        Returns:
            Actions for all agents
        """
        n_agents = env.n_agents
        action_dim = 4  # Assuming 4 links per agent max
        
        if self.agent_type == 'random':
            # Random power allocation
            actions = np.random.rand(n_agents, action_dim) * 0.2
            
        elif self.agent_type == 'equal':
            # Equal power allocation to all links
            actions = np.ones((n_agents, action_dim)) * 0.2
            
        elif self.agent_type == 'greedy':
            # Greedy: maximum power to strongest links
            actions = np.zeros((n_agents, action_dim))
            for i in range(n_agents):
                # Allocate more power to first link (simplified)
                actions[i, 0] = 0.5
                actions[i, 1:] = 0.1
        
        else:
            raise ValueError(f"Unknown baseline type: {self.agent_type}")
        
        return actions


class LEOISACTrainer:
    """
    Main trainer class that orchestrates the training and evaluation pipeline.
    """
    
        def __init__(self, config: ExperimentConfig, args: argparse.Namespace):
        """
        Initialize trainer with config and command line arguments.
        Modified to use vectorized environments.
        """
        self.config = config
        self.args = args
        
        # Override config with command line arguments
        if args.n_satellites is not None:
            self.config.n_satellites = args.n_satellites
        if args.max_episodes is not None:
            self.config.max_episodes = args.max_episodes
        if args.batch_size is not None:
            self.config.batch_size = args.batch_size
        
        # Set device
        if args.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(args.device)
        print(f"Using device: {self.device}")
        
        # Create experiment directory
        self.experiment_dir = self._create_experiment_dir()
        
        # Save configuration
        self._save_config()
        
        # Initialize CSV loggers
        self.train_csv = CSVLogger(self.experiment_dir / 'logs', 'training.csv')
        self.train_csv.initialize([
            'episode', 'step', 'total_reward', 'comm_reward', 'sens_reward',
            'throughput_gbps', 'gdop_m', 'sinr_db', 'loss_actor', 'loss_critic'
        ])
        
        self.eval_csv = CSVLogger(self.experiment_dir / 'logs', 'evaluation.csv')
        self.eval_csv.initialize([
            'episode', 'algorithm', 'total_reward', 'throughput_gbps',
            'gdop_m', 'sinr_db', 'convergence_time'
        ])
        
        # ========== MODIFIED: Use vectorized environments ==========
        # Determine number of parallel environments
        self.num_parallel_envs = getattr(args, 'num_envs', 4)  # Default to 4 parallel envs
        
        # Create configurations for parallel environments
        env_configs = []
        for i in range(self.num_parallel_envs):
            env_config = {
                'constellation': {
                    'n_satellites': self.config.n_satellites,
                    'altitude_km': self.config.altitude_km,
                    'inclination_deg': getattr(self.config, 'inclination_deg', 53.0),
                    'propagation_model': getattr(self.config, 'propagation_model', 'keplerian'),
                    'max_isl_range_km': getattr(self.config, 'max_isl_range_km', 5000.0)
                },
                'hardware': {
                    'frequency_ghz': self.config.frequency_ghz,
                    'bandwidth_ghz': self.config.bandwidth_ghz,
                    'antenna_diameter_m': getattr(self.config, 'antenna_diameter_m', 0.5),
                    'tx_power_max_dbm': self.config.tx_power_dbm,
                    'noise_figure_db': getattr(self.config, 'noise_figure_db', 3.0),
                    'hardware_level': getattr(self.config, 'hardware_level', 'High-Performance')
                },
                'isac': {
                    'w_comm': self.config.w_comm,
                    'w_sens': self.config.w_sens,
                    'w_penalty': self.config.w_penalty,
                    'sensing_mode': getattr(self.config, 'sensing_mode', 'cooperative_orbit_determination'),
                    'min_links_for_sensing': getattr(self.config, 'min_links_for_sensing', 3),
                    'time_step_s': getattr(self.config, 'time_step_s', 1.0),
                    'episode_length': self.config.max_steps
                }
            }
            env_configs.append(env_config)
        
        # Initialize vectorized training environment
        print(f"Creating {self.num_parallel_envs} parallel training environments...")
        self.env = SubprocVecEnv(env_configs)
        self.n_agents = self.env.num_agents
        self.agent_ids = self.env.agent_ids
        
        # Create single evaluation environment (keep original for evaluation)
        self.eval_env = self._create_environment()
        # ========== END MODIFIED ==========
        
        # Initialize agent with device and GRU flag
        self.agent = self._create_agent()
        
        # Initialize baselines
        self.baselines = {
            'random': BaselineAgent('random'),
            'equal': BaselineAgent('equal'),
            'greedy': BaselineAgent('greedy')
        }
        
        # Initialize centralized solver (if enabled)
        if self.config.benchmark_sca:
            self.sca_solver = CentralizedSCASolver(
                max_iterations=20,
                convergence_threshold=1e-3,
                verbose=False
            )
        else:
            self.sca_solver = None
        
        # Initialize logging
        self.logger = self._initialize_logging()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = defaultdict(list)
        self.evaluation_results = []
        
        # Exploration noise schedule
        self.current_noise = 1.0
    
    
    def _create_experiment_dir(self) -> Path:
        """Create directory for experiment outputs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"leo_isac_exp_{timestamp}"
        exp_dir = Path(self.config.log_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (exp_dir / 'models').mkdir(exist_ok=True)
        (exp_dir / 'logs').mkdir(exist_ok=True)
        (exp_dir / 'results').mkdir(exist_ok=True)
        
        return exp_dir
    
    def _save_config(self):
        """Save configuration to experiment directory."""
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4, default=str)
    
    def _create_environment(self) -> LEO_ISAC_Env:
        """Create LEO-ISAC environment with configuration."""
        const_config = ConstellationConfig(
            n_satellites=self.config.n_satellites,
            altitude_km=self.config.altitude_km
        )
        
        hw_config = HardwareConfig(
            frequency_ghz=self.config.frequency_ghz,
            bandwidth_ghz=self.config.bandwidth_ghz,
            tx_power_max_dbm=self.config.tx_power_dbm
        )
        
        isac_config = ISACConfig(
            w_comm=self.config.w_comm,
            w_sens=self.config.w_sens,
            w_penalty=self.config.w_penalty,
            episode_length=self.config.max_steps
        )
        
        return LEO_ISAC_Env(const_config, hw_config, isac_config)
    
    def _create_agent(self) -> MADDPG_Agent:
        """Create MADDPG agent with configuration."""
        # Get dimensions from environment
        dummy_obs = self.env.reset()
        obs_dim = list(dummy_obs.values())[0].shape[0]
        action_dim = 4  # Max 4 links per agent
        
        agent = MADDPG_Agent(
            num_agents=self.env.n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            tau=self.config.tau,
            gamma=self.config.gamma,
            buffer_capacity=self.config.buffer_capacity,
            batch_size=self.config.batch_size,
            use_prioritized_replay=self.config.use_per,
            use_difference_rewards=self.config.use_difference_rewards,
            device=str(self.device),  # Pass device to agent
            use_gru=self.args.use_gru  # Use command line flag for GRU
        )
        
        # Load checkpoint if resuming
        if self.args.resume:
            print(f"Loading checkpoint from {self.args.resume}")
            agent.load_models(self.args.resume)
        
        return agent
    
    def _initialize_logging(self) -> Dict:
        """Initialize logging infrastructure with consistent filenames."""
        logger = {
            'csv_path': self.experiment_dir / 'logs' / 'training.csv',  # 改为 training.csv
            'eval_csv_path': self.experiment_dir / 'logs' / 'evaluation.csv'  # 改为 evaluation.csv
        }
        
        # Initialize TensorBoard if available
        if self.config.use_tensorboard:
            logger['tensorboard'] = SummaryWriter(
                log_dir=self.experiment_dir / 'tensorboard'
            )
        
        # Initialize CSV headers - 确保标头与 run_smoke_test.py 期望的一致
        with open(logger['csv_path'], 'w') as f:
            # 使用正确的标头名称
            f.write("episode,step,total_reward,comm_reward,sens_reward,throughput_gbps,gdop_m,sinr_db,loss_actor,loss_critic\n")
        
        with open(logger['eval_csv_path'], 'w') as f:
            # 评估CSV的标头
            f.write("episode,algorithm,total_reward,throughput_gbps,gdop_m,sinr_db,convergence_time\n")
        
        return logger
    
    def train(self):
        """
        Main training loop with Colab-compatible progress bars.
        """
        print("=" * 70)
        print("Starting LEO-ISAC MARL Training")
        print(f"Experiment: {self.experiment_dir.name}")
        print(f"Device: {self.config.device}")
        print(f"Agents: {self.env.n_agents}")
        print("=" * 70)
        
        # Initialize metrics for tracking
        best_reward = -float('inf')
        recent_rewards = []
        training_history = []
        
        # Create main progress bar (Colab-compatible)
        episode_pbar = tqdm(
            range(self.config.max_episodes),
            desc="Training Episodes",
            unit="ep",
            leave=True
        )
        
        for episode in episode_pbar:
            episode_start = time.time()
            
            # Run training episode
            episode_reward, episode_metrics = self._run_training_episode(episode)
            
            # Track metrics
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            avg_reward = np.mean(recent_rewards[-10:]) if recent_rewards else 0
            
            # Update exploration noise
            self._update_exploration_noise()
            
            # Log training progress
            self._log_training(episode, episode_reward, episode_metrics)
            
            # Store history for plotting
            training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'avg_reward': avg_reward,
                'throughput': episode_metrics.get('throughput', 0),
                'gdop': episode_metrics.get('gdop', np.inf)
            })
            
            # Update progress bar description with metrics
            episode_pbar.set_postfix({
                'R': f'{episode_reward:.2f}',
                'Avg': f'{avg_reward:.2f}',
                'Best': f'{best_reward:.2f}',
                'Tput': f"{episode_metrics.get('throughput', 0):.1f}",
                'ε': f'{self.current_noise:.3f}'
            })
            
            # Periodic evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                print(f"\n{'='*70}")
                print(f"Evaluation at Episode {episode + 1}")
                print(f"{'='*70}")
                self._evaluate(episode)
                print(f"{'='*70}\n")
            
            # Save models
            if (episode + 1) % self.config.save_frequency == 0:
                self._save_models(episode)
                if self.config.verbose:
                    print(f"✓ Models saved at episode {episode + 1}")
            
            # Periodic status update for Colab
            if IN_COLAB and episode % 10 == 0 and episode > 0:
                episode_time = time.time() - episode_start
                print(f"\n[Episode {episode}] "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg(10): {avg_reward:.2f} | "
                    f"Throughput: {episode_metrics.get('throughput', 0):.2f} Gbps | "
                    f"Time: {episode_time:.2f}s")
        
        # Store training history for analysis
        self.training_history = training_history
        
        # Final evaluation
        print("\n" + "=" * 70)
        print("Training Complete - Running Final Evaluation")
        print("=" * 70)
        self._evaluate(self.config.max_episodes, final=True)
        
        # Save final models
        self._save_models(self.config.max_episodes, final=True)
        
        # Generate summary report
        self._generate_summary_report()
        
        # Plot training curves if in Colab
        if IN_COLAB:
            self._plot_training_curves()
        
        print("\n✓ Training completed successfully!")
        print(f"Results saved to: {self.experiment_dir}")
    
    def _plot_training_curves(self):
        """Plot training curves (for Colab visualization)."""
        if not hasattr(self, 'training_history') or not self.training_history:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            history = self.training_history
            episodes = [h['episode'] for h in history]
            rewards = [h['reward'] for h in history]
            avg_rewards = [h['avg_reward'] for h in history]
            throughputs = [h['throughput'] for h in history]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Rewards
            axes[0, 0].plot(episodes, rewards, alpha=0.3, label='Raw')
            axes[0, 0].plot(episodes, avg_rewards, label='Avg(10)', linewidth=2)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Throughput
            axes[0, 1].plot(episodes, throughputs)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Throughput (Gbps)')
            axes[0, 1].set_title('Communication Performance')
            axes[0, 1].grid(True, alpha=0.3)
            
            # GDOP (if available)
            gdops = [h['gdop'] for h in history]
            valid_gdops = [g if g < np.inf else np.nan for g in gdops]
            if any(not np.isnan(g) for g in valid_gdops):
                axes[1, 0].plot(episodes, valid_gdops)
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('GDOP (m)')
                axes[1, 0].set_title('Sensing Performance')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Learning curve
            axes[1, 1].plot(episodes, np.cumsum(rewards) / (np.arange(len(rewards)) + 1))
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Return')
            axes[1, 1].set_title('Cumulative Average Return')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.experiment_dir / 'training_curves.png', dpi=150)
            plt.show()
            
            print("✓ Training curves plotted")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")

        def _run_training_episode(self, episode: int) -> Tuple[float, Dict]:
        """
        Run a single training episode with vectorized environments.
        Modified to handle batched data from parallel environments.
        """
        # Reset all environments
        observations = self.env.reset()  # Shape: (num_envs, num_agents, obs_dim)
        self.agent.reset_noise()
        self.agent.reset_hidden_states(batch_size=self.num_parallel_envs)
        
        episode_rewards = np.zeros(self.num_parallel_envs)
        episode_metrics = defaultdict(lambda: [])
        episode_dones = np.zeros(self.num_parallel_envs, dtype=bool)
        
        for step in range(self.config.max_steps):
            # Reshape observations for agent processing
            # From (num_envs, num_agents, obs_dim) to list of arrays
            obs_list = []
            for env_idx in range(self.num_parallel_envs):
                if not episode_dones[env_idx]:
                    obs_list.append(observations[env_idx])
            
            if len(obs_list) == 0:
                break  # All episodes done
            
            # Select actions for all active environments
            if episode < self.config.warmup_episodes:
                # Random actions during warmup
                actions = np.random.rand(self.num_parallel_envs, self.n_agents, 4) * 0.3
            else:
                # Get actions from agent for each environment
                actions = []
                for env_idx in range(self.num_parallel_envs):
                    if not episode_dones[env_idx]:
                        env_obs = [observations[env_idx, i] for i in range(self.n_agents)]
                        env_actions = self.agent.select_actions(
                            env_obs,
                            add_noise=True,
                            noise_scale=self.current_noise
                        )
                        actions.append(env_actions)
                    else:
                        actions.append(np.zeros((self.n_agents, 4)))
                actions = np.array(actions)
            
            # Environment step
            next_observations, rewards, dones, infos = self.env.step(actions)
            
            # Process results for each environment
            for env_idx in range(self.num_parallel_envs):
                if not episode_dones[env_idx]:
                    # Accumulate rewards
                    env_reward = np.mean(rewards[env_idx])
                    episode_rewards[env_idx] += env_reward
                    
                    # Store experience
                    if episode >= self.config.warmup_episodes:
                        self.agent.store_experience(
                            observations[env_idx],
                            actions[env_idx],
                            rewards[env_idx],
                            next_observations[env_idx],
                            np.array([dones[env_idx]] * self.n_agents),
                            infos[env_idx]
                        )
                    
                    # Collect metrics
                    episode_metrics['throughput'].append(infos[env_idx].get('total_throughput', 0))
                    episode_metrics['gdop'].append(infos[env_idx].get('gdop', np.inf))
                    
                    # Check if episode done
                    if dones[env_idx]:
                        episode_dones[env_idx] = True
            
            # Learn from experience (more frequently due to more data)
            if episode >= self.config.warmup_episodes and \
               step % max(1, self.config.update_frequency // self.num_parallel_envs) == 0:
                learn_metrics = self.agent.learn()
                for key, value in learn_metrics.items():
                    episode_metrics[f'learn_{key}'].append(value)
            
            # Update observations
            observations = next_observations
            
            # Check if all episodes are done
            if np.all(episode_dones):
                break
        
        # Aggregate metrics across parallel environments
        total_episode_reward = np.mean(episode_rewards)
        
        aggregated_metrics = {}
        for key, values in episode_metrics.items():
            if values:
                aggregated_metrics[key] = np.mean(values)
        
        return total_episode_reward, aggregated_metrics
    
    def _evaluate(self, episode: int, final: bool = False):
        """
        Evaluate current policy against benchmarks with progress indication to stderr.
        
        Args:
            episode: Current training episode
            final: Whether this is the final evaluation
        """
        print(f"\nEvaluation at Episode {episode}:", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        sys.stderr.flush()
        
        evaluation_results = {}
        
        # Create progress bar for evaluation (output to stderr)
        eval_algorithms = ['MADDPG']
        if self.config.compare_baselines:
            eval_algorithms.extend(['random', 'equal', 'greedy'])
        if self.config.benchmark_sca and self.sca_solver:
            eval_algorithms.append('SCA')
        
        eval_pbar = tqdm(
            eval_algorithms,
            desc="Evaluating",
            unit="algo",
            ncols=80,
            file=sys.stderr,  # Output to stderr
            leave=False
        )
        
        for algo_name in eval_pbar:
            eval_pbar.set_description(f"Eval {algo_name:8s}")
            
            if algo_name == 'MADDPG':
                metrics = self._evaluate_agent(self.agent, "MADDPG")
                evaluation_results['MADDPG'] = metrics
            elif algo_name in self.baselines:
                metrics = self._evaluate_baseline(self.baselines[algo_name], algo_name)
                evaluation_results[algo_name] = metrics
            elif algo_name == 'SCA' and self.sca_solver:
                metrics = self._evaluate_sca()
                evaluation_results['SCA'] = metrics
        
        eval_pbar.close()
        
        # Print comparison (to stdout for logging)
        self._print_evaluation_comparison(evaluation_results)
        
        # Log evaluation results
        self._log_evaluation(episode, evaluation_results)
        
        # Store for final analysis
        self.evaluation_results.append({
            'episode': episode,
            'results': evaluation_results
        })

    def print_status(self, message: str, level: str = "INFO"):
        """
        Print status message with timestamp.
        
        Args:
            message: Status message
            level: Message level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✓"
        }.get(level, "•")
        
        tqdm.write(f"[{timestamp}] {prefix} {message}")
    
    def _evaluate_agent(self, agent: MADDPG_Agent, name: str) -> Dict:
        """
        Evaluate a specific agent over multiple episodes.
        
        Args:
            agent: Agent to evaluate
            name: Agent name for logging
            
        Returns:
            Evaluation metrics
        """
        agent.set_eval_mode()
        
        total_rewards = []
        throughputs = []
        gdops = []
        sinrs = []
        
        for _ in range(self.config.eval_episodes):
            observations = self.eval_env.reset()
            agent.reset_hidden_states()
            
            episode_reward = 0
            episode_throughput = []
            episode_gdop = []
            
            obs_list = [observations[agent_id] for agent_id in self.eval_env.agent_ids]
            
            for step in range(self.config.max_steps):
                # Get actions without noise
                actions = agent.select_actions(obs_list, add_noise=False)
                
                # Convert to environment format
                action_dict = {}
                for i, agent_id in enumerate(self.eval_env.agent_ids):
                    # 使用 link_registry 而不是 link_states
                    agent_links = [
                        lid for lid, (tx, rx) in self.eval_env.link_registry.items()
                        if tx == agent_id
                    ]
                    
                    power_alloc = {}
                    for j, link_id in enumerate(agent_links[:4]):
                        if j < len(actions[i]):
                            power_alloc[link_id] = float(actions[i][j])
                    
                    action_dict[agent_id] = {
                        'power_allocation': power_alloc,
                        'beam_selection': {}
                    }
                
                # Step
                next_observations, rewards, done, info = self.eval_env.step(action_dict)
                
                # Collect metrics
                rewards_array = np.array([rewards[agent_id] for agent_id in self.eval_env.agent_ids])
                episode_reward += np.mean(rewards_array)
                episode_throughput.append(info.get('total_throughput', 0))
                episode_gdop.append(info.get('gdop', np.inf))
                
                obs_list = [next_observations[agent_id] for agent_id in self.eval_env.agent_ids]
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            throughputs.append(np.mean(episode_throughput))
            gdops.append(np.mean([g for g in episode_gdop if g < np.inf]))
        
        agent.set_train_mode()
        
        return {
            'total_reward': np.mean(total_rewards),
            'throughput': np.mean(throughputs),
            'gdop': np.nanmean(gdops),
            'reward_std': np.std(total_rewards)
        }
    
    def _evaluate_baseline(self, baseline: BaselineAgent, name: str) -> Dict:
        """
        Evaluate a baseline agent.
        
        Args:
            baseline: Baseline agent
            name: Baseline name
            
        Returns:
            Evaluation metrics
        """
        total_rewards = []
        throughputs = []
        
        for _ in range(self.config.eval_episodes):
            observations = self.eval_env.reset()
            episode_reward = 0
            episode_throughput = []
            
            for step in range(self.config.max_steps):
                # Get baseline actions
                actions = baseline.select_actions(self.eval_env, observations)
                
                # Convert to environment format
                action_dict = {}
                for i, agent_id in enumerate(self.eval_env.agent_ids):
                    # 使用 link_registry 而不是 link_states
                    agent_links = [
                        lid for lid, (tx, rx) in self.eval_env.link_registry.items()
                        if tx == agent_id
                    ]
                    
                    power_alloc = {}
                    for j, link_id in enumerate(agent_links[:4]):
                        if j < len(actions[i]):
                            power_alloc[link_id] = float(actions[i][j])
                    
                    action_dict[agent_id] = {
                        'power_allocation': power_alloc,
                        'beam_selection': {}
                    }
                
                # Step
                next_observations, rewards, done, info = self.eval_env.step(action_dict)
                
                rewards_array = np.array([rewards[agent_id] for agent_id in self.eval_env.agent_ids])
                episode_reward += np.mean(rewards_array)
                episode_throughput.append(info.get('total_throughput', 0))
                
                observations = next_observations
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            throughputs.append(np.mean(episode_throughput))
        
        return {
            'total_reward': np.mean(total_rewards),
            'throughput': np.mean(throughputs),
            'gdop': np.inf,  # Baselines don't optimize sensing
            'reward_std': np.std(total_rewards)
        }
    
    def _evaluate_sca(self) -> Dict:
        """
        Evaluate centralized SCA benchmark.
        
        Returns:
            SCA evaluation metrics
        """
        # Convert environment state to NetworkState for SCA solver
        network_state = self._env_to_network_state(self.eval_env)
        
        # Solve with SCA
        start_time = time.time()
        solution = self.sca_solver.solve(network_state)
        solve_time = time.time() - start_time
        
        return {
            'total_reward': solution.total_utility,
            'throughput': solution.communication_utility / 1e9,  # Convert to Gbps
            'gdop': 10.0,  # Placeholder - would compute from FIM
            'solve_time': solve_time,
            'converged': solution.converged
        }
    
    def _env_to_network_state(self, env: LEO_ISAC_Env) -> NetworkState:
        """
        Convert environment state to NetworkState for SCA solver.
        
        This provides the true high-fidelity channel state information
        from the physics engine to the centralized SCA benchmark.
        
        Args:
            env: Environment instance
            
        Returns:
            NetworkState for SCA solver with true channel gains
        """
        # Get true channel gains from physics interface
        physics_interface = env.physical_layer_interface
        
        # Get direct channel gains
        direct_channels = physics_interface.get_all_direct_channel_gains()
        
        # Get interference channel gains
        interference_channels = physics_interface.get_all_interference_channel_gains()
        
        # Extract active links and satellites
        active_links = [lid for lid, metrics in env.link_states.items() 
                    if metrics.get('active', False)]
        satellites = env.agent_ids
        
        # Build link mapping
        link_mapping = {}
        for link_id in active_links:
            if link_id in env.link_registry:
                link_mapping[link_id] = env.link_registry[link_id]
        
        # Set power and bandwidth budgets
        power_budgets = {sat: env.max_tx_power_w for sat in satellites}
        bandwidth_budgets = {sat: env.bandwidth_hz for sat in satellites}
        per_link_power_max = {link: env.max_tx_power_w / 2 for link in active_links}
        
        # Set QoS requirements (example: first 2 links have min SINR requirements)
        min_sinr = {}
        for i, link in enumerate(active_links[:2]):
            min_sinr[link] = 10.0  # 10 dB minimum SINR
        
        # Set weights
        link_weights = {link: 1.0 for link in active_links}
        
        # Get system parameters from environment
        noise_power = env.noise_power if hasattr(env, 'noise_power') else 1e-10
        frequency_hz = env.frequency_hz if hasattr(env, 'frequency_hz') else 300e9
        bandwidth_hz = env.bandwidth_hz if hasattr(env, 'bandwidth_hz') else 10e9
        
        return NetworkState(
            direct_channels=direct_channels,
            interference_channels=interference_channels,
            active_links=active_links,
            satellites=satellites,
            link_mapping=link_mapping,
            noise_power=noise_power,
            frequency_hz=frequency_hz,
            bandwidth_hz=bandwidth_hz,
            power_budgets=power_budgets,
            bandwidth_budgets=bandwidth_budgets,
            per_link_power_max=per_link_power_max,
            hpa_saturation_power=15.0,  # 15W saturation power
            hpa_smoothness=3.0,  # Rapp model parameter
            min_sinr=min_sinr,
            max_sensing_distortion=100.0,
            link_weights=link_weights,
            sensing_weight=env.isac_config.w_sens if hasattr(env, 'isac_config') else 0.1
        )
    
    def _print_evaluation_comparison(self, results: Dict):
        """Print formatted evaluation comparison."""
        print("\n" + "=" * 70)
        print("Performance Comparison:")
        print("-" * 70)
        print(f"{'Algorithm':<15} {'Reward':<12} {'Throughput':<15} {'GDOP':<12}")
        print("-" * 70)
        
        for name, metrics in results.items():
            reward = metrics.get('total_reward', 0)
            throughput = metrics.get('throughput', 0)
            gdop = metrics.get('gdop', np.inf)
            
            print(f"{name:<15} {reward:<12.2f} {throughput:<15.2f} Gbps {gdop:<12.2f} m")
        
        # Calculate improvement over baselines
        if 'MADDPG' in results and 'random' in results:
            improvement = (results['MADDPG']['total_reward'] - 
                         results['random']['total_reward']) / \
                        abs(results['random']['total_reward']) * 100
            print(f"\nMADDPG improvement over random: {improvement:.1f}%")
        
        if 'MADDPG' in results and 'SCA' in results:
            gap = (results['SCA']['total_reward'] - 
                  results['MADDPG']['total_reward']) / \
                 abs(results['SCA']['total_reward']) * 100
            print(f"Gap to centralized SCA: {gap:.1f}%")
        
        print("=" * 70)
    
    def _update_exploration_noise(self):
        """Update exploration noise with decay."""
        self.current_noise = max(
            self.config.min_noise,
            self.current_noise * self.config.noise_decay_rate
        )
    
    def _log_training(self, episode: int, reward: float, metrics: Dict):
        """Log training metrics with consistent column names."""
        # CSV logging with correct column names
        with open(self.logger['csv_path'], 'a') as f:
            # 提取各个奖励组件（如果有的话）
            comm_reward = metrics.get('comm_reward', 0)
            sens_reward = metrics.get('sens_reward', 0) 
            throughput = metrics.get('throughput', 0)
            gdop = metrics.get('gdop', np.inf)
            sinr_db = metrics.get('sinr_db', 0)
            loss_actor = metrics.get('learn_actor_loss_mean', 0)
            loss_critic = metrics.get('learn_critic_loss_mean', 0)
            
            f.write(f"{episode},{self.env.episode_step},{reward:.4f},"
                f"{comm_reward:.4f},{sens_reward:.4f},"
                f"{throughput:.4f},{gdop:.4f},{sinr_db:.4f},"
                f"{loss_actor:.4f},{loss_critic:.4f}\n")
        
        # 同时更新 train_csv logger
        self.train_csv.log({
            'episode': episode,
            'step': self.env.episode_step,
            'total_reward': reward,
            'comm_reward': comm_reward,
            'sens_reward': sens_reward,
            'throughput_gbps': throughput,
            'gdop_m': gdop,
            'sinr_db': sinr_db,
            'loss_actor': loss_actor,
            'loss_critic': loss_critic
        })
        
        # TensorBoard logging
        if self.config.use_tensorboard and 'tensorboard' in self.logger:
            writer = self.logger['tensorboard']
            writer.add_scalar('Training/Reward', reward, episode)
            writer.add_scalar('Training/Throughput', throughput, episode)
            writer.add_scalar('Training/GDOP', gdop, episode)
            writer.add_scalar('Training/NoiseLevel', self.current_noise, episode)
            
            # Learning metrics
            for key, value in metrics.items():
                if key.startswith('learn_'):
                    writer.add_scalar(f'Learning/{key[6:]}', value, episode)
    
    def _log_evaluation(self, episode: int, results: Dict):
        """Log evaluation results with consistent column names."""
        # CSV logging with correct column names
        with open(self.logger['eval_csv_path'], 'a') as f:
            for name, metrics in results.items():
                f.write(f"{episode},{name},{metrics.get('total_reward', 0):.4f},"
                    f"{metrics.get('throughput', 0):.4f},"
                    f"{metrics.get('gdop', np.inf):.4f},"
                    f"{metrics.get('sinr_db', 0):.4f},"
                    f"{metrics.get('convergence_time', 0):.4f}\n")
        
        # 同时更新 eval_csv logger
        for name, metrics in results.items():
            self.eval_csv.log({
                'episode': episode,
                'algorithm': name,
                'total_reward': metrics.get('total_reward', 0),
                'throughput_gbps': metrics.get('throughput', 0),
                'gdop_m': metrics.get('gdop', np.inf),
                'sinr_db': metrics.get('sinr_db', 0),
                'convergence_time': metrics.get('convergence_time', 0)
            })
        
        # TensorBoard logging
        if self.config.use_tensorboard and 'tensorboard' in self.logger:
            writer = self.logger['tensorboard']
            for name, metrics in results.items():
                writer.add_scalar(f'Evaluation/{name}/Reward', 
                                metrics.get('total_reward', 0), episode)
                writer.add_scalar(f'Evaluation/{name}/Throughput', 
                                metrics.get('throughput', 0), episode)
    
    def _save_models(self, episode: int, final: bool = False):
        """Save agent models."""
        save_dir = self.experiment_dir / 'models'
        
        if final:
            save_path = save_dir / 'final'
        else:
            save_path = save_dir / f'episode_{episode}'
        
        self.agent.save_models(str(save_path))
        
        if self.config.verbose:
            print(f"Models saved to {save_path}")
    
    def _generate_summary_report(self):
        """Generate final summary report of training."""
        report_path = self.experiment_dir / 'results' / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("LEO-ISAC MARL Training Summary Report\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Experiment: {self.experiment_dir.name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Episodes: {self.config.max_episodes}\n")
            f.write(f"Satellites: {self.config.n_satellites}\n\n")
            
            # Final evaluation results
            if self.evaluation_results:
                final_eval = self.evaluation_results[-1]['results']
                f.write("Final Performance:\n")
                f.write("-" * 40 + "\n")
                
                for name, metrics in final_eval.items():
                    f.write(f"\n{name}:\n")
                    f.write(f"  Total Reward: {metrics.get('total_reward', 0):.4f}\n")
                    f.write(f"  Throughput: {metrics.get('throughput', 0):.4f} Gbps\n")
                    f.write(f"  GDOP: {metrics.get('gdop', np.inf):.4f} m\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Training Complete\n")
            f.write("=" * 70 + "\n")
        
        print(f"\nSummary report saved to {report_path}")

def parse_arguments():
    """Enhanced argument parser with all necessary options."""
    parser = argparse.ArgumentParser(
        description='Train LEO-ISAC Multi-Agent Reinforcement Learning System'
    )
    
    # Core arguments
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to YAML configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for logs and results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Training device (auto detects CUDA)')
    
    # Environment arguments
    parser.add_argument('--n_satellites', type=int, default=None,
                       help='Override number of satellites from config')
    parser.add_argument('--use_gru', action='store_true',
                       help='Use GRU in actor/critic networks')
    parser.add_argument('--no_gru', dest='use_gru', action='store_false',
                       help='Use MLP baseline without GRU')
    parser.set_defaults(use_gru=False)  # Default to MLP for stability
    
    # Training arguments
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Override maximum episodes from config')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='Model save frequency (episodes)')
    
    # 添加向量化环境参数
    parser.add_argument('--num_envs', type=int, default=4,
                       help='Number of parallel environments for training')
    
    return parser.parse_args()


def setup_reproducibility(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for reproducibility")


class CSVLogger:
    """Simple CSV logger for training metrics."""
    
    def __init__(self, log_dir: Path, filename: str):
        """Initialize CSV logger."""
        self.log_path = log_dir / filename
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.writer = None
        
    def initialize(self, fieldnames: List[str]):
        """Initialize CSV with headers."""
        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.file.flush()
    
    def log(self, data: Dict):
        """Log a row of data."""
        if self.writer:
            self.writer.writerow(data)
            self.file.flush()
    
    def close(self):
        """Clean up environment resources."""
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'eval_env'):
            self.eval_env.close()


def main():
    """Main entry point with reproducibility setup."""
    # Parse arguments first
    args = parse_arguments()
    
    # 立即设置随机种子以确保可重现性 - 这是关键添加
    setup_reproducibility(args.seed)
    
    # Load configuration
    if args.config and Path(args.config).exists():
        config = ExperimentConfig.from_yaml(args.config)
        print(f"✓ Loaded configuration from {args.config}")
    else:
        config = ExperimentConfig.from_args(args)
        print("✓ Using command-line configuration")
    
    # Create and run trainer
    print(f"✓ Starting experiment with seed={args.seed}")
    trainer = LEOISACTrainer(config, args)
    trainer.train()
    
    print("\n✓ Experiment completed successfully!")

if __name__ == "__main__":
    main()