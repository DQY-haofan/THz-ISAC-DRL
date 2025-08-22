#!/usr/bin/env python3
"""
Quick sanity check for core components
"""

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    try:
        from leo_isac_env import LEO_ISAC_Env
        from maddpg_agent import MADDPG_Agent
        from networks import ActorNetworkMLP
        from physical_layer_interface import PhysicalLayerInterface
        print("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_environment():
    """Test environment creation and step."""
    print("\nTesting environment...")
    try:
        from leo_isac_env import LEO_ISAC_Env, ConstellationConfig, HardwareConfig, ISACConfig
        
        # Minimal config
        const_config = ConstellationConfig(n_satellites=2)
        hw_config = HardwareConfig()
        isac_config = ISACConfig(episode_length=10)
        
        # Create environment
        env = LEO_ISAC_Env(const_config, hw_config, isac_config)
        
        # Reset
        obs = env.reset()
        print(f"✓ Environment reset: {len(obs)} agents")
        
        # Random step
        actions = {}
        for agent_id in env.agent_ids:
            actions[agent_id] = {
                'power_allocation': {f"LINK_0": 0.1},
                'beam_selection': {}
            }
        
        obs, rewards, done, info = env.step(actions)
        print(f"✓ Environment step: rewards={list(rewards.values())}")
        
        return True
    except Exception as e:
        print(f"✗ Environment error: {e}")
        return False

def test_networks():
    """Test neural network creation."""
    print("\nTesting networks...")
    try:
        from networks import ActorNetworkMLP, ActorNetwork
        
        # Test MLP version
        actor_mlp = ActorNetworkMLP(
            obs_dim=20,
            action_dim=4,
            hidden_dims=[64, 32]
        )
        
        # Test forward pass
        obs = torch.randn(2, 20)
        actions = actor_mlp(obs)
        
        print(f"✓ MLP Actor created: output shape {actions.shape}")
        
        # Test with use_gru flag
        actor_gru = ActorNetwork(
            obs_dim=20,
            action_dim=4,
            use_gru=False  # Should work like MLP
        )
        
        actions2 = actor_gru(obs)
        print(f"✓ Configurable Actor created: output shape {actions2.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Network error: {e}")
        return False

def test_agent():
    """Test MADDPG agent creation."""
    print("\nTesting agent...")
    try:
        from maddpg_agent import MADDPG_Agent
        
        agent = MADDPG_Agent(
            num_agents=2,
            obs_dim=20,
            action_dim=4,
            batch_size=32,
            buffer_capacity=1000,
            device='cpu'
        )
        
        # Test action selection
        obs = [np.random.randn(20) for _ in range(2)]
        actions = agent.select_actions(obs, add_noise=False)
        
        print(f"✓ Agent created: actions shape {actions.shape}")
        
        # Test experience storage
        agent.store_experience(
            np.random.randn(2, 20),
            actions,
            np.array([1.0, 1.0]),
            np.random.randn(2, 20),
            np.array([False, False])
        )
        
        print(f"✓ Experience stored: buffer size {len(agent.replay_buffer)}")
        
        return True
    except Exception as e:
        print(f"✗ Agent error: {e}")
        return False

def main():
    """Run all quick checks."""
    print("=" * 60)
    print("LEO-ISAC Quick Sanity Check")
    print("=" * 60)
    
    all_pass = True
    
    all_pass &= test_imports()
    all_pass &= test_environment()
    all_pass &= test_networks()
    all_pass &= test_agent()
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ All components working!")
        print("Ready for smoke test: python run_smoke_test.py")
    else:
        print("❌ Some components failed. Fix errors before proceeding.")
    print("=" * 60)

if __name__ == "__main__":
    main()