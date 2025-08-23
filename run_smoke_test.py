#!/usr/bin/env python3
"""
Smoke Test Runner for LEO-ISAC MARL System
==========================================

This script executes a quick end-to-end test to verify system functionality.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json



def run_training():
    """Execute the training script with test configuration."""
    print("=" * 60)
    print("Starting Smoke Test Training")
    print("=" * 60)
    
    # Command to run training
    cmd = [
        sys.executable,  # Use current Python interpreter
        "train.py",
        "--config", "test_config.yml",
        "--seed", "123",
        "--no_gru",  # Ensure MLP baseline
        "--verbose"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run training and capture output
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds")
    return True

def analyze_results():
    """Analyze training results and create plots."""
    print("\n" + "=" * 60)
    print("Analyzing Results")
    print("=" * 60)
    
    # Find the most recent test run directory
    test_dir = Path("./test_run")
    
    # Check if directory exists
    if not test_dir.exists():
        print("Error: Test run directory not found!")
        return False
    
    # Find the most recent experiment subdirectory
    exp_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("leo_isac_exp_")]
    if not exp_dirs:
        print("Error: No experiment directories found!")
        return False
    
    latest_exp = max(exp_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Analyzing experiment: {latest_exp.name}")
    
    # Load training CSV
    train_csv_path = latest_exp / "logs" / "training.csv"
    eval_csv_path = latest_exp / "logs" / "evaluation.csv"
    
    if not train_csv_path.exists():
        print(f"Error: Training CSV not found at {train_csv_path}")
        return False
    
    # Read training data
    try:
        train_df = pd.read_csv(train_csv_path)
    except Exception as e:
        print(f"Error reading training CSV: {e}")
        return False
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('LEO-ISAC MARL Smoke Test Results', fontsize=16)
    
    # Plot 1: Total Reward
    if 'total_reward' in train_df.columns:
        ax = axes[0, 0]
        # Use rolling average for smoother curve
        window = min(10, len(train_df) // 10)
        reward_smooth = train_df['total_reward'].rolling(window=window, min_periods=1).mean()
        ax.plot(train_df['episode'], train_df['total_reward'], alpha=0.3, label='Raw')
        ax.plot(train_df['episode'], reward_smooth, label='Smoothed', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Learning Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Check for learning trend
        first_quarter = reward_smooth[:len(reward_smooth)//4].mean()
        last_quarter = reward_smooth[-len(reward_smooth)//4:].mean()
        improvement = (last_quarter - first_quarter) / abs(first_quarter) * 100 if first_quarter != 0 else 0
        
        ax.text(0.05, 0.95, f'Improvement: {improvement:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Communication vs Sensing Rewards
    if 'comm_reward' in train_df.columns and 'sens_reward' in train_df.columns:
        ax = axes[0, 1]
        ax.plot(train_df['episode'], train_df['comm_reward'], label='Communication', alpha=0.7)
        ax.plot(train_df['episode'], train_df['sens_reward'], label='Sensing', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward Components')
        ax.set_title('ISAC Reward Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Throughput
    if 'throughput_gbps' in train_df.columns:
        ax = axes[0, 2]
        throughput_smooth = train_df['throughput_gbps'].rolling(window=window, min_periods=1).mean()
        ax.plot(train_df['episode'], train_df['throughput_gbps'], alpha=0.3)
        ax.plot(train_df['episode'], throughput_smooth, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Throughput (Gbps)')
        ax.set_title('Communication Performance')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: GDOP
    if 'gdop_m' in train_df.columns:
        ax = axes[1, 0]
        # Filter out infinite values
        gdop_finite = train_df['gdop_m'].replace([np.inf, -np.inf], np.nan)
        if not gdop_finite.isna().all():
            gdop_smooth = gdop_finite.rolling(window=window, min_periods=1).mean()
            ax.plot(train_df['episode'], gdop_finite, alpha=0.3)
            ax.plot(train_df['episode'], gdop_smooth, linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('GDOP (m)')
            ax.set_title('Sensing Performance')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    # Plot 5: Actor/Critic Losses
    if 'loss_actor' in train_df.columns and 'loss_critic' in train_df.columns:
        ax = axes[1, 1]
        ax.plot(train_df['episode'], train_df['loss_actor'], label='Actor', alpha=0.7)
        ax.plot(train_df['episode'], train_df['loss_critic'], label='Critic', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Network Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Evaluation Results (if available)
    if eval_csv_path.exists():
        try:
            eval_df = pd.read_csv(eval_csv_path)
            ax = axes[1, 2]
            
            # Plot comparison between algorithms
            algorithms = eval_df['algorithm'].unique()
            for algo in algorithms:
                algo_data = eval_df[eval_df['algorithm'] == algo]
                ax.plot(algo_data['episode'], algo_data['total_reward'], 
                       label=algo, marker='o', markersize=4)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Evaluation Reward')
            ax.set_title('Algorithm Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except:
            axes[1, 2].text(0.5, 0.5, 'No evaluation data', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure
    output_path = latest_exp / "smoke_test_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results plot saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    if 'total_reward' in train_df.columns:
        print(f"Initial Reward (first 10%): {train_df['total_reward'][:len(train_df)//10].mean():.3f}")
        print(f"Final Reward (last 10%): {train_df['total_reward'][-len(train_df)//10:].mean():.3f}")
        print(f"Peak Reward: {train_df['total_reward'].max():.3f}")
        print(f"Reward Std Dev: {train_df['total_reward'].std():.3f}")
    
    if 'throughput_gbps' in train_df.columns:
        print(f"Average Throughput: {train_df['throughput_gbps'].mean():.2f} Gbps")
    
    # Success criteria check
    print("\n" + "=" * 60)
    print("Success Criteria Check")
    print("=" * 60)
    
    success = True
    
    # Check 1: Training completed without errors
    print("✓ Training completed without errors")
    
    # Check 2: Positive learning trend
    if 'total_reward' in train_df.columns:
        if improvement > 10:  # At least 10% improvement
            print(f"✓ Positive learning trend detected ({improvement:.1f}% improvement)")
        else:
            print(f"⚠ Weak learning trend ({improvement:.1f}% improvement)")
            success = False if improvement < 0 else success
    
    # Check 3: Convergence stability
    if 'total_reward' in train_df.columns:
        final_std = train_df['total_reward'][-20:].std()
        initial_std = train_df['total_reward'][:20].std()
        if final_std < initial_std:
            print(f"✓ Variance reduction observed ({initial_std:.3f} → {final_std:.3f})")
        else:
            print(f"⚠ No variance reduction ({initial_std:.3f} → {final_std:.3f})")
    
    return success

def main():
    """Main smoke test execution with fixed config handling."""
    print("LEO-ISAC MARL System - Smoke Test")
    print("=" * 60)
    
    # Check if test config exists - 修复的部分
    if not Path("test_config.yml").exists():
        print("\n" + "=" * 60)
        print("ERROR: test_config.yml not found!")
        print("=" * 60)
        print("\nPlease ensure test_config.yml is present in the root directory.")
        print("This file defines the simplified configuration for smoke testing.")
        print("\nExpected location: ./test_config.yml")
        return 1
    
    print("✓ Found test_config.yml")
    
    # Run training
    if not run_training():
        print("\n❌ Training failed! Check error messages above.")
        return 1
    
    # Analyze results
    if not analyze_results():
        print("\n⚠ Analysis completed with warnings")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Smoke Test PASSED!")
    print("=" * 60)
    print("\nThe system is functioning correctly and showing learning behavior.")
    print("You can now proceed with full-scale experiments.")
    
    return 0
    
if __name__ == "__main__":
    exit(main())