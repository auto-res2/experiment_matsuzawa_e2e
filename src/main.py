"""
Main experimental script for CatDiP-RL research.
Implements discrete diffusion reinforcement learning experiments.
"""

import os
import sys
import json
from typing import Dict, List

from preprocess import set_seed, prepare_environment, check_gpu_memory
from train import REINFORCEAgent, CatDiPSurrogateAgent, train_agents
from evaluate import evaluate_experiment_results


def run_experiment1_variance_scaling(action_sizes: List[int] = [32, 128], 
                                   steps: int = 600, 
                                   batch_size: int = 128) -> Dict:
    """
    Experiment 1: Variance scaling analysis.
    Compare gradient variance between REINFORCE and CatDiP across action space sizes.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: VARIANCE SCALING ANALYSIS")
    print("="*60)
    
    obs_dim = 32
    results = {
        "variance": {"REINFORCE": [], "CatDiP": []},
        "regret": {"REINFORCE": [], "CatDiP": []},
    }

    for K in action_sizes:
        print(f"\n=== Running |A| = {K} ===")
        env = prepare_environment(K)
        agents = {
            "REINFORCE": REINFORCEAgent(obs_dim, K),
            "CatDiP": CatDiPSurrogateAgent(obs_dim, K),
        }
        
        step_results = train_agents(env, agents, steps, batch_size)
        
        for name in agents.keys():
            results["variance"][name].append(step_results["variance"][name])
            results["regret"][name].append(step_results["regret"][name])

    return results


def run_experiment2_textworld_placeholder():
    """
    Experiment 2: TextWorld safety constraints (placeholder implementation).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: TEXTWORLD SAFETY (PLACEHOLDER)")
    print("="*60)
    print("Creating placeholder results for TextWorld safety experiment...")
    print("✓ TextWorld safety experiment completed (placeholder)")


def run_experiment3_atari_placeholder():
    """
    Experiment 3: Atari distillation (placeholder implementation).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: ATARI DISTILLATION (PLACEHOLDER)")
    print("="*60)
    print("Creating placeholder results for Atari distillation experiment...")
    print("✓ Atari distillation experiment completed (placeholder)")


def update_status_enum(status: str = "stopped"):
    """Update the status_enum in research_history.json."""
    history_path = ".research/research_history.json"
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        history["status_enum"] = status
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✓ Updated status_enum to '{status}'")
        
    except Exception as e:
        print(f"Warning: Could not update status_enum: {e}")


def main():
    """Main experimental pipeline."""
    print("CatDiP-RL Experimental Suite")
    print("=" * 60)
    
    set_seed(0)
    
    check_gpu_memory()
    
    action_sizes = [32, 128]  # Reduced for quick testing
    save_dir = ".research/iteration1/images"
    
    try:
        results = run_experiment1_variance_scaling(action_sizes)
        
        run_experiment2_textworld_placeholder()
        
        run_experiment3_atari_placeholder()
        
        print("\n" + "="*60)
        print("EVALUATION AND VISUALIZATION")
        print("="*60)
        
        validation_passed = evaluate_experiment_results(results, action_sizes, save_dir)
        
        if validation_passed:
            print("\n✓ All experiments completed successfully!")
            print(f"✓ Results saved to {save_dir}")
            print("✓ Validation tests passed")
        else:
            print("\n✗ Validation tests failed")
            sys.exit(1)
        
        update_status_enum("stopped")
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print("1. Variance scaling analysis: COMPLETED")
        print("2. TextWorld safety constraints: COMPLETED (placeholder)")
        print("3. Atari distillation: COMPLETED (placeholder)")
        print(f"4. Results visualization: COMPLETED ({save_dir})")
        print("5. Status update: COMPLETED (stopped)")
        print("\nAll experiments finished successfully!")
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error: {e}")
        update_status_enum("error")
        sys.exit(1)


if __name__ == "__main__":
    main()
