#!/usr/bin/env python3
"""
Example script demonstrating the curriculum learning system for robot poses.

This script shows how to:
1. Create the environment with curriculum-based pose randomization
2. Control curriculum difficulty manually
3. Monitor curriculum progression
"""

import sys
from pathlib import Path

# Add workspace to path
WORKSPACE_DIR = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE_DIR / "source"))

import torch
import gymnasium as gym

# Import curriculum control functions
from manipulation_tasks.reinforcement_learning.pick_up_block.mdp import (
    set_pose_curriculum_difficulty,
    get_pose_curriculum_difficulty,
)


def example_1_basic_curriculum():
    """Example 1: Create environment with automatic curriculum."""
    print("\n" + "="*70)
    print("Example 1: Basic Curriculum Learning")
    print("="*70)
    
    # Create environment (curriculum enabled by default)
    print("\nCreating environment with curriculum learning enabled...")
    
    # Note: This assumes you have the proper isaaclab setup
    # In practice, you'd use:
    # env = gym.make("BlockPickup-Franka-v0")
    
    print("✓ Environment created")
    print("✓ Robot poses will automatically progress from easy to hard during training")


def example_2_manual_difficulty_control():
    """Example 2: Manually control curriculum difficulty."""
    print("\n" + "="*70)
    print("Example 2: Manual Difficulty Control")
    print("="*70)
    
    # Easy phase: positions only
    print("\nSetting difficulty to EASY (0.2)...")
    set_pose_curriculum_difficulty(0.2)
    print(f"Current difficulty: {get_pose_curriculum_difficulty()}")
    print("→ Robot spawns with position variations only (small ranges)")
    
    # Medium phase: positions + rotations
    print("\nSetting difficulty to MEDIUM (0.5)...")
    set_pose_curriculum_difficulty(0.5)
    print(f"Current difficulty: {get_pose_curriculum_difficulty()}")
    print("→ Robot spawns with position and minor rotation variations")
    
    # Hard phase: full variations
    print("\nSetting difficulty to HARD (1.0)...")
    set_pose_curriculum_difficulty(1.0)
    print(f"Current difficulty: {get_pose_curriculum_difficulty()}")
    print("→ Robot spawns with full pose variations (position + rotation)")


def example_3_monitoring_progression():
    """Example 3: Monitor curriculum progression during training."""
    print("\n" + "="*70)
    print("Example 3: Monitoring Curriculum Progress")
    print("="*70)
    
    print("\nDemonstrating curriculum progression over 'training steps':")
    print("-" * 70)
    
    # Simulate training progression
    num_steps = 100000
    check_points = [0, 0.25, 0.5, 0.75, 1.0]
    
    for progress in check_points:
        step = int(progress * num_steps)
        # In real training, this would be automatically updated
        simulated_difficulty = progress  # Linear progression
        set_pose_curriculum_difficulty(simulated_difficulty)
        
        current_diff = get_pose_curriculum_difficulty()
        
        phase = "Phase 1: Position variations only"
        if current_diff >= 0.3 and current_diff < 0.6:
            phase = "Phase 2: Positions + minor rotations"
        elif current_diff >= 0.6:
            phase = "Phase 3: Full pose variations"
        
        print(f"Step {step:6d} | Difficulty: {current_diff:.2f} | {phase}")
    
    print("-" * 70)


def example_4_curriculum_phases_explained():
    """Example 4: Detailed explanation of curriculum phases."""
    print("\n" + "="*70)
    print("Example 4: Understanding Curriculum Phases")
    print("="*70)
    
    phases = [
        {
            "name": "Phase 1: Foundation (0.0-0.3)",
            "difficulty": 0.15,
            "description": "Robot learns basic pick-up with minimal pose variation",
            "changes": [
                "X position variation: ±0.15-0.30m (lateral)",
                "Y position variation: ±0.10-0.20m (forward-backward)",
                "Z position variation: ±0.02m (height)",
                "Rotation: NONE (fixed orientation)",
            ],
            "agent_focus": "Learning how to reach and grasp in fixed orientation",
        },
        {
            "name": "Phase 2: Adaptation (0.3-0.6)",
            "difficulty": 0.45,
            "description": "Robot learns to adapt to different approach angles",
            "changes": [
                "X position variation: ±0.30-0.50m (increased lateral)",
                "Y position variation: ±0.20-0.45m (increased forward-backward)",
                "Z position variation: ±0.04m (slightly more height)",
                "Roll & Pitch rotation: ±0.0-0.2 rad (~0-11°)",
                "Yaw rotation: NONE (still table-aligned)",
            ],
            "agent_focus": "Adapting to different starting orientations while table-aligned",
        },
        {
            "name": "Phase 3: Mastery (0.6-1.0)",
            "difficulty": 0.8,
            "description": "Robot learns full pose variation for complete robustness",
            "changes": [
                "X position variation: ±0.50m (maximum lateral)",
                "Y position variation: ±0.45m (maximum forward-backward)",
                "Z position variation: ±0.06m (maximum height)",
                "Roll & Pitch rotation: ±0.3 rad (~±17°)",
                "Yaw rotation: ±π radians (full 360° around vertical)",
            ],
            "agent_focus": "Handling arbitrary robot poses and orientations",
        },
    ]
    
    for phase in phases:
        set_pose_curriculum_difficulty(phase["difficulty"])
        print(f"\n{phase['name']}")
        print(f"   {phase['description']}")
        print(f"   Configuration:")
        for change in phase["changes"]:
            print(f"      • {change}")
        print(f"   Agent learns: {phase['agent_focus']}")


def example_5_combining_with_other_curriculum():
    """Example 5: Using pose curriculum alongside other curriculum terms."""
    print("\n" + "="*70)
    print("Example 5: Combining Multiple Curriculum Terms")
    print("="*70)
    
    print("\nThe pose curriculum works alongside other curriculum terms:")
    print("-" * 70)
    print("1. robot_pose_difficulty")
    print("   └─ Increases robot/object pose variation over time")
    print()
    print("2. action_rate")
    print("   └─ Gradually increases penalty for excessive actions")
    print()
    print("3. joint_vel")
    print("   └─ Gradually increases penalty for high joint velocities")
    print("-" * 70)
    
    print("\nProgression example:")
    for step_pct in range(0, 101, 25):
        progress = step_pct / 100.0
        set_pose_curriculum_difficulty(progress)
        
        # Simulate other curriculum terms
        action_rate_weight = -1e-4 + progress * (-1e-1 - (-1e-4))
        
        print(f"\nTraining: {step_pct:3d}%")
        print(f"  Pose difficulty:        {get_pose_curriculum_difficulty():.2f}")
        print(f"  Action rate penalty:    {action_rate_weight:.2e}")


def example_6_disabling_curriculum():
    """Example 6: Disabling or freezing curriculum at specific level."""
    print("\n" + "="*70)
    print("Example 6: Controlling Curriculum Behavior")
    print("="*70)
    
    print("\nOption 1: Always use default poses (no randomization)")
    set_pose_curriculum_difficulty(0.0)
    print(f"✓ Set difficulty to 0.0")
    print(f"  Current: {get_pose_curriculum_difficulty()}")
    print("  Result: Robot always spawns at exact same pose")
    
    print("\nOption 2: Freeze at medium difficulty for balanced training")
    set_pose_curriculum_difficulty(0.5)
    print(f"✓ Set difficulty to 0.5")
    print(f"  Current: {get_pose_curriculum_difficulty()}")
    print("  Result: Moderate pose variation, no extreme configurations")
    
    print("\nOption 3: Maximum difficulty (full randomization)")
    set_pose_curriculum_difficulty(1.0)
    print(f"✓ Set difficulty to 1.0")
    print(f"  Current: {get_pose_curriculum_difficulty()}")
    print("  Result: Maximum pose variation for robust training")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("CURRICULUM LEARNING FOR ROBOT POSES - EXAMPLES")
    print("="*70)
    
    example_1_basic_curriculum()
    example_2_manual_difficulty_control()
    example_3_monitoring_progression()
    example_4_curriculum_phases_explained()
    example_5_combining_with_other_curriculum()
    example_6_disabling_curriculum()
    
    print("\n" + "="*70)
    print("For more details, see: CURRICULUM_LEARNING.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
