"""Curriculum learning terms for progressive robot pose randomization.

This module provides functions for gradually increasing the difficulty of robot
initialization poses over training, implementing curriculum learning for the
pick-up-block task.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform, matrix_from_quat, quat_from_matrix
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Global difficulty parameter - updated by curriculum manager
_pose_curriculum_difficulty = 0.0


def set_pose_curriculum_difficulty(difficulty: float) -> None:
    """Set the global pose curriculum difficulty.
    
    Args:
        difficulty: Value from 0.0 (easy) to 1.0 (hard).
    """
    global _pose_curriculum_difficulty
    _pose_curriculum_difficulty = difficulty


def get_pose_curriculum_difficulty() -> float:
    """Get the current pose curriculum difficulty."""
    return _pose_curriculum_difficulty


def reset_robot_pose_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot base pose with curriculum-based difficulty.
    
    The robot starts in fixed configuration and progressively can spawn in more
    varied positions and orientations as training progresses.
    
    Args:
        env: The environment object.
        env_ids: The environment indices to reset.
        asset_cfg: The scene entity configuration for the robot.
    
    Note:
        The difficulty is controlled by the global _pose_curriculum_difficulty,
        which is updated by the curriculum manager during training.
    """
    # Get current difficulty from global state
    difficulty = get_pose_curriculum_difficulty()
    
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Default/home position
    default_pos = torch.tensor([0.2, 0.0, 0.24], device=env.device, dtype=torch.float32)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32)
    
    # Define pose variation ranges that scale with curriculum difficulty
    # Phase 1 (0.0-0.3): Position variations only, small
    # Phase 2 (0.3-0.6): Position variations increase, small rotation
    # Phase 3 (0.6-1.0): Full pose variations
    
    num_envs = len(env_ids)
    batch_size = asset.data.root_pos_w.shape[0]
    
    # Sample random poses
    sampled_pos = default_pos.unsqueeze(0).repeat(batch_size, 1).clone()
    sampled_quat = default_quat.unsqueeze(0).repeat(batch_size, 1).clone()
    
    if difficulty > 0.0:
        # Position variations (enabled at all curriculum levels)
        # X: left-right variation (workspace dependent)
        # Y: forward-backward variation  
        # Z: height variation (minimal, mostly at table surface)
        
        if difficulty < 0.3:
            # Early curriculum: small position variations
            x_range = 0.15 + difficulty / 0.3 * 0.15  # 0.15 to 0.30m
            y_range = 0.10 + difficulty / 0.3 * 0.10  # 0.10 to 0.20m
            z_range = 0.02  # Small height variation
        elif difficulty < 0.6:
            # Mid curriculum: moderate position + minor rotations
            x_range = 0.30 + (difficulty - 0.3) / 0.3 * 0.20  # 0.30 to 0.50m
            y_range = 0.20 + (difficulty - 0.3) / 0.3 * 0.25  # 0.20 to 0.45m
            z_range = 0.04  # Slightly more height variation
        else:
            # Late curriculum: full variations
            x_range = 0.50  # Up to ±0.50m left-right
            y_range = 0.45  # Up to ±0.45m forward-backward
            z_range = 0.06  # Up to ±0.06m height variation
        
        # Sample position offsets from default
        pos_offset = torch.zeros((batch_size, 3), device=env.device, dtype=torch.float32)
        pos_offset[:, 0] = sample_uniform(-x_range, x_range, (batch_size,), env.device)
        pos_offset[:, 1] = sample_uniform(-y_range, y_range, (batch_size,), env.device)
        pos_offset[:, 2] = sample_uniform(-z_range, z_range, (batch_size,), env.device)
        
        sampled_pos = sampled_pos + pos_offset
        
        # Orientation variations (scaled with curriculum)
        if difficulty >= 0.3:
            if difficulty < 0.6:
                # Small roll-pitch variations only (yaw locked to table alignment)
                roll_range = (difficulty - 0.3) / 0.3 * 0.2  # 0 to 0.2 rad (≈11°)
                pitch_range = (difficulty - 0.3) / 0.3 * 0.2  # 0 to 0.2 rad
                yaw_range = 0.0
            else:
                # Full rotation variations
                roll_range = 0.3  # ±0.3 rad (≈17°)
                pitch_range = 0.3  # ±0.3 rad
                yaw_range = math.pi  # ±180° for full rotation around vertical axis
            
            # Sample rotation angles
            roll = sample_uniform(-roll_range, roll_range, (batch_size,), env.device)
            pitch = sample_uniform(-pitch_range, pitch_range, (batch_size,), env.device)
            yaw = sample_uniform(-yaw_range, yaw_range, (batch_size,), env.device)
            
            # Convert Euler to quaternion (XYZ order)
            # Using simplified rotation matrix approach
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)
            cp = torch.cos(pitch * 0.5)
            sp = torch.sin(pitch * 0.5)
            cr = torch.cos(roll * 0.5)
            sr = torch.sin(roll * 0.5)
            
            sampled_quat = torch.stack([
                cy * cp * cr + sy * sp * sr,  # w
                cy * cp * sr - sy * sp * cr,  # x
                cy * sp * cr + sy * cp * sr,  # y
                sy * cp * cr - cy * sp * sr,  # z
            ], dim=-1)
    
    # Convert env_ids to long tensor for indexing
    env_ids_long = env_ids.long()
    
    # Combine position and quaternion into root pose (x, y, z, qw, qx, qy, qz)
    root_pose = torch.cat([sampled_pos, sampled_quat], dim=-1)
    
    # Set the root state for specified environments directly
    asset.data.root_link_pose_w[env_ids_long] = root_pose[env_ids_long].clone()
    
    # Reset velocities to zero (6D: linear velocity + angular velocity)
    root_vel = torch.zeros((batch_size, 6), device=env.device, dtype=torch.float32)
    asset.data.root_link_vel_w[env_ids_long] = root_vel[env_ids_long].clone()


def reset_object_pose_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
    """Reset the object base pose with curriculum-based difficulty.
    
    The object starts at a default location near the robot and progressively
    can spawn in more varied positions as training progresses.
    
    Args:
        env: The environment object.
        env_ids: The environment indices to reset.
        asset_cfg: The scene entity configuration for the object.
    
    Note:
        The difficulty is controlled by the global _pose_curriculum_difficulty,
        which is updated by the curriculum manager during training.
    """
    # Get current difficulty from global state
    difficulty = get_pose_curriculum_difficulty()
    
    # Extract the asset
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Default object position (on table surface in front of robot)
    default_pos = torch.tensor([0.35, 0.0, 1.1], device=env.device, dtype=torch.float32)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32)
    
    num_envs = len(env_ids)
    batch_size = asset.data.root_pos_w.shape[0]
    
    sampled_pos = default_pos.unsqueeze(0).repeat(batch_size, 1).clone()
    sampled_quat = default_quat.unsqueeze(0).repeat(batch_size, 1).clone()
    
    if difficulty > 0.0:
        # Gradual expansion of object placement region
        if difficulty < 0.5:
            # Early: small variations around default position
            x_range = difficulty / 0.5 * 0.15  # 0 to 0.15m
            y_range = difficulty / 0.5 * 0.15  # 0 to 0.15m
            z_range = 0.02  # Very small height variation
        else:
            # Late: larger placement region
            x_range = 0.15 + (difficulty - 0.5) / 0.5 * 0.20  # 0.15 to 0.35m
            y_range = 0.15 + (difficulty - 0.5) / 0.5 * 0.30  # 0.15 to 0.45m
            z_range = 0.04  # Small height variation
        
        # Sample position offsets
        pos_offset = torch.zeros((batch_size, 3), device=env.device, dtype=torch.float32)
        pos_offset[:, 0] = sample_uniform(-x_range, x_range, (batch_size,), env.device)
        pos_offset[:, 1] = sample_uniform(-y_range, y_range, (batch_size,), env.device)
        pos_offset[:, 2] = sample_uniform(-z_range, z_range, (batch_size,), env.device)
        
        sampled_pos = sampled_pos + pos_offset
        
        # Rotation variations for the object (late curriculum only)
        if difficulty > 0.6:
            # Random orientation for the object
            rot_range = (difficulty - 0.6) / 0.4 * math.pi  # 0 to π radians around Z
            yaw = sample_uniform(-rot_range, rot_range, (batch_size,), env.device)
            
            # Simple Z-axis rotation quaternion
            c = torch.cos(yaw * 0.5)
            s = torch.sin(yaw * 0.5)
            sampled_quat = torch.stack([c, torch.zeros_like(c), torch.zeros_like(s), s], dim=-1)
    
    # Convert env_ids to long tensor for indexing
    env_ids_long = env_ids.long()
    
    # Combine position and quaternion into root pose (x, y, z, qw, qx, qy, qz)
    root_pose = torch.cat([sampled_pos, sampled_quat], dim=-1)
    
    # Set the root state for specified environments directly
    asset.data.root_link_pose_w[env_ids_long] = root_pose[env_ids_long].clone()
    
    # Reset velocities to zero (6D: linear velocity + angular velocity)
    root_vel = torch.zeros((batch_size, 6), device=env.device, dtype=torch.float32)
    asset.data.root_link_vel_w[env_ids_long] = root_vel[env_ids_long].clone()


def update_pose_difficulty(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    num_steps: int = 50000,
    initial_value: float = 0.0,
    final_value: float = 1.0,
) -> None:
    """Update the pose curriculum difficulty based on training progress.
    
    This function is called by the curriculum manager to gradually
    increase the difficulty of robot/object initialization poses.
    
    Args:
        env: The environment object.
        env_ids: Environment indices to update (provided by curriculum manager).
        num_steps: Number of training steps to reach final difficulty.
        initial_value: Starting difficulty (0.0).
        final_value: Target difficulty (1.0).
    """
    # Get the current training step count
    if not hasattr(env, "_pose_curriculum_step"):
        env._pose_curriculum_step = 0
    else:
        env._pose_curriculum_step += 1
    
    # Calculate current difficulty as linear interpolation
    progress = min(env._pose_curriculum_step / num_steps, 1.0)
    difficulty = initial_value + (final_value - initial_value) * progress
    
    # Update the global difficulty for event functions
    set_pose_curriculum_difficulty(difficulty)
