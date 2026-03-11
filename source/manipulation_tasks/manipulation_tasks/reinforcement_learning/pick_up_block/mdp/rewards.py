from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reward_ee_distance_to_object(
    env: ManagerBasedEnv, 
    std: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for end effector being close to the object.
    
    The reward is computed as exp(-distance^2 / std^2).
    
    Args:
        env: The environment.
        std: Standard deviation for the Gaussian reward.
        robot_cfg: Configuration for the robot asset (should include end effector body).
        object_cfg: Configuration for the object asset.
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get end effector position
    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids].squeeze(1)
    
    # Get object position
    obj_pos = obj.data.body_pos_w.squeeze(1)
    
    distance = torch.norm(obj_pos - ee_pos, dim=-1)
    
    # Heavy-tailed reward to avoid vanishing gradients
    reward = 1.0 / (1.0 + (distance / std) ** 2)
    return reward


def reward_ee_orientation_alignment(
    env: ManagerBasedEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for end effector orienting downwards towards the object.
    
    The reward is computed based on how well the gripper's local Z-axis 
    aligns with the world downward vector (0, 0, -1).
    
    Args:
        env: The environment.
        std: Standard deviation for the Gaussian reward.
        robot_cfg: Configuration for the robot asset (should include end effector body).
        object_cfg: Configuration for the object asset. (Not used in downward check but kept for API match).
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    import isaaclab.utils.math as math_utils
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get end effector orientation (quaternion: w, x, y, z)
    ee_quat = robot.data.body_quat_w[:, robot_cfg.body_ids].squeeze(1)
    
    # The RobotIQ gripper points downwards along its local Z-axis.
    # We rotate the local Z vector [0, 0, 1] into the world frame:
    local_z = torch.zeros((ee_quat.shape[0], 3), device=ee_quat.device)
    local_z[:, 2] = 1.0
    
    world_z = math_utils.quat_apply(ee_quat, local_z)
    
    # Desired pointing vector is straight down (0, 0, -1)
    target_z = torch.zeros_like(world_z)
    target_z[:, 2] = -1.0
    
    # Compute dot product
    z_dot = torch.sum(world_z * target_z, dim=-1)
    
    # Map dot product [-1, 1] to an angular error distance [0, 2]
    angular_distance = 1.0 - z_dot
    
    # Gaussian reward
    reward = torch.exp(-angular_distance ** 2 / std ** 2)

    return reward


def reward_object_lift(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
    minimal_height: float,
) -> torch.Tensor:
    """Reward for lifting the object above a specified height.
    
    Returns a binary reward (0 or 1) based on whether the object is lifted
    above the minimal height threshold.
    
    Args:
        env: The environment.
        object_cfg: Configuration for the object asset.
        minimal_height: Minimum height (in meters) above the environment origin for the reward.
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get object position in world frame
    obj_pos_w = obj.data.body_pos_w.squeeze(1)
    
    # Get environment origin position
    env_origin_w = env.scene.env_origins
    
    # Compute object height relative to environment origin
    obj_height = (obj_pos_w - env_origin_w)[:, 2]
    
    # Binary reward: 1 if object is above minimal height, 0 otherwise
    reward = (obj_height > minimal_height).float()
    
    return reward


def reward_object_lift_smooth(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
) -> torch.Tensor:
    """Smooth reward for lifting the object to a target height.
    
    Uses a Gaussian reward that increases as the object approaches the target height.
    
    Args:
        env: The environment.
        object_cfg: Configuration for the object asset.
        target_height: Target height (in meters) above the environment origin.
        std: Standard deviation for the Gaussian reward.
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get object position in world frame
    obj_pos_w = obj.data.body_pos_w.squeeze(1)
    
    # Get environment origin position
    env_origin_w = env.scene.env_origins
    
    # Compute object height relative to environment origin
    obj_height = (obj_pos_w - env_origin_w)[:, 2]
    
    # Compute distance from target height
    height_error = torch.abs(obj_height - target_height)
    
    # Gaussian reward
    reward = torch.exp(-height_error ** 2 / std ** 2)
    
    return reward


def reward_finger_center_distance_to_object(
    env: ManagerBasedEnv,
    std: float,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for the midpoint between fingers being close to the object."""
    left_finger: RigidObject = env.scene[left_finger_cfg.name]
    right_finger: RigidObject = env.scene[right_finger_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get center between both inner fingers
    left_pos = left_finger.data.body_pos_w[:, left_finger_cfg.body_ids].squeeze(1)
    right_pos = right_finger.data.body_pos_w[:, right_finger_cfg.body_ids].squeeze(1)
    center_pos = (left_pos + right_pos) / 2.0
    
    obj_pos = obj.data.body_pos_w.squeeze(1)
    distance = torch.norm(obj_pos - center_pos, dim=-1)
    
    # Heavy-tailed reward to avoid vanishing gradients
    reward = 1.0 / (1.0 + (distance / std) ** 2)
    return reward


def reward_deep_grasping(
    env: ManagerBasedEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    left_finger_sensor_cfg: SceneEntityCfg,
    right_finger_sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float = 0.05,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward for grasping deeply (closing fingers ONLY when near the cube)."""
    # 1. Distance check
    left_finger: RigidObject = env.scene[left_finger_cfg.name]
    right_finger: RigidObject = env.scene[right_finger_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    left_pos = left_finger.data.body_pos_w[:, left_finger_cfg.body_ids].squeeze(1)
    right_pos = right_finger.data.body_pos_w[:, right_finger_cfg.body_ids].squeeze(1)
    center_pos = (left_pos + right_pos) / 2.0
    obj_pos = obj.data.body_pos_w.squeeze(1)
    
    distance = torch.norm(obj_pos - center_pos, dim=-1)
    is_close = distance < distance_threshold
    
    # 2. Bimanual contact check
    left_sensor: ContactSensor = env.scene.sensors[left_finger_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_finger_sensor_cfg.name]

    left_cube_forces = torch.nan_to_num(left_sensor.data.force_matrix_w[:, 0, 0, :], nan=0.0)
    right_cube_forces = torch.nan_to_num(right_sensor.data.force_matrix_w[:, 0, 0, :], nan=0.0)

    left_contact_mag = torch.linalg.norm(left_cube_forces, dim=-1)
    right_contact_mag = torch.linalg.norm(right_cube_forces, dim=-1)

    is_left_grasping = left_contact_mag > force_threshold
    is_right_grasping = right_contact_mag > force_threshold
    
    # Deep grasp is only when both grasping AND being at the center of the block
    is_grasping = is_left_grasping & is_right_grasping & is_close

    return is_grasping.float()


def reward_grasped_lift_smooth(
    env: ManagerBasedEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    left_finger_sensor_cfg: SceneEntityCfg,
    right_finger_sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    distance_threshold: float = 0.05,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Smooth lift reward that is strictly conditional on deep_grasping being active."""
    # Rely on the strict grasping definition to ensure we only reward actual lifting
    is_grasping = reward_deep_grasping(
        env=env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        left_finger_sensor_cfg=left_finger_sensor_cfg,
        right_finger_sensor_cfg=right_finger_sensor_cfg,
        object_cfg=object_cfg,
        distance_threshold=distance_threshold,
        force_threshold=force_threshold,
    )
    
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.body_pos_w.squeeze(1)
    env_origin_w = env.scene.env_origins
    
    # Compute object height relative to environment origin
    obj_height = (obj_pos_w - env_origin_w)[:, 2]
    height_error = torch.abs(obj_height - target_height)
    
    # Gaussian lift reward
    lift_reward = torch.exp(-height_error ** 2 / std ** 2)
    
    # Only supply gradient if actively grasped natively
    return lift_reward * is_grasping
