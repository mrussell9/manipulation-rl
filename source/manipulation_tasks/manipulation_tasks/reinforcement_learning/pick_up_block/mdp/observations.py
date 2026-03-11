from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor

@generic_io_descriptor(
    units="m", axes=["X", "Y", "Z"]
)
def object_position_rel(env: ManagerBasedEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the location of the object in the environment frame."""
    obj: RigidObject = env.scene[object_cfg.name]

    # Get object position in world frame (shape: num_envs, 3)
    # For a single body object, body_pos_w has shape (num_envs, 1, 3)
    obj_pos_w = obj.data.body_pos_w[:, 0, :]
    
    # Get environment origin position in world frame
    env_origin_w = env.scene.env_origins
    
    # Return object position relative to environment origin
    return obj_pos_w - env_origin_w

@generic_io_descriptor(
    units="units", axes=["W", "X", "Y", "Z"]
)
def object_orientation_rel(env: ManagerBasedEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the orientation of the object in the environment frame."""
    obj: RigidObject = env.scene[object_cfg.name]

    # Get object orientation in world frame (shape: num_envs, 4)
    # For a single body object, body_quat_w has shape (num_envs, 1, 4)
    obj_quat = obj.data.body_quat_w[:, 0, :]
    
    # Return object orientation (quaternion is already in the correct frame)
    return obj_quat