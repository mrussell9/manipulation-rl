from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def terminate_on_non_cube_contact(
    env: ManagerBasedEnv,
    arm_sensor_cfg: SceneEntityCfg,
    left_finger_sensor_cfg: SceneEntityCfg,
    right_finger_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if the robot arm touches anything, or if fingers touch non-cube objects.

    Uses three separate contact sensors to avoid the ``force_matrix_w`` filter
    mismatch that occurs when a single multi-body sensor is filtered against a
    per-environment prim (e.g. the cube).  The mismatch happens because Isaac Lab
    expects ``num_bodies × num_envs`` filter prim entries, but the cube pattern
    only resolves to ``num_envs`` prims.

    Sensor design
    -------------
    * ``arm_contact`` — tracks ``panda_link.*`` (8 arm links), **no filter**.
      Any net contact force > threshold on any arm link triggers termination
      (covers self-collisions and collisions with the table / environment).
    * ``left_finger_contact`` / ``right_finger_contact`` — each tracks a **single
      body** (``left_inner_finger`` / ``right_inner_finger``) with
      ``filter_prim_paths_expr=[cube]``.  Because there is exactly 1 body per
      sensor and 1 cube per env the filter resolves cleanly:
      ``1 body × N_envs == N_envs cube prims``.
      Non-cube contact = ``net_forces_w - force_matrix_w[cube]``  (vector
      subtraction, then magnitude).  This allows normal grasping forces on the
      cube without false terminations.

    Args:
        env: The environment instance.
        arm_sensor_cfg: Scene entity for the arm-link contact sensor.
        left_finger_sensor_cfg: Scene entity for the left inner finger sensor.
        right_finger_sensor_cfg: Scene entity for the right inner finger sensor.
        force_threshold: Contact force magnitude (N) above which contact is
            considered harmful.

    Returns:
        Boolean tensor of shape ``(num_envs,)`` — True where episode should end.
    """
    arm_sensor: ContactSensor = env.scene.sensors[arm_sensor_cfg.name]
    left_sensor: ContactSensor = env.scene.sensors[left_finger_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_finger_sensor_cfg.name]

    # ------------------------------------------------------------------
    # 1.  Arm links: ANY contact terminates.
    # net_forces_w shape: (num_envs, num_arm_bodies, 3)
    # ------------------------------------------------------------------
    arm_forces_mag = torch.linalg.norm(arm_sensor.data.net_forces_w, dim=-1)  # (N, B)
    arm_bad = torch.any(arm_forces_mag > force_threshold, dim=-1)  # (N,)

    # ------------------------------------------------------------------
    # 2.  Left inner finger: non-cube contact terminates.
    # net_forces_w   shape: (N, 1, 3)  — total force on the finger body
    # force_matrix_w shape: (N, 1, 1, 3) — cube-only portion
    # non_cube_force = total - cube  (3D vector subtraction)
    # ------------------------------------------------------------------
    left_net = left_sensor.data.net_forces_w[:, 0, :]  # (N, 3)
    left_cube = torch.nan_to_num(
        left_sensor.data.force_matrix_w[:, 0, 0, :], nan=0.0
    )  # (N, 3)
    left_bad = torch.linalg.norm(left_net - left_cube, dim=-1) > force_threshold  # (N,)

    # ------------------------------------------------------------------
    # 3.  Right inner finger: same logic.
    # ------------------------------------------------------------------
    right_net = right_sensor.data.net_forces_w[:, 0, :]  # (N, 3)
    right_cube = torch.nan_to_num(
        right_sensor.data.force_matrix_w[:, 0, 0, :], nan=0.0
    )  # (N, 3)
    right_bad = torch.linalg.norm(right_net - right_cube, dim=-1) > force_threshold  # (N,)

    # [DEBUG] Temporarily returning only arm_bad to verify if finger contacts 
    # (e.g., self-collisions) were causing the early terminations. 
    # Re-add `| left_bad | right_bad` when you want to enable it again!
    return arm_bad  # | left_bad | right_bad

def terminate_on_arm_contact(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if the robot arm touches the table.
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity for the arm-link contact sensor.
        force_threshold: Contact force magnitude (N) above which contact is
            considered harmful.
            
    Returns:
        Boolean tensor of shape ``(num_envs,)`` — True where episode should end.
    """
    arm_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # force_matrix_w shape: (N, max_contact_data_count_per_prim, num_filter_prims, 3)
    # Since filter_prims is just the table (1), we take [:, :, 0, :]
    # Then take the norm over the spatial dimension, and check if ANY link exceeds threshold
    arm_table_forces = torch.nan_to_num(
        arm_sensor.data.force_matrix_w[:, :, 0, :], nan=0.0
    )  # (N, B, 3)
    arm_forces_mag = torch.linalg.norm(arm_table_forces, dim=-1)  # (N, B)
    arm_bad = torch.any(arm_forces_mag > force_threshold, dim=-1)  # (N,)
    return arm_bad