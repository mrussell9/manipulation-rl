# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kinova Robotics arms.

The following configuration parameters are available:

* :obj:`COBOT_CFG`: The Kinova JACO2 (7-Dof) arm with a 3-finger gripper.

Reference: https://github.com/ut-amrl/ros2_kortex/tree/cobot-sim
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))


"""Configuration for the AMRL Cobot with Kinova Gen3 arm and Robotiq 85 gripper."""
COBOT_FIXED_BASE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        make_instanceable=True,
        asset_path=f"{EXT_DIR}/data/kinova/cobot/urdf/cobot.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        rot=(-0.7071, 0.0, 0.0, 0.7071),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
            "joint_7": 0.0,
            "robotiq_85_.*": 0.0,
            # ".*_left_finger_joint": 0.0,  # all finger joints start at 0.0
            # ".*_inner_finger_joint": 0.0,
            # ".*_inner_finger_knuckle_joint": 0.0,
            # ".*_outer_.*_joint": 0.0,
        },
    ),
    actuators={
        # Arm values taken from IsaacLab Kinova Gen3 config
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            effort_limit_sim={
                "joint_[1-2]": 80.0,
                "joint_[3-4]": 40.0,
                "joint_[5-7]": 20.0,
            },
            stiffness={
                "joint_[1-4]": 40.0,
                "joint_[5-7]": 15.0,
            },
            damping={
                "joint_[1-4]": 1.0,
                "joint_[5-7]": 0.5,
            },
        ),
        # Hand values taken from IsaacLab Robotiq 85 gripper config
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=[".*_left_knuckle_joint", ".*_right_knuckle_joint"],  # "right_outer_knuckle_joint" is its mimic joint
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=11.25,
            damping=0.1,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger_tip_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.2,
            damping=0.001,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
