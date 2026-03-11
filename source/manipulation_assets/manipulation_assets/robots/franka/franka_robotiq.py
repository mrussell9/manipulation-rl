import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

LOCAL_ASSETS_PATH = os.path.join(os.path.dirname(__file__), "../../../data/franka/usd")

FRANKA_ROBOTIQ_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSETS_PATH}/frana_collide.usd",
        variants={"Gripper": "Robotiq_2F_85"},
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.85, 0, 0.76),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "finger_joint": 0.0,
            ".*_inner_finger_joint": 0.0,
            ".*_inner_finger_knuckle_joint": 0.0,
            ".*_outer_.*_joint": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=5200.0,
            velocity_limit_sim=2.175,
            stiffness=1100.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=720.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=80.0,
        ),
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
            effort_limit_sim=1650,
            velocity_limit_sim=10.0,
            stiffness=8,
            damping=0.5,
        ),
        # enable the gripper to grasp in a parallel manner
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=50,
            velocity_limit_sim=10.0,
            stiffness=0.2,
            damping=0.001,
        ),
        # set PD to zero for passive joints in close-loop gripper
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot with Robotiq_2f_85 gripper."""