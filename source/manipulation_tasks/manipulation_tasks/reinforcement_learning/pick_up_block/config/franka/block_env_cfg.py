from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


import isaaclab.envs.mdp as mdp
from manipulation_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG
import  manipulation_tasks.reinforcement_learning.pick_up_block.mdp as man_mdp


FLAT_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    
    # robot
    robot: ArticulationCfg = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # props
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 1.2], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # sensors
    # Arm links: any contact = bad (self-collision or env collision on arm).
    # Exclude panda_link0 (robot base – physically mounted on/near the table,
    # so it will always read a large contact force and would trigger immediately)
    # and panda_link1 (close to the table surface at the reset pose).
    # Links 2-7 are the "free" part of the arm that should never touch anything.
    arm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot/panda_link[2-7]",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/PackingTable"],
        history_length=3,
        track_air_time=False,
        debug_vis=False,
    )

    # Inner finger sensors: one body each, filtered to the cube.
    # 1 body × N_envs = N_envs  →  exactly matches the N_envs cube prims.
    # force_matrix_w[:,0,0,:] gives the cube-only force on that finger;
    # net_forces_w[:,0,:] gives the total force, so:
    #   non_cube_force = net_forces_w - force_matrix_w  (vector subtraction)
    left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot/left_inner_finger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=3,
        track_air_time=False,
        debug_vis=False,
    )
    
    right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot/right_inner_finger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=3,
        track_air_time=False,
        debug_vis=False,
    )
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=300.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
    gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.04},
            close_command_expr={"finger_joint": 1.0},
        )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_pos_rel = ObsTerm(
            func=man_mdp.object_position_rel, 
            params={
                "object_cfg": SceneEntityCfg("cube")
                }
            )
        cube_ori_rel = ObsTerm(
            func=man_mdp.object_orientation_rel, 
            params={
                "object_cfg": SceneEntityCfg("cube")
                }
            )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cube_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube", body_names=".*"),
            "static_friction_range": (1.0, 1.25),
            "dynamic_friction_range": (1.25, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )


    reset_robot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0, 0.4],
                "y": [-0.3, 0.3],
                "z": [0.18, 0.30],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.6, 0.6),
            "velocity_range": (-0.2, 0.2),
        },
    )

    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0, 0.0],
                "y": [-0.0, 0.0],
                "z": [0.0, 0.0],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""    
    reaching_cube = RewTerm(
        func=man_mdp.reward_finger_center_distance_to_object,
        weight=2.0,
        params={
            "std": 0.1,
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            "object_cfg": SceneEntityCfg("cube")
        }
    )

    aligning_cube = RewTerm(
        func=man_mdp.reward_ee_orientation_alignment,
        weight=0.5,
        params={
            "std": 0.5,
            "robot_cfg": SceneEntityCfg("robot", body_names=["robotiq_base_link"]),
            "object_cfg": SceneEntityCfg("cube")
        }
    )

    grasping_cube = RewTerm(
        func=man_mdp.reward_deep_grasping,
        weight=4.0,
        params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            "left_finger_sensor_cfg": SceneEntityCfg("left_finger_contact"),
            "right_finger_sensor_cfg": SceneEntityCfg("right_finger_contact"),
            "object_cfg": SceneEntityCfg("cube"),
            "distance_threshold": 0.05,
            "force_threshold": 1.0,
        }
    )

    object_lifting_smooth = RewTerm(
        func=man_mdp.reward_grasped_lift_smooth,
        weight=10.0,
        params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            "left_finger_sensor_cfg": SceneEntityCfg("left_finger_contact"),
            "right_finger_sensor_cfg": SceneEntityCfg("right_finger_contact"),
            "object_cfg": SceneEntityCfg("cube"),
            "target_height": 1.25,  # 0.9m table height + 0.3m above
            "std": 0.05,
            "distance_threshold": 0.05,
            "force_threshold": 1.0,
        }
    )

    # Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    # Original reward weight curriculum terms
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate_l2", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if arm links 2-7 touch anything (self-collision / env collision).
    # 5N threshold avoids false triggers from physics-settling transients after reset.
    arm_collision = DoneTerm(
        func=man_mdp.terminate_on_arm_contact,
        params={
            "sensor_cfg": SceneEntityCfg("arm_contact"),
            "force_threshold": 5.0,
        },
    )


@configclass
class BlockEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_max_rigid_contact_count = 10 * 2**20
