from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

"""Path to the extension source directory."""

##
# Configuration
##


# cabinet
CABINET_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Cabinet",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(5.0, 0, 0.4),
        rot=(0.1, 0.0, 0.0, 0.0),
        joint_pos={
            "door_left_joint": 0.0,
            "door_right_joint": 0.0,
            "drawer_bottom_joint": 0.0,
            "drawer_top_joint": 0.0,
        },
    ),
    actuators={
        "drawers": ImplicitActuatorCfg(
            joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
            effort_limit=87.0,
            velocity_limit=100.0,
            stiffness=10.0,
            damping=1.0,
        ),
        "doors": ImplicitActuatorCfg(
            joint_names_expr=["door_left_joint", "door_right_joint"],
            effort_limit=87.0,
            velocity_limit=100.0,
            stiffness=10.0,
            damping=2.5,
        ),
    },
)