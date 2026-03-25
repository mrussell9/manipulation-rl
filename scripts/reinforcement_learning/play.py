# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--wandb_policy", action="store_true", default=False, help="Use policy saved on WandB for inference.")
parser.add_argument("--wandb_run", type=str, default="", help="WandB run path (e.g. 'username/project/run_id').")
parser.add_argument("--wandb_model", type=str, default="", help="Name of the model artifact to load from WandB.")

# Dataset recording (LeRobot)
parser.add_argument(
    "--record_dataset",
    action="store_true",
    default=False,
    help="Record observations/actions to a local LeRobot dataset under /data/openpi/fine_tune/",
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="/data/openpi/fine_tune",
    help="Parent directory where the LeRobot dataset folder will be created.",
)
parser.add_argument(
    "--dataset_repo_id",
    type=str,
    default="isaac_franka_block_pick",
    help="Local dataset name (folder) to write under dataset_root.",
)
parser.add_argument(
    "--dataset_task",
    type=str,
    default="Pick up the red block and hold it",
    help="Task string to store in the dataset (used for prompt/task conditioning).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras when video or dataset recording is enabled
if args_cli.video or args_cli.record_dataset:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from manipulation_tasks.reinforcement_learning.pick_up_block.runners import BlockRunner as OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import manipulation_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from manipulation_utils.wandb_utils import load_wandb_policy

# PLACEHOLDER: Extension template (do not remove this comment)


def _to_uint8_chw_image(tensor: torch.Tensor):
    """Convert IsaacLab RGB (HWC uint8/float) tensor to numpy CHW uint8."""
    img = tensor.squeeze().detach().cpu()
    img = img.clamp(0, 255).to(torch.uint8)
    img_np = img.numpy()
    # IsaacLab: HWC -> LeRobot/OpenPI: CHW
    return img_np.transpose(2, 0, 1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif args_cli.wandb_policy:
        # load the policy
        resume_path = ""
        log_dir = ""
        # load configuration
        run_path = args_cli.wandb_run
        model_name = args_cli.wandb_model
        resume_path, _ = load_wandb_policy(run_path, model_name, log_root_path)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    dataset = None
    if args_cli.record_dataset:
        # Import only when needed so play works without lerobot installed.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        dataset_root = Path(args_cli.dataset_root)
        dataset = LeRobotDataset.create(
            repo_id=args_cli.dataset_repo_id,
            root=dataset_root,
            fps=int(1.0 / dt) if dt > 0 else 50,
            robot_type="franka_panda",
            features={
                "observation.images.wrist": {"dtype": "video", "shape": (3, 224, 224), "names": ["channels", "height", "width"]},
                "observation.images.base": {"dtype": "video", "shape": (3, 224, 224), "names": ["channels", "height", "width"]},
                "observation.state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
                },
            },
        )

    # reset environment
    obs = env.get_observations()
    import pdb; pdb.set_trace()
    timestep = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)

                if dataset is not None:
                    # These keys assume your env exposes camera observations under obs["camera"] and vector obs under obs["policy"].
                    wrist_img = _to_uint8_chw_image(obs["camera"]["wrist_camera"])
                    base_img = _to_uint8_chw_image(obs["camera"]["base_camera"])
                    state = obs["policy"][:, 0:8].squeeze().cpu().numpy()
                    act = actions.squeeze().cpu().numpy()
                    dataset.add_frame(
                        {
                            "observation.images.wrist": wrist_img,
                            "observation.images.base": base_img,
                            "observation.state": state,
                            "action": act,
                            "task": args_cli.dataset_task,
                        }
                    )

                # env stepping
                obs, _, dones, _ = env.step(actions)

                if dataset is not None and dones.any():
                    dataset.save_episode()

                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)

            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # close the simulator
        env.close()

        if dataset is not None:
            # Important: closes parquet writers and writes footers/metadata.
            try:
                dataset.finalize()
            except Exception:
                pass


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()