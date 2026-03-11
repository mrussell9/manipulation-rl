# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch

from rsl_rl.env import VecEnv


class Pi0Runner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, device="cpu"):
        self.device = device
        self.env = env

        obs = self.env.get_observations()


    def run(self):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        # start learning
        obs = self.env.get_observations().to(self.device)

        while True:
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
