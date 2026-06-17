# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .assets import ROBOT_USD

MASK_SIZE = 30


@configclass
class FindTheFruitEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_scale = 1.0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_temp_buffer_capacity=2**26,
            gpu_found_lost_pairs_capacity=2**27,
            min_position_iteration_count=128,
        ),
        gravity=(0.0, 0.0, -9.81),
    )

    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/mybuddy",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        actuators={
            ".*": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=1000.0,
                velocity_limit=10.0,
                stiffness=800.0,
                damping=4.0,
            ),
        },
    )

    arm_dof_name = [
        "bc2bl",
        "left_arm_j1",
        "left_arm_j2",
        "left_arm_j3",
        "left_arm_j4",
        "left_arm_j5",
    ]
    link_dof_name = ["left_arm_l1", "left_arm_l3", "left_arm_l4", "left_arm_l5", "left_arm_l6"]

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0.0, 0.0, 0.53833, 0.84274), convention="opengl"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=4.0,
            focus_distance=0.6,
            horizontal_aperture=5.760000228881836,
            vertical_aperture=3.5999999046325684,
            clipping_range=(0.076, 10.0),
            f_stop=240.0,
        ),
        width=256,
        height=256,
    )

    num_goal_cubes: int = 2

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(arm_dof_name),))
    observation_space = {
        "joints": gym.spaces.Box(low=-20.00, high=20.0, shape=(len(arm_dof_name),), dtype=np.float32),
        "rgb": gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(tiled_camera.height, tiled_camera.width, 5),
            dtype=np.float32,
        ),
        "ee_position": gym.spaces.Box(low=-20.00, high=20.0, shape=(3,), dtype=np.float32),
    }

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=50, env_spacing=2.5, replicate_physics=False)

    max_y_pos = 0.0
    maxlen = 10
    rew_scale_alive = -0.1
