# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from collections.abc import Sequence

import omni.usd
import isaacsim.core.utils.stage as stage_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .assets import PLANT_USD, BACKGROUND_USD
from .env_cfg import FindTheFruitEnvCfg, MASK_SIZE
from .scene_builder import add_plant, spawn_plant
from .vision_utils import quaternion_to_rotation_matrix, create_batched_masks, calculate_plant_mask


class FindTheFruitEnv(DirectRLEnv):
    cfg: FindTheFruitEnvCfg

    def __init__(self, cfg: FindTheFruitEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_dof_idx, _ = self._robot.find_joints(self.cfg.arm_dof_name)
        self._arm_pos_idx = self._robot.find_bodies(["left_arm_l6"])
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        self.image_sequences = deque(maxlen=20)
        self.root_init_angles = (
            torch.deg2rad(torch.tensor([0, -90, 0, 120, -120, 180], device=self.device))
            .repeat(self.num_envs, 1)
            .view(-1, len(self.cfg.arm_dof_name))
        )
        self.last_angles = self.root_init_angles.clone()

        # Camera projection matrix (computed once at init)
        self.camera_intrinsics = self._tiled_camera.data.intrinsic_matrices
        rotation_matrix = quaternion_to_rotation_matrix(self._tiled_camera.data.quat_w_opengl)
        translation_vector = -torch.bmm(rotation_matrix, self._tiled_camera.data.pos_w.unsqueeze(-1))
        extrinsic_matrix = torch.cat((rotation_matrix, translation_vector), dim=2)
        self.projection_matrix = torch.bmm(self.camera_intrinsics, extrinsic_matrix)

        # Multi-goal buffers
        self.root_cube_positions = torch.stack(
            [cube.data.root_com_state_w.clone() for cube in self.goal_cubes], dim=1
        )
        self.target_cube_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cube_maskas = torch.zeros(
            (self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, self.cfg.num_goal_cubes),
            device=self.device,
            dtype=torch.float32,
        )
        self.rgb_image = torch.zeros(
            (self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.occlusion_time_below_threshold = torch.zeros(self.num_envs, 1, device=self.device)
        self.joint_readings = self._robot.data.joint_pos[:, self._arm_dof_idx]

    def close(self):
        super().close()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        spawn_plant(
            "/World/envs/env_.*/Plant1",
            translation=(0.06, -0.25, 0.0),
            usd_path=PLANT_USD,
            device=self.device,
        )

        light_cfg = sim_utils.SphereLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), radius=1.0)
        light_cfg.func("/World/envs/env_.*/Light", light_cfg, translation=(0.0, -0.4, 2.0))

        bg_color = (0.6, 0.6, 0.6)
        background_cfg = sim_utils.UsdFileCfg(
            usd_path=BACKGROUND_USD, scale=(1.0, 1.0, 1.0)
        )
        background_cfg.func(
            "/World/envs/env_.*/BackgroundPlane",
            background_cfg,
            translation=(0.0, 0.25, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        ground_cfg = sim_utils.CuboidCfg(
            size=(1.25, 0.25, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=bg_color)
        )
        ground_cfg.func("/World/envs/env_.*/GroundPlane2", ground_cfg, translation=(0.0, -0.35, 0.0))

        # Goal cubes (one per fruit class)
        fruit_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        self.goal_cubes = []
        for i in range(self.cfg.num_goal_cubes):
            cube_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/GoalCube_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=fruit_colors[i % len(fruit_colors)]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.4, 0.28)),
            )
            self.goal_cubes.append(RigidObject(cube_cfg))

        # Contact sensors per link
        contact_sensors = []
        for link_name in self.cfg.link_dof_name:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/mybuddy/{link_name}",
                update_period=0.0,
                history_length=6,
                debug_vis=True,
            ))
            contact_sensors.append(sensor)

        self.deformable_material_path = omni.usd.get_stage_next_free_path(self.stage, "/plant_material", True)

        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[])

        self.deformable_objects_list = add_plant(
            self.stage, self.scene.env_prim_paths, self.deformable_material_path, n_plants=1
        )

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        for i, sensor in enumerate(contact_sensors):
            self.scene.sensors[f"contact_sensor_{i}"] = sensor
        for i, cube in enumerate(self.goal_cubes):
            self.scene.rigid_objects[f"goal_cube_{i}"] = cube
        for i, obj in enumerate(self.deformable_objects_list):
            self.scene.deformable_objects[f"stalk_{i}"] = obj

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.detection_sequence = deque(
            [torch.zeros(self.num_envs, device=self.device) for _ in range(self.cfg.maxlen)],
            maxlen=self.cfg.maxlen,
        )
        self.done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.raw_actions = actions.clone()
        self.actions = (
            torch.clamp(self.raw_actions, -1, 1) * self.dt * self.action_scale
            + self._robot.data.joint_pos[:, self._arm_dof_idx]
        )
        self.actions[:, 0].clamp_(-1.0, 0.7)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.actions, joint_ids=self._arm_dof_idx)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        joint_readings = self._robot.data.joint_pos[:, self._arm_dof_idx]

        self.rgb_image = self._tiled_camera.data.output["rgb"].clone() / 255.0
        self.rgb_image_raw = self.rgb_image.clone()
        self.rgb_image -= torch.mean(self.rgb_image, dim=(1, 2), keepdim=True)
        self.rgb_image = self.rgb_image.to(dtype=torch.float32)

        self.depth_image = self._tiled_camera.data.output["depth"].clone()
        self.depth_image[self.depth_image == float("inf")] = 0

        self.ee_position = (
            torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]])
            - self._robot.data.root_com_pos_w
        )

        cube_masks_selected = torch.gather(
            self.cube_maskas,
            3,
            self.target_cube_idx.view(-1, 1, 1, 1).expand(
                -1, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 1
            ),
        )
        image_obs = torch.cat((self.rgb_image, cube_masks_selected, self.depth_image), dim=3)

        return {"policy": {
            "rgb": image_obs.to(dtype=torch.float32),
            "joints": joint_readings.to(dtype=torch.float32),
            "ee_position": self.ee_position.to(dtype=torch.float32),
        }}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        plant_mask = calculate_plant_mask(self.depth_image)

        one_hot = F.one_hot(self.target_cube_idx, num_classes=self.cfg.num_goal_cubes).float().unsqueeze(1).unsqueeze(1)
        selected_mask = torch.sum(self.cube_maskas * one_hot, dim=3, keepdim=True)
        occlusion_pixels = torch.sum(selected_mask * plant_mask, dim=(1, 2, 3)).reshape(-1, 1)
        occlusion_reward = (1 - occlusion_pixels / (MASK_SIZE * MASK_SIZE)) * 10.0

        full_visibility_reward = torch.where(
            occlusion_pixels <= 160,
            torch.tensor(3.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        ).reshape(self.num_envs)

        full_visibility_mask = full_visibility_reward > 0
        self.detection_sequence.append(full_visibility_mask)

        sustained_detection = torch.all(torch.stack(list(self.detection_sequence)), dim=0)
        sustained_reward = sustained_detection.float() * 20.0

        action_penalty = torch.norm(self.raw_actions.clamp(-1.0, 1.0), dim=1) * -0.06
        action_penalty *= full_visibility_mask.float()

        self.done = sustained_detection
        occlusion_reward = occlusion_reward.reshape(self.num_envs)

        return self.contact_reward + full_visibility_reward + sustained_reward + action_penalty + occlusion_reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.contact_reward = self._calculate_contact_reward(len(self.cfg.link_dof_name))
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.in_contact | self.done, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids][:, self._arm_dof_idx] + self.root_init_angles[env_ids]
        if np.random.uniform() < 0.5:
            joint_pos[:, 2] = sample_uniform(-0.52, -0.17, joint_pos[:, 2].shape, self.device)
        else:
            joint_pos[:, 2] = sample_uniform(0.2, 0.3, joint_pos[:, 2].shape, self.device)
        self.last_angles[env_ids] = joint_pos.clone()

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, self._arm_dof_idx, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, self._arm_dof_idx, env_ids)

        for obj in self.deformable_objects_list:
            obj: DeformableObject
            nodal_state = obj.data.default_nodal_state_w.clone()[env_ids]
            obj.write_nodal_state_to_sim(nodal_state, env_ids)

        # Randomise target cube and reposition all cubes
        rand_idx = torch.randint(0, self.cfg.num_goal_cubes, (len(env_ids),), device=self.device)
        self.target_cube_idx[env_ids] = rand_idx

        base_states = self.root_cube_positions[env_ids]
        new_positions = base_states[:, :, :3].clone()
        new_positions[..., 0] += sample_uniform(0.00, 0.15, new_positions[..., 0].shape, self.device)

        for k, cube in enumerate(self.goal_cubes):
            states_k = base_states[:, k, :].clone()
            states_k[:, :3] = new_positions[:, k, :3]
            cube.write_root_pose_to_sim(states_k[:, :7], env_ids)
            cube.write_root_velocity_to_sim(torch.zeros_like(states_k[:, 7:]), env_ids)

        # Recompute cube masks for all channels
        for k in range(self.cfg.num_goal_cubes):
            pos_k = new_positions[:, k, :3]
            ones = torch.ones((pos_k.shape[0], 1), device=self.device)
            pts_hom = torch.cat((pos_k, ones), dim=1).unsqueeze(2)
            img_hom = torch.bmm(self.projection_matrix[env_ids], pts_hom).squeeze(2)
            img_2d = img_hom[:, :2] / img_hom[:, 2:].expand(-1, 2)
            masks_k = torch.flip(
                create_batched_masks(
                    (self.cfg.tiled_camera.width, self.cfg.tiled_camera.height), img_2d, MASK_SIZE
                ),
                dims=[2],
            ).to(dtype=torch.float32, device=self.device)
            self.cube_maskas[env_ids, :, :, k] = masks_k

        raw_rgb = (self._tiled_camera.data.output["rgb"].clone()[env_ids] / 255.0).to(dtype=torch.float32)
        self.rgb_image[env_ids] = raw_rgb - torch.mean(raw_rgb, dim=(1, 2), keepdim=True)

        self.last_ee_position = (
            torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]])
            - self._robot.data.root_com_pos_w
        )

        tensor_seq = torch.stack(list(self.detection_sequence))
        tensor_seq[:, env_ids] = 0
        self.detection_sequence = deque(list(tensor_seq), maxlen=self.cfg.maxlen)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calculate_contact_reward(self, num_sensors: int) -> torch.Tensor:
        readings = torch.cat(
            [self.scene.sensors[f"contact_sensor_{i}"].data.net_forces_w for i in range(num_sensors)], dim=1
        )
        self.in_contact = torch.any(readings != 0, dim=(1, 2))
        return torch.where(self.in_contact, torch.tensor(-5.0, device=self.device), torch.tensor(0.0, device=self.device))
