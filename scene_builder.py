# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for building the Find-the-Fruit scene: deformable plant setup and physics attachments."""

from __future__ import annotations

import omni.usd
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.utils.math import quat_from_euler_xyz

import torch


def spawn_plant(stage_func_target: str, translation: tuple, usd_path: str, device: str) -> None:
    """Spawn the plant USD at the given prim path with a fixed pose."""
    import torch as _torch
    euler = _torch.tensor([0.0, 0.0, 0.0], device=device)
    quat = quat_from_euler_xyz(*euler).cpu().numpy()
    cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=(0.05, 0.05, 0.05))
    cfg.func(
        stage_func_target,
        cfg,
        translation=translation,
        orientation=(quat[0], quat[1], quat[2], quat[3]),
    )


def make_deformable(stage, deformable_material_path: str, prim_dict: dict, simulation_resolution: int = 10) -> None:
    """Apply deformable-body physics to plant mesh prims and assign material."""
    key, value = list(prim_dict.items())[0]
    deformableUtils.add_deformable_body_material(
        stage,
        deformable_material_path,
        youngs_modulus=7.5e8,
        poissons_ratio=0.49,
        damping_scale=1.0,
        dynamic_friction=0.5,
        density=2.0,
    )
    deformableUtils.add_physx_deformable_body(
        stage,
        value.GetPath(),
        collision_simplification=True,
        simulation_hexahedral_resolution=simulation_resolution,
        solver_position_iteration_count=128,
        self_collision=False,
    )
    physicsUtils.add_physics_material_to_prim(stage, value.GetPrim(), deformable_material_path)

    for _, value in list(prim_dict.items())[1:]:
        deformableUtils.add_physx_deformable_body(
            stage,
            value.GetPath(),
            collision_simplification=True,
            simulation_hexahedral_resolution=simulation_resolution,
            solver_position_iteration_count=128,
            self_collision=False,
        )
        physicsUtils.add_physics_material_to_prim(stage, value.GetPrim(), deformable_material_path)


def attach_cylinder_to_ground(stage, prim_dict: dict, prim_name: str) -> None:
    """Attach the base stalk to the ground plane and sub-stalks to the main stalk via PhysX attachments."""
    key, value = list(prim_dict.items())[0]
    attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
    att.GetActor0Rel().SetTargets([value.GetPath()])
    att.GetActor1Rel().SetTargets(["/World/ground/GroundPlane/CollisionPlane"])
    api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    api.GetPrim().GetAttribute("physxAutoAttachment:deformableVertexOverlapOffset").Set(0.05)
    api.GetPrim().GetAttribute("physxAutoAttachment:enableDeformableVertexAttachments").Set(True)
    api.GetPrim().GetAttribute("physxAutoAttachment:enableRigidSurfaceAttachments").Set(True)
    api.GetPrim().GetAttribute("physxAutoAttachment:enableCollisionFiltering").Set(True)
    api.GetPrim().GetAttribute("physxAutoAttachment:collisionFilteringOffset").Set(0.01)
    api.GetPrim().GetAttribute("physxAutoAttachment:enableDeformableFilteringPairs").Set(True)

    for key, value in list(prim_dict.items())[1:]:
        attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
        att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
        att.GetActor0Rel().SetTargets([value.GetPath()])
        att.GetActor1Rel().SetTargets([f"{prim_name}/stalk/plant_023"])
        api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
        api.GetPrim().GetAttribute("physxAutoAttachment:deformableVertexOverlapOffset").Set(0.005)


def add_plant(stage, env_prim_paths: list[str], deformable_material_path: str, n_plants: int = 1) -> list:
    """Register deformable plant objects and configure their physics for all envs.

    Returns a flat list of DeformableObject instances (stalk + 5 sub-stalks per plant).
    """
    deformable_objects: list[DeformableObject] = []

    for plant_number in range(1, n_plants + 1):
        stalk_cfg = DeformableObjectCfg(
            prim_path=f"/World/envs/env_.*/Plant{plant_number}/stalk",
            debug_vis=True,
        )
        deformable_objects.append(DeformableObject(stalk_cfg))

        for i in range(1, 6):
            sub_cfg = DeformableObjectCfg(
                prim_path=f"/World/envs/env_.*/Plant{plant_number}/stalk{i}",
                debug_vis=True,
            )
            deformable_objects.append(DeformableObject(sub_cfg))

        for env_path in env_prim_paths:
            plant_prim = stage.GetPrimAtPath(f"{env_path}/Plant{plant_number}")
            plant_meshes = [mesh.GetAllChildren()[0] for mesh in plant_prim.GetAllChildren()]
            plant_meshes = plant_meshes[1:]
            prim_dict = dict(zip(plant_prim.GetAllChildrenNames()[1:], plant_meshes))
            make_deformable(stage, deformable_material_path, prim_dict)
            attach_cylinder_to_ground(stage, prim_dict, f"{env_path}/Plant{plant_number}")

    return deformable_objects
