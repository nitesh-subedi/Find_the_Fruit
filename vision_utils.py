# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401  (keep for downstream callers)


def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert a batch of quaternions (w, x, y, z) to rotation matrices (N, 3, 3)."""
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    quat = quat / quat.norm(dim=1, keepdim=True)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=1,
    ).reshape(-1, 3, 3)


def create_batched_masks(
    image_size: tuple[int, int],
    cube_pixel_locations: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Return binary masks (n_envs, H, W) with an n×n block set to 1 at each cube pixel location."""
    n_envs = cube_pixel_locations.shape[0]
    height, width = image_size
    masks = torch.zeros((n_envs, height, width), dtype=torch.float32, device=cube_pixel_locations.device)
    px = cube_pixel_locations[:, 0]
    py = cube_pixel_locations[:, 1]
    half = n // 2
    x0 = torch.clamp(px - half, 0, width - 1).int()
    x1 = torch.clamp(px + half + 1, 0, width).int()
    y0 = torch.clamp(py - half, 0, height - 1).int()
    y1 = torch.clamp(py + half + 1, 0, height).int()
    for i in range(n_envs):
        masks[i, y0[i]:y1[i], x0[i]:x1[i]] = 1.0
    return masks


@torch.jit.script
def calculate_plant_mask(depth_obs: torch.Tensor) -> torch.Tensor:
    """Return a float mask selecting depth values corresponding to the plant (0.1–0.4 m)."""
    return ((depth_obs >= 0.1) & (depth_obs <= 0.4)).clone().detach().float()
