# 🍎 Find the Fruit

### Zero-Shot Sim2Real RL for Occlusion-Aware Plant Manipulation

> **Paper**: [Find the Fruit: Zero-Shot Sim2Real RL for Occlusion-Aware Plant Manipulation](https://arxiv.org/abs/2505.16547)  
> **Authors**: Nitesh Subedi, Hsin-Jung Yang, Devesh K. Jha, Soumik Sarkar  
> **arXiv**: [2505.16547](https://arxiv.org/abs/2505.16547)

---

## Overview

This package implements the **Find the Fruit** simulation environment — a GPU-accelerated Direct RL environment built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab). A robotic arm learns to reposition plant stems and leaves to reveal occluded target fruit(s), trained entirely in simulation and transferred zero-shot to the real world.

Autonomous harvesting in open-field settings presents a complex manipulation problem. Plants exhibit significant structural variation (every plant is different), and target fruits are often heavily occluded by stems and leaves. This environment enables learning occlusion-aware manipulation policies that decouple **high-level kinematic planning** from **low-level compliant control**, simplifying sim-to-real transfer and allowing the learned policy to generalize across plants with different stiffness and morphology.

In real-world experiments with multiple plant setups, the system achieves up to **86.7% success** in exposing target fruits.

---

## Environment Details

| Property | Value |
|---|---|
| **Gym ID** | `Isaac-FindTheFruit-Direct-v0` |
| **Robot** | MyBuddy (left arm, 6 DoF) |
| **Simulator** | NVIDIA Isaac Sim (PhysX) |
| **RL Framework** | [SKRL](https://github.com/Toni-SM/skrl) (PPO) |
| **Episode Length** | 5.0 s |
| **Sim dt** | 1/60 s (decimation = 2) |
| **Num Envs (default)** | 50 |

### Robot Configuration

The environment uses the **MyBuddy** dual-arm robot (left arm only). Six joints are controlled via position targets:

| Joint | Name |
|---|---|
| Base rotation | `bc2bl` |
| Shoulder | `left_arm_j1` |
| Elbow | `left_arm_j2` |
| Wrist 1 | `left_arm_j3` |
| Wrist 2 | `left_arm_j4` |
| Wrist 3 | `left_arm_j5` |

### Action Space

- **Type**: `Box(-1, 1)` with shape `(6,)`
- Actions represent normalized joint velocity offsets scaled by `action_scale × dt`, applied as delta position targets on top of the current joint positions.

### Observation Space

| Key | Shape | Description |
|---|---|---|
| `joints` | `(6,)` | Current joint positions |
| `rgb` | `(256, 256, 5)` | Mean-subtracted RGB (3ch) + target fruit mask (1ch) + depth (1ch) |
| `ee_position` | `(3,)` | End-effector position relative to robot base |

### Scene

- **Plant**: Deformable-body USD model (`plant_v21.usd`) with PhysX soft-body simulation — stalks and sub-stalks are attached to the ground plane and respond to contact forces.
- **Goal Cubes**: Colored spheres representing target fruit positions, randomly repositioned each episode.
- **Camera**: 256×256 tiled RGB-D camera mounted overhead with pinhole projection.
- **Background**: USD backdrop with neutral coloring for clean segmentation.

---

## Reward Structure

The reward function encourages the robot to expose occluded fruits while minimizing unnecessary contact:

| Component | Description |
|---|---|
| **Occlusion reward** | `(1 − occluded_pixels / mask_area) × 10` — proportional to how much of the target fruit is visible |
| **Visibility bonus** | `+3.0` when occlusion pixels drop below threshold (160 px) |
| **Sustained detection** | `+20.0` for maintaining visibility for 10 consecutive steps |
| **Action penalty** | `-0.06 × ‖action‖` — encourages efficiency once fruit is visible |
| **Contact penalty** | `-5.0` on any link contact with the plant — discourages damaging collisions |

The episode terminates early upon sustained detection (success) or on contact (failure).

---

## Project Structure

```
mybuddy/
├── __init__.py              # Gym registration (Isaac-FindTheFruit-Direct-v0)
├── find_the_fruit_env.py    # Main environment class (FindTheFruitEnv)
├── env_cfg.py               # Environment & robot configuration (FindTheFruitEnvCfg)
├── scene_builder.py         # Deformable plant spawning & physics attachments
├── vision_utils.py          # Camera projection, mask generation, plant segmentation
├── utils.py                 # Depth noise augmentation (sim2real domain randomization)
├── assets.py                # USD asset paths
├── assets/
│   ├── mybuddy_rotate.usd   # MyBuddy robot model
│   ├── plant_v21.usd        # Deformable plant model
│   └── background.usd       # Scene background
└── agents/
    └── skrl_ppo_cfg.yaml    # SKRL PPO training hyperparameters
```

---

## Getting Started

### Prerequisites

- [NVIDIA Isaac Sim 4.5+](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (installed and configured)
- [SKRL](https://github.com/Toni-SM/skrl) RL library

### Training

```bash
# From the IsaacLab root directory
./isaaclab.sh -p -m skrl_cfg --task Isaac-FindTheFruit-Direct-v0 --headless
```

### Key Training Hyperparameters (PPO)

| Parameter | Value |
|---|---|
| Rollout steps | 32 |
| Learning epochs | 5 |
| Mini-batches | 8 |
| Learning rate | 3e-4 (KL-adaptive) |
| Discount (γ) | 0.99 |
| GAE (λ) | 0.95 |
| Total timesteps | 400,000 |
| Gradient clip | 1.0 |

---

## Citation

If you find this code useful in your research, please cite:

```bibtex
@article{subedi2025findthefruit,
  title   = {Find the Fruit: Zero-Shot Sim2Real RL for Occlusion-Aware Plant Manipulation},
  author  = {Subedi, Nitesh and Yang, Hsin-Jung and Jha, Devesh K. and Sarkar, Soumik},
  journal = {arXiv preprint arXiv:2505.16547},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.16547}
}
```