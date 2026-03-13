# Multi-Agent Reinforcement Learning for Cooperative Origami Folding

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Classic%2011-orange)](http://gazebosim.org/)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-PPO-red)](https://stable-baselines3.readthedocs.io/)

A multi-agent reinforcement learning system that trains two UR5 robotic arms to cooperatively perform a 9-step origami dragon fold sequence in ROS 2 and Gazebo simulation.

---

## Overview

This project implements a centralized PPO (Proximal Policy Optimization) policy that simultaneously controls all 12 joints of two UR5 robotic arms. The robots are positioned facing each other on opposite sides of a table and learn to coordinate their end-effectors to reach geometrically complementary target positions for each step of an origami dragon fold.

**Key Features:**

- Single centralized PPO policy controlling 12 DOF (6 per robot)
- Real-time joint state feedback via `/joint_states` ROS 2 topic
- Curriculum learning with progressive success threshold (0.13m to 0.07m)
- Per-robot asymmetric target assignment for cooperative fold geometry
- UR5 forward kinematics with correct coordinate frame transforms for mirrored robot configuration
- VecNormalize for stable reward scaling
- TensorBoard logging

<!-- **Demo Video:** [YouTube Link] -->

---

## System Architecture

```
+----------------------------------------------------------+
|                    Training Loop (PPO)                    |
|                                                          |
|   Observation (24-dim)          Action (12-dim)          |
|   +------------------+         +---------------------+   |
|   | Joint angles x12 |         | delta-joint R1 x6   |   |
|   | Goal R1 (x,y,z)  |--PPO--> | delta-joint R2 x6   |   |
|   | Goal R2 (x,y,z)  |         +---------------------+   |
|   | EE R1   (x,y,z)  |                  |                |
|   | EE R2   (x,y,z)  |                  v                |
|   +------------------+       /joint_trajectory topic     |
+----------+-----------------------------------------+------+
           |                                         |
           v                                         v
+---------------------+              +------------------------+
|  /joint_states      |              |  Gazebo Classic 11     |
|  (real feedback)    |<-------------|  Two UR5 Arms + Paper  |
+---------------------+              +------------------------+
```

**Robot Configuration:**

```
     Robot 1                        Robot 2
  (x = -0.25m)                  (x = +0.25m)
  facing right -->           <-- facing left (rotated 180 deg)
       |                              |
       +------------- Paper ----------+
                  (x=0, z=0.312m)
```

---

## Project Structure

```
origami_rl/
├── scripts/
│   ├── train.py              # Main PPO training script
│   ├── train_resume.py       # Resume training from checkpoint
│   ├── enjoy.py              # Run trained policy in Gazebo
│   ├── origami_env.py        # Gymnasium environment (core)
│   ├── task_manager.py       # Fold step management
│   ├── utils.py              # UR5 forward kinematics + DH params
│   ├── spawn_paper.py        # Spawns paper object in Gazebo
│   └── dragon_points.json    # 9-step fold target positions
├── urdf/
│   └── multi_ur5.urdf.xacro  # Dual UR5 robot description
├── config/
│   └── ur_controllers.yaml   # ROS 2 joint trajectory controllers
└── README.md
```

---

## Requirements

| Dependency        | Version          |
| ----------------- | ---------------- |
| Ubuntu            | 22.04 LTS        |
| ROS 2             | Humble Hawksbill |
| Gazebo            | Classic 11       |
| Python            | 3.10             |
| stable-baselines3 | >= 2.0           |
| gymnasium         | >= 0.26          |
| numpy             | >= 1.24          |
| torch             | >= 2.0 (CPU)     |

**ROS 2 Packages required:**

- `ur_description`
- `ur_simulation_gazebo`
- `gazebo_ros2_control`
- `joint_state_broadcaster`

---

## Installation

**1. Clone into your ROS 2 workspace:**

```bash
cd ~/ros2_ws/src
git clone https://github.com/YOUR_USERNAME/origami_rl.git
```

**2. Install Python dependencies:**

```bash
pip install stable-baselines3 gymnasium numpy torch --break-system-packages
```

**3. Install ROS 2 UR packages:**

```bash
sudo apt install ros-humble-ur-description \
                 ros-humble-ur-simulation-gazebo \
                 ros-humble-gazebo-ros2-control \
                 ros-humble-joint-state-broadcaster
```

**4. Build the workspace:**

```bash
cd ~/ros2_ws
colcon build --packages-select origami_rl
source install/setup.bash
```

**5. Set initial joint positions so robots start near the paper:**

```bash
cat > ~/ros2_ws/install/ur_description/share/ur_description/config/initial_positions.yaml << 'YAML'
shoulder_pan_joint: 0.496
shoulder_lift_joint: -1.191
elbow_joint: -2.707
wrist_1_joint: -1.57
wrist_2_joint: -1.57
wrist_3_joint: 0.0
YAML
```

---

## Usage

### Step 1 — Launch Gazebo

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py ur_type:=ur5e
```

Wait for both controllers to activate:

```
[INFO] Configured and activated robot1_joint_trajectory_controller
[INFO] Configured and activated robot2_joint_trajectory_controller
```

### Step 2 — Spawn Paper

```bash
cd ~/ros2_ws/src/origami_rl/scripts
python3 spawn_paper.py
```

### Step 3 — Train or Run

```bash
# Train from scratch
CUDA_VISIBLE_DEVICES= python3 train.py

# Resume from checkpoint
CUDA_VISIBLE_DEVICES= python3 train_resume.py

# Run trained policy (evaluation)
python3 enjoy.py
```

---

## Training

### Hyperparameters

| Parameter         | Value                         |
| ----------------- | ----------------------------- |
| Algorithm         | PPO (Stable-Baselines3)       |
| Total timesteps   | 500,000                       |
| Learning rate     | 3e-4                          |
| n_steps           | 2048                          |
| batch_size        | 64                            |
| n_epochs          | 10                            |
| gamma             | 0.99                          |
| ent_coef          | 0.001                         |
| clip_range        | 0.2                           |
| Policy network    | MLP [256, 256, 128] + Tanh    |
| Action space      | Box(-0.05, 0.05, shape=(12,)) |
| Observation space | Box(shape=(24,))              |

### Monitor with TensorBoard

```bash
tensorboard --logdir ./logs/ --port 6006
# Open http://localhost:6006
```

### Curriculum Learning Schedule

The success threshold decreases linearly during training:

```
success_dist = 0.13 - progress x (0.13 - 0.07)
```

- Start: 0.13m (just below home-to-goal distance of ~0.16m, forces real movement)
- End: 0.07m (precise end-effector positioning required)

---

## Results

| Metric                      | Value            |
| --------------------------- | ---------------- |
| Explained Variance          | 0.93             |
| Peak ep_rew_mean            | +6.0             |
| Folds completed per episode | 2-3 steps        |
| Training speed              | 30-42 steps/sec  |
| Total policy parameters     | 211,865          |
| Training time               | ~4-5 hours (CPU) |

### Fold Sequence

| Stage               | Steps | Fold Names                                        |
| ------------------- | ----- | ------------------------------------------------- |
| 1 - Base Creases    | 1-2   | diagonal_fold_1, diagonal_fold_2                  |
| 2 - Square Base     | 3-4   | squash_fold_left, squash_fold_right               |
| 3 - Bird Base       | 5-6   | petal_fold_front, petal_fold_back                 |
| 4 - Dragon Features | 7-9   | head_reverse_fold, tail_reverse_fold, wing_spread |

### Sample Terminal Output

```
--- Episode 1 ---
  step  150 | fold 0/9 | reward  57.9 | R1goal [-0.06  0.06  0.39] | R2goal [ 0.06 -0.06  0.39]
FOLD: diagonal_fold_1  d1=0.126 d2=0.042 thresh=0.13
FOLD: diagonal_fold_2  d1=0.128 d2=0.105 thresh=0.13
FOLD: squash_fold_left d1=0.101 d2=0.088 thresh=0.13
  step  300 | fold 2/9 | reward 107.2 | R1goal [-0.07  0.00  0.41] | R2goal [ 0.07  0.00  0.41]
  Episode ended | steps=793 | reward=147.5
```

---

## Key Design Decisions

### 1. Real Joint State Feedback

The environment subscribes to `/joint_states` rather than tracking accumulated action commands. Accumulated tracking drifts progressively from actual Gazebo joint positions due to controller tolerance rejections, making the observation space inaccurate and preventing learning.

### 2. Coordinate Frame Transform for Robot 2

Robot 2 is rotated 180 degrees in the URDF. The FK implementation accounts for this transform:

```python
# Robot 1 (no extra rotation):
world_pos = local_pos + [-0.25, 0, 0.31]

# Robot 2 (Rz(pi) flips x and y axes):
world_pos = [-local_x + 0.25, -local_y, local_z + 0.31]
```

### 3. Low Entropy Coefficient

Using `ent_coef=0.01` caused policy unlearning — the action standard deviation grew from 1.0 to 1.84 over 100k steps as the entropy bonus began to dominate the reward signal. Reducing to `ent_coef=0.001` stabilized training completely.

### 4. Curriculum Threshold Must Stay Below Home Distance

If `success_dist > distance(home_position, goal)`, robots earn success bonuses without actually moving. The curriculum is initialized at 0.13m while the home-to-goal distance is ~0.16m, ensuring genuine movement is required from the first episode.

---

## Common Issues and Fixes

| Issue                                      | Cause                                                    | Fix                                                     |
| ------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------- |
| Robots start pointing straight up (z=1.0m) | `initial_positions.yaml` not applied                     | Add `initial_positions` dict to `urdf.xacro` macro call |
| Both robots go to the same target          | Old `dragon_points.json` format with single `target` key | Update to use `target_r1` and `target_r2` keys          |
| `value_loss = 88,000` at start             | Reward scale too large for PPO critic                    | Add `VecNormalize` wrapper                              |
| Policy randomizes after 100k steps         | `ent_coef` too high                                      | Reduce to `0.001`                                       |
| Joint violations at 6.9 rad                | No joint limit clamping                                  | Clamp all joint angles before publishing                |
| `ParameterValue` YAML parse error          | Launch file missing import                               | Add `ParameterValue(content, value_type=str)` wrapper   |

---

## Author

**Shishir Chandra Das**
Chittagong University of Engineering & Technology (CUET)
shishirdas726@gmail.com

---

## Acknowledgements

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation
- [Universal Robots ROS2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) — UR5 packages
- [ur_simulation_gazebo](https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation) — Gazebo simulation
