"""
utils.py  —  FIXED VERSION
============================
Improvements:
  1. get_ee_position() includes all 6 joints for correct FK
     (previous version only used joints 0-2 → wrong arm tip position)
  2. calculate_reward() now receives collision flag properly
"""

import numpy as np


# ── UR5 DH parameters ────────────────────────────────────────────────────────
# a,   d,       alpha
UR5_DH = [
    [0,       0.089159,  np.pi / 2],
    [-0.425,  0,         0],
    [-0.39225,0,         0],
    [0,       0.10915,   np.pi / 2],
    [0,       0.09465,  -np.pi / 2],
    [0,       0.0823,    0],
]


def _dh_transform(a, d, alpha, theta) -> np.ndarray:
    """Single DH 4×4 transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,   -st * ca,  st * sa,  a * ct],
        [st,    ct * ca, -ct * sa,  a * st],
        [0,     sa,       ca,       d],
        [0,     0,        0,        1],
    ])


def get_ee_position(joint_angles: np.ndarray, robot_id: int = 1) -> np.ndarray:
    """
    Full UR5 forward kinematics using all 6 joints.
    Returns [x, y, z] of the end-effector in WORLD frame.

    robot_id=1 → base at x = -0.5
    robot_id=2 → base at x = +0.5
    """
    T = np.eye(4)
    for i, (a, d, alpha) in enumerate(UR5_DH):
        theta = float(joint_angles[i])
        T     = T @ _dh_transform(a, d, alpha, theta)

    # Local EE position
    local_pos = T[:3, 3]

    # Apply robot base offset (must match your Gazebo URDF)
    base_x = -0.5 if robot_id == 1 else 0.5
    world_pos = local_pos.copy()
    world_pos[0] += base_x

    return world_pos.astype(np.float32)


def calculate_reward(
    r1_ee: np.ndarray,
    r2_ee: np.ndarray,
    goal_pos: np.ndarray,
    collision_risk: bool,
) -> float:
    """
    Shaped reward:
      • collision → large penalty
      • distance to goal → negative (continuous gradient)
      • arms too close → soft penalty
    """
    if collision_risk:
        return -20.0

    dist1    = float(np.linalg.norm(r1_ee - goal_pos))
    dist2    = float(np.linalg.norm(r2_ee - goal_pos))
    arm_dist = float(np.linalg.norm(r1_ee - r2_ee))

    reward = -(dist1 + dist2)

    # Proximity bonus — ramps up inside 0.3 m
    reward += max(0.0, 0.3 - (dist1 + dist2)) * 20.0

    # Collision avoidance soft penalty
    if arm_dist < 0.12:
        reward -= 10.0 * (1.0 - arm_dist / 0.12)

    return float(reward)