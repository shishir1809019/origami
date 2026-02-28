import numpy as np

UR5_DH = [
    [0,        0.089159,  np.pi/2],
    [-0.425,   0,         0      ],
    [-0.39225, 0,         0      ],
    [0,        0.10915,   np.pi/2],
    [0,        0.09465,  -np.pi/2],
    [0,        0.0823,    0      ],
]

def _dh(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ],
    ])

def _fk_local(joint_angles):
    T = np.eye(4)
    for i, (a, d, alpha) in enumerate(UR5_DH):
        T = T @ _dh(a, d, alpha, float(joint_angles[i]))
    return T[:3, 3]

def get_ee_position(joint_angles: np.ndarray, robot_id: int = 1) -> np.ndarray:
    """
    UR5 FK with correct world-frame transform.

    The URDF has a built-in Rz(pi) in base_link-base_link_inertia for BOTH robots.
    Robot2 also has Rz(pi) from world_to_robot2 joint → total Rz(2pi) = no rotation.

    So:
      Robot1: world = local + [-0.25, 0, 0]          (net rotation = 0)
      Robot2: world = Rz(pi)*local + [+0.25, 0, 0]   (net rotation = pi)
    
    Rz(pi) * [x, y, z] = [-x, -y, z]
    """
    p = _fk_local(joint_angles)

    if robot_id == 1:
        # R1 base at x=-0.25, no net rotation
        return np.array([p[0] - 0.25,  p[1],  p[2]], dtype=np.float32)
    else:
        # R2 base at x=+0.25, net Rz(pi) → x flips, y flips
        return np.array([0.25 - p[0], -p[1],  p[2]], dtype=np.float32)


def calculate_reward(r1_ee, r2_ee, goal_pos, collision_risk):
    if collision_risk:
        return -20.0
    dist1    = float(np.linalg.norm(r1_ee - goal_pos))
    dist2    = float(np.linalg.norm(r2_ee - goal_pos))
    arm_dist = float(np.linalg.norm(r1_ee - r2_ee))
    reward   = -(dist1 + dist2)
    reward  += max(0.0, 0.3 - (dist1 + dist2)) * 20.0
    if arm_dist < 0.12:
        reward -= 10.0 * (1.0 - arm_dist / 0.12)
    return float(reward)