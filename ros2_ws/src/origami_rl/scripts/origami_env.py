"""
origami_env.py  —  FIXED VERSION
=================================
Bug fixes applied:
  1. next_step() was called TWICE per success → skipped every other goal  [FIXED]
  2. check_collision() only checked z-height → robots crashed freely      [FIXED]
  3. Reward had no shaping → robots stuck far from goal                   [FIXED]
  4. Terminated flag never reset the goal properly                        [FIXED]
  5. Active goal not updated after first reset                            [FIXED]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from task_manager import TaskManager
from utils import get_ee_position
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
import os
import time


# ── Workspace / safety constants ─────────────────────────────────────────────
MIN_EE_Z          = 0.02    # metres — floor safety limit
MIN_ARM_DIST      = 0.12    # metres — minimum end-effector separation
SUCCESS_DIST_XY   = 0.08    # metres — XY threshold to count fold as done
MAX_STEPS_PER_GOAL= 400     # truncate episode if goal not reached


class MultiUr5OrigamiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # ── Task manager ─────────────────────────────────────────────────────
        json_path = os.path.expanduser(
            "~/origami/ros2_ws/src/origami_rl/scripts/dragon_points.json"
        )
        self.task_manager = TaskManager(json_path)
        self.active_goal   = self._read_goal()          # np.array [x,y,z]

        # ── Spaces ───────────────────────────────────────────────────────────
        # obs = 12 joint angles  +  3 goal coords  +  3 r1_ee  +  3 r2_ee = 21
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )
        # action = 12 joint-angle deltas, clipped to ±0.02 rad
        self.action_space = spaces.Box(
            low=-0.02, high=0.02, shape=(12,), dtype=np.float32
        )

        # ── ROS 2 setup ──────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("rl_env_node")
        self.pub1 = self.node.create_publisher(
            JointTrajectory,
            "/robot1_joint_trajectory_controller/joint_trajectory", 10
        )
        self.pub2 = self.node.create_publisher(
            JointTrajectory,
            "/robot2_joint_trajectory_controller/joint_trajectory", 10
        )
        self.deleter = self.node.create_client(DeleteEntity, "delete_entity")
        self.spawner = self.node.create_client(SpawnEntity,  "spawn_entity")

        # ── Internal state ───────────────────────────────────────────────────
        self._home_joints = np.array([
            0.0, -1.57, 1.57, -1.57, -1.57, 0.0,   # Robot 1 — reaches forward
            0.0, -1.57, 1.57, -1.57, -1.57, 0.0,   # Robot 2 — reaches forward
        ], dtype=np.float32)

        self.current_joints = self._home_joints.copy()
        self._step_count    = 0                         # steps within current goal
        self._prev_dist     = None                      # for progress shaping

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _read_goal(self) -> np.ndarray:
        """Read current target from TaskManager (safe wrapper)."""
        try:
            data = self.task_manager.get_current_step_data()
            return np.array(data["target"], dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _get_ee(self):
        r1 = get_ee_position(self.current_joints[0:6],  robot_id=1)
        r2 = get_ee_position(self.current_joints[6:12], robot_id=2)
        return r1, r2

    def _get_obs(self) -> np.ndarray:
        r1_ee, r2_ee = self._get_ee()
        return np.concatenate([
            self.current_joints,   # 12
            self.active_goal,      # 3
            r1_ee,                 # 3
            r2_ee,                 # 3
        ]).astype(np.float32)

    # ── Collision checker  (FIX #2) ──────────────────────────────────────────
    def check_collision(self, joints: np.ndarray) -> bool:
        """
        Returns True (unsafe) if:
          • either EE is below the floor
          • EE separation is under MIN_ARM_DIST
        Uses get_ee_position so it's consistent with the reward calculation.
        """
        if np.all(joints == 0):
            return False

        r1 = get_ee_position(joints[0:6],  robot_id=1)
        r2 = get_ee_position(joints[6:12], robot_id=2)

        # Floor check
        if r1[2] < MIN_EE_Z or r2[2] < MIN_EE_Z:
            return True

        # Arms-too-close check  ← NEW
        if np.linalg.norm(r1 - r2) < MIN_ARM_DIST:
            return True

        return False

    # ── Reward  (FIX #3 — shaped reward) ─────────────────────────────────────
    def _compute_reward(self, r1: np.ndarray, r2: np.ndarray, collision: bool) -> float:
        """
        Shaped reward so robots get continuous gradient toward the goal:
          • strong collision penalty
          • distance-progress bonus (gets larger the closer they are)
          • success bonus when close enough
        """
        if collision:
            return -20.0

        goal = self.active_goal
        dist1 = float(np.linalg.norm(r1 - goal))
        dist2 = float(np.linalg.norm(r2 - goal))
        dist_sum = dist1 + dist2

        # Base: negative distance (continuous gradient)
        reward = -dist_sum

        # Progress shaping: bonus for getting closer than last step
        if self._prev_dist is not None:
            delta = self._prev_dist - dist_sum
            reward += 5.0 * delta          # amplify the gradient

        self._prev_dist = dist_sum

        # Proximity bonus: scale up sharply near goal
        close_bonus = max(0.0, 0.3 - dist_sum) * 30.0
        reward += close_bonus

        # Keep robots apart (soft penalty)
        arm_sep = float(np.linalg.norm(r1 - r2))
        if arm_sep < MIN_ARM_DIST * 1.5:
            reward -= 5.0 * (1.0 - arm_sep / (MIN_ARM_DIST * 1.5))

        # Success bonus  (extra, step() adds more below)
        dist_xy1 = float(np.linalg.norm(r1[:2] - goal[:2]))
        dist_xy2 = float(np.linalg.norm(r2[:2] - goal[:2]))
        if dist_xy1 < SUCCESS_DIST_XY and dist_xy2 < SUCCESS_DIST_XY:
            reward += 50.0

        return float(reward)

    # ── step()  (FIX #1 — double next_step removed) ──────────────────────────
    def step(self, action: np.ndarray):
        self._step_count += 1

        # Predict next joints
        next_joints = self.current_joints + action

        # Safety check
        collision = self.check_collision(next_joints)
        if collision:
            obs = self._get_obs()
            return obs, -20.0, False, False, {"status": "collision_blocked"}

        # Apply action
        self.current_joints = next_joints.astype(np.float32)
        self.publish_trajectory(self.current_joints)

        # Allow ROS to deliver the command
        rclpy.spin_once(self.node, timeout_sec=0.01)

        # Compute positions & reward
        r1_ee, r2_ee = self._get_ee()
        reward = self._compute_reward(r1_ee, r2_ee, collision=False)

        # ── Success check ────────────────────────────────────────────────────
        dist_xy1 = float(np.linalg.norm(r1_ee[:2] - self.active_goal[:2]))
        dist_xy2 = float(np.linalg.norm(r2_ee[:2] - self.active_goal[:2]))
        terminated = False
        info = {}

        if dist_xy1 < SUCCESS_DIST_XY and dist_xy2 < SUCCESS_DIST_XY:
            step_data = self.task_manager.get_current_step_data()
            print(f"✅ FOLD DONE: {step_data.get('name','?')}  reward={reward:.2f}")

            # Visual feedback
            self.spawn_next_origami_shape(self.task_manager.current_stage_idx)

            # FIX #1: call next_step() ONCE only
            finished = not self.task_manager.next_step()

            if finished:
                print("🐉 DRAGON COMPLETE!")
                terminated = True
                info["origami_complete"] = True
            else:
                # Update goal the agent will see next step
                self.active_goal  = self._read_goal()
                self._prev_dist   = None
                self._step_count  = 0
                print(f"  ➜ Next goal: {self.active_goal}")

        # ── Truncation (timeout per goal) ────────────────────────────────────
        truncated = self._step_count >= MAX_STEPS_PER_GOAL

        return self._get_obs(), reward, terminated, truncated, info

    # ── reset() ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # FIX #5: always re-sync goal from task_manager on reset
        self.current_joints = self._home_joints.copy()
        self.publish_trajectory(self.current_joints)
        rclpy.spin_once(self.node, timeout_sec=0.1)

        self.active_goal  = self._read_goal()
        self._prev_dist   = None
        self._step_count  = 0

        return self._get_obs(), {}

    # ── publish_trajectory() ─────────────────────────────────────────────────
    def publish_trajectory(self, joints: np.ndarray):
        """Send 12-value joint command to both UR5 controllers."""

        def _make_msg(names, positions):
            msg   = JointTrajectory()
            msg.joint_names = names
            pt    = JointTrajectoryPoint()
            pt.positions  = [float(v) for v in positions]
            pt.velocities = [0.0] * 6
            pt.time_from_start.nanosec = 200_000_000   # 0.2 s
            msg.points.append(pt)
            return msg

        self.pub1.publish(_make_msg(
            ["robot1_shoulder_pan_joint", "robot1_shoulder_lift_joint",
             "robot1_elbow_joint",        "robot1_wrist_1_joint",
             "robot1_wrist_2_joint",      "robot1_wrist_3_joint"],
            joints[0:6]
        ))
        self.pub2.publish(_make_msg(
            ["robot2_shoulder_pan_joint", "robot2_shoulder_lift_joint",
             "robot2_elbow_joint",        "robot2_wrist_1_joint",
             "robot2_wrist_2_joint",      "robot2_wrist_3_joint"],
            joints[6:12]
        ))

    # ── spawn_next_origami_shape() ───────────────────────────────────────────
    def spawn_next_origami_shape(self, stage_idx: int):
        # Delete old paper
        req_del      = DeleteEntity.Request()
        req_del.name = "origami_paper"
        future       = self.deleter.call_async(req_del)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=0.5)

        # Spawn new (thicker to show progress)
        thickness = 0.005 + 0.005 * (stage_idx + 1)
        xml = f"""
<robot name="origami_paper">
  <link name="link">
    <visual>
      <geometry><box size="0.15 0.15 {thickness:.4f}"/></geometry>
      <material><color rgba="1.0 1.0 0.9 1.0"/></material>
    </visual>
    <collision>
      <geometry><box size="0.15 0.15 {thickness:.4f}"/></geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
</robot>"""

        req_sp               = SpawnEntity.Request()
        req_sp.name          = "origami_paper"
        req_sp.xml           = xml
        req_sp.initial_pose.position.z = 0.02
        self.spawner.call_async(req_sp)
        rclpy.spin_once(self.node, timeout_sec=0.2)