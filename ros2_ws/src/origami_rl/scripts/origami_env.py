import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from task_manager import TaskManager
from utils import get_ee_position
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
import os

MIN_EE_Z     = 0.02
MIN_ARM_DIST = 0.08
SUCCESS_DIST = 0.20   # LARGE threshold — robots must experience success early
MAX_STEPS    = 500    # More steps per episode

R1_JOINTS = [
    "robot1_shoulder_pan_joint", "robot1_shoulder_lift_joint",
    "robot1_elbow_joint",        "robot1_wrist_1_joint",
    "robot1_wrist_2_joint",      "robot1_wrist_3_joint",
]
R2_JOINTS = [
    "robot2_shoulder_pan_joint", "robot2_shoulder_lift_joint",
    "robot2_elbow_joint",        "robot2_wrist_1_joint",
    "robot2_wrist_2_joint",      "robot2_wrist_3_joint",
]

class MultiUr5OrigamiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        json_path = os.path.expanduser(
            "~/origami/ros2_ws/src/origami_rl/scripts/dragon_points.json")
        self.task_manager = TaskManager(json_path)
        self._update_goals()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(12,), dtype=np.float32)

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("rl_env_node")
        self.pub1 = self.node.create_publisher(
            JointTrajectory,
            "/robot1_joint_trajectory_controller/joint_trajectory", 10)
        self.pub2 = self.node.create_publisher(
            JointTrajectory,
            "/robot2_joint_trajectory_controller/joint_trajectory", 10)
        self.deleter = self.node.create_client(DeleteEntity, "delete_entity")
        self.spawner = self.node.create_client(SpawnEntity,  "spawn_entity")
        self._pause   = self.node.create_client(Empty, "/gazebo/pause_physics")
        self._unpause = self.node.create_client(Empty, "/gazebo/unpause_physics")

        self._actual_joints        = np.zeros(12, dtype=np.float32)
        self._joint_state_received = False
        self._js_name_to_idx       = {}

        self.node.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10)

        self._home_joints = np.array([
            0.496, -1.191, -2.707, -1.57, -1.57, 0.0,
            0.496, -1.191, -2.707, -1.57, -1.57, 0.0,
        ], dtype=np.float32)

        self.current_joints = self._home_joints.copy()
        self._step_count    = 0
        self._prev_dist_r1  = None
        self._prev_dist_r2  = None
        self._joint_limits  = [6.28, 6.28, 3.14, 6.28, 6.28, 6.28]

        # Curriculum: start easy, get harder
        self.success_dist = 0.13

    def _joint_state_cb(self, msg: JointState):
        if not self._js_name_to_idx:
            all_joints = R1_JOINTS + R2_JOINTS
            for i, name in enumerate(all_joints):
                if name in msg.name:
                    self._js_name_to_idx[name] = (i, msg.name.index(name))
        for internal_idx, msg_idx in self._js_name_to_idx.values():
            self._actual_joints[internal_idx] = msg.position[msg_idx]
        self._joint_state_received = True

    def _wait_for_joint_states(self, timeout=2.0):
        import time; t0 = time.time()
        while not self._joint_state_received and (time.time()-t0) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _sync_joints(self):
        rclpy.spin_once(self.node, timeout_sec=0.05)
        if self._joint_state_received:
            self.current_joints = self._actual_joints.copy()

    def _update_goals(self):
        self.goal_r1 = self.task_manager.get_target_r1()
        self.goal_r2 = self.task_manager.get_target_r2()

    def _get_ee(self):
        r1 = get_ee_position(self.current_joints[0:6],  robot_id=1)
        r2 = get_ee_position(self.current_joints[6:12], robot_id=2)
        return r1, r2

    def _get_obs(self):
        r1, r2 = self._get_ee()
        return np.concatenate([
            self.current_joints, self.goal_r1, self.goal_r2, r1, r2
        ]).astype(np.float32)

    def _sim(self, pause):
        svc = self._pause if pause else self._unpause
        if svc.service_is_ready():
            svc.call_async(Empty.Request())
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def check_collision(self, joints):
        r1 = get_ee_position(joints[0:6],  robot_id=1)
        r2 = get_ee_position(joints[6:12], robot_id=2)
        if r1[2] < MIN_EE_Z or r2[2] < MIN_EE_Z: return True
        if np.linalg.norm(r1-r2) < MIN_ARM_DIST:  return True
        return False

    def _compute_reward(self, r1, r2):
        d1 = float(np.linalg.norm(r1 - self.goal_r1))
        d2 = float(np.linalg.norm(r2 - self.goal_r2))
        reward = 0.0

        # Strong progress signal
        if self._prev_dist_r1 is not None:
            reward += (self._prev_dist_r1 - d1) * 50.0
        if self._prev_dist_r2 is not None:
            reward += (self._prev_dist_r2 - d2) * 50.0

        self._prev_dist_r1 = d1
        self._prev_dist_r2 = d2

        # Small proximity bonus only very close to goal
        if d1 < 0.15: reward += (0.15 - d1) * 3.0
        if d2 < 0.15: reward += (0.15 - d2) * 3.0

        # Collision penalty
        if np.linalg.norm(r1-r2) < MIN_ARM_DIST * 1.5:
            reward -= 2.0

        return float(reward)

    def step(self, action):
        self._step_count += 1
        next_joints = self.current_joints + action

        for i in range(6):
            next_joints[i]   = float(np.clip(next_joints[i],
                               -self._joint_limits[i], self._joint_limits[i]))
            next_joints[i+6] = float(np.clip(next_joints[i+6],
                               -self._joint_limits[i], self._joint_limits[i]))

        if self.check_collision(next_joints):
            return self._get_obs(), -2.0, False, False, {"status": "collision"}

        self._sim(pause=False)
        self.publish_trajectory(next_joints)
        rclpy.spin_once(self.node, timeout_sec=0.1)
        self._sync_joints()
        self._sim(pause=True)

        r1, r2  = self._get_ee()
        reward  = self._compute_reward(r1, r2)
        d1      = float(np.linalg.norm(r1 - self.goal_r1))
        d2      = float(np.linalg.norm(r2 - self.goal_r2))

        terminated = False
        info       = {}

        if d1 < self.success_dist and d2 < self.success_dist:
            name = self.task_manager.get_current_step_data().get('name','?')
            print(f"✅ FOLD: {name}  d1={d1:.3f} d2={d2:.3f} thresh={self.success_dist:.2f}")
            reward += 10.0
            self.spawn_next_origami_shape(self.task_manager.current_stage_idx)
            if not self.task_manager.next_step():
                print("🐉 DRAGON COMPLETE!")
                terminated = True
                info["origami_complete"] = True
            else:
                self._update_goals()
                self._prev_dist_r1 = None
                self._prev_dist_r2 = None
                self._step_count   = 0

        truncated = self._step_count >= MAX_STEPS
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sim(pause=False)
        self.publish_trajectory(self._home_joints)
        self._wait_for_joint_states()
        self._sync_joints()
        self._sim(pause=True)
        self._update_goals()
        self._prev_dist_r1 = None
        self._prev_dist_r2 = None
        self._step_count   = 0
        return self._get_obs(), {}

    def publish_trajectory(self, joints):
        def _msg(names, positions):
            msg = JointTrajectory()
            msg.joint_names = names
            pt  = JointTrajectoryPoint()
            pt.positions  = [float(v) for v in positions]
            pt.velocities = [0.0] * 6
            pt.time_from_start.nanosec = 100_000_000
            msg.points.append(pt)
            return msg
        self.pub1.publish(_msg(R1_JOINTS, joints[0:6]))
        self.pub2.publish(_msg(R2_JOINTS, joints[6:12]))

    def spawn_next_origami_shape(self, stage_idx):
        req = DeleteEntity.Request()
        req.name = "origami_paper"
        rclpy.spin_until_future_complete(
            self.node, self.deleter.call_async(req), timeout_sec=0.5)
        t   = 0.005 + 0.005*(stage_idx+1)
        xml = f"""<robot name="origami_paper"><link name="link">
          <visual><geometry><box size="0.15 0.15 {t:.4f}"/></geometry></visual>
          <collision><geometry><box size="0.15 0.15 {t:.4f}"/></geometry></collision>
          <inertial><mass value="0.01"/>
          <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
          </inertial></link></robot>"""
        req2 = SpawnEntity.Request()
        req2.name = "origami_paper"
        req2.xml  = xml
        req2.initial_pose.position.z = 0.001
        self.spawner.call_async(req2)
        rclpy.spin_once(self.node, timeout_sec=0.1)