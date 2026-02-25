import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from task_manager import TaskManager
from utils import get_ee_position, calculate_reward
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
import os

class MultiUr5OrigamiEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Path to goals
        json_path = os.path.expanduser('~/origami/ros2_ws/src/origami_rl/scripts/dragon_points.json')
        self.task_manager = TaskManager(json_path)
        
        self.current_tasks = self.task_manager.get_current_stage()
        self.active_goal = np.array(self.current_tasks['step_1']['target'], dtype=np.float32)

        # FIXED: State is 12 joints + 3 goal coordinates = 15 total
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        # Actions: 12 joint position deltas
        self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(12,), dtype=np.float32)
        
        # ROS 2 Setup
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('rl_env_node')
        self.pub1 = self.node.create_publisher(JointTrajectory, '/robot1_joint_trajectory_controller/joint_trajectory', 10)
        self.pub2 = self.node.create_publisher(JointTrajectory, '/robot2_joint_trajectory_controller/joint_trajectory', 10)
        
        self.current_joints = np.zeros(12) # [R1_j1...j6, R2_j1...j6]
        # Add to __init__
        self.deleter = self.node.create_client(DeleteEntity, 'delete_entity')
        self.spawner = self.node.create_client(SpawnEntity, 'spawn_entity')

    def _get_obs(self):
        # Observation = current joints + current goal XYZ
        return np.concatenate([self.current_joints, self.active_goal]).astype(np.float32)

    def verify_safety(self, action):
        """ Requirement #4: Verification Algorithm """
        # Predict next state
        next_joints = self.current_joints + action
        
        # Simple distance check between robot bases + reach
        # In a full version, use Forward Kinematics here
        r1_pos = next_joints[:3] # Simplified proxy for position
        r2_pos = next_joints[6:9]
        dist = np.linalg.norm(r1_pos - r2_pos)
        
        return dist > 0.1 # Returns True if safe

    def step(self, action):
        # 1. Safety Check (Req #4)
        next_joints = self.current_joints + action
        if self.check_collision(next_joints):
            # Penalty for hitting the safety floor
            return self._get_obs(), -10.0, False, False, {"status": "Safety Blocked"}

        self.current_joints = next_joints
        self.publish_trajectory(self.current_joints)
        
        # 2. Extract Data (Req #1)
        r1_ee = get_ee_position(self.current_joints[0:6], robot_id=1)
        r2_ee = get_ee_position(self.current_joints[6:12], robot_id=2)
        
        # FIX: Get current data dynamically
        current_step_data = self.task_manager.get_current_step_data()
        target_xyz = np.array(current_step_data['target'])

        # 3. SUCCESS TRIGGER (Req #3)
        # We check XY distance (0.07m per robot) to trigger the fold
        dist_xy_r1 = np.linalg.norm(r1_ee[:2] - target_xyz[:2])
        dist_xy_r2 = np.linalg.norm(r2_ee[:2] - target_xyz[:2])
        total_error_xy = dist_xy_r1 + dist_xy_r2
        
        terminated = False
        # Use a simpler reward that guides them horizontally
        reward = -total_error_xy 

        if dist_xy_r1 < 0.12 and dist_xy_r2 < 0.12:
            print(f"--- FOLD SUCCESS: {current_step_data} ---")
            
            # TRIGGER VISUAL CHANGE (Req #3)
            self.spawn_next_origami_shape(self.task_manager.current_stage_idx)
            self.task_manager.next_step()
            
            # ADVANCE TO NEXT JSON TARGET
            if not self.task_manager.next_step():
                print("--- DRAGON COMPLETE ---")
                terminated = True
            else:
                # Update the goal the AI sees
                new_step = self.task_manager.get_current_step_data()
                self.active_goal = np.array(new_step['target'], dtype=np.float32)
                print(f"Next Goal: {self.active_goal}")

        return self._get_obs(), float(reward), terminated, False, {}

    def move_robots(self, action):
        # This sends the RL output to the ROS 2 controllers
        msg1 = JointTrajectory()
        msg1.joint_names = [f"robot1_{j}" for j in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]
        
        point1 = JointTrajectoryPoint()
        point1.positions = action[0:6].tolist() # First 6 actions for Robot 1
        point1.time_from_start.sec = 0
        point1.time_from_start.nanosec = 500000000 # 0.5 seconds
        msg1.points.append(point1)
        
        self.pub1.publish(msg1)
        # Repeat similar logic for pub2 using action[6:12]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset internal joint tracker to a clean 'Home' state
        # Lean the robots slightly away from each other initially
        self.current_joints = np.array([
            0.0, -1.0, 1.0, 0.0, 0.0, 0.0,  # Robot 1
            3.14, -1.0, 1.0, 0.0, 0.0, 0.0  # Robot 2
        ])
        
        self.publish_trajectory(self.current_joints)
        
        obs = self._get_obs()
        return obs, {}
    
    def publish_trajectory(self, action):
            """
            Takes the 12-value action from RL and sends it to the two UR5 robots.
            """
            # --- Robot 1 ---
            msg1 = JointTrajectory()
            msg1.joint_names = [
                "robot1_shoulder_pan_joint", "robot1_shoulder_lift_joint", 
                "robot1_elbow_joint", "robot1_wrist_1_joint", 
                "robot1_wrist_2_joint", "robot1_wrist_3_joint"
            ]
            
            point1 = JointTrajectoryPoint()
            # Robot 1 uses indices 0 to 6
            point1.positions = action[0:6].astype(float).tolist() 
            point1.velocities = [0.0] * 6
            point1.time_from_start.nanosec = 200000000 # Increased to 0.2s for smoothness
            msg1.points.append(point1)
            self.pub1.publish(msg1)

            # --- Robot 2 ---
            msg2 = JointTrajectory()
            msg2.joint_names = [
                "robot2_shoulder_pan_joint", "robot2_shoulder_lift_joint", 
                "robot2_elbow_joint", "robot2_wrist_1_joint", 
                "robot2_wrist_2_joint", "robot2_wrist_3_joint"
            ]
            
            point2 = JointTrajectoryPoint()
            # FIXED: Robot 2 MUST use indices 6 to 12
            point2.positions = action[6:12].astype(float).tolist()
            point2.velocities = [0.0] * 6
            point2.time_from_start.nanosec = 200000000 # Increased to 0.2s for smoothness
            msg2.points.append(point2)
            self.pub2.publish(msg2)

    def check_collision(self, joints):
        # If joints are all zero, the controller hasn't started yet.
        # Don't block the code, just wait.
        if np.all(joints == 0):
            return False 

        r1_ee = get_ee_position(joints[0:6], 1)
        r2_ee = get_ee_position(joints[6:12], 2)

        # Lower safety floor to 0.005 to prevent the 'Stuck' loop
        if r1_ee[2] < 0.005 or r2_ee[2] < 0.005:
            return True
        return False
    
    def calculate_origami_reward(self, r1_pos, r2_pos):
        # Get the current goal from your JSON
        goal = np.array(self.active_goal)
        
        # Calculate distance from robots to the paper fold point
        dist1 = np.linalg.norm(r1_pos - goal)
        dist2 = np.linalg.norm(r2_pos - goal)
        
        # Negative distance means the reward gets BIGGER as distance gets SMALLER
        reward = -(dist1 + dist2)
        
        # Bonus for being very close (Requirement #1: Motion Extraction)
        if dist1 < 0.02 or dist2 < 0.02:
            reward += 10.0
            
        return reward
    
    def spawn_next_origami_shape(self, stage_idx):
        """Swaps the paper model in Gazebo to show visual progress."""
        # 1. Delete current paper
        # deleter = self.node.create_client(DeleteEntity, 'delete_entity')
        req_del = DeleteEntity.Request()
        req_del.name = "origami_paper"
        self.deleter.call_async(req_del)

        # --- THIS IS WHERE NEXT_SHAPE_XML GOES ---
        # We use stage_idx to make the "box" thicker or different for each stage
        # Requirement #3: Autonomous Operation
        next_shape_xml = f"""
        <robot name="origami_paper">
        <link name="link">
            <visual>
            <geometry>
                <box size="0.15 0.15 {0.01 * (stage_idx + 1)}"/>
            </geometry>
            <material name="white">
                <color rgba="1.0 1.0 1.0 1.0"/>
            </material>
            </visual>
        </link>
        </robot>
        """

        # 3. Spawn the new shape using the variable defined above
        # spawner = self.node.create_client(SpawnEntity, 'spawn_entity')
        req_spawn = SpawnEntity.Request()
        req_spawn.name = "origami_paper"
        req_spawn.xml = next_shape_xml # <--- The variable is used here
        req_spawn.initial_pose.position.z = 0.02
        self.spawner.call_async(req_spawn)
        # This allows the background tasks to finish so the environment doesn't hang
        rclpy.spin_once(self.node, timeout_sec=0.1)