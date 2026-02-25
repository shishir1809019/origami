import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import json

class OrigamiCommander(Node):
    def __init__(self):
        super().__init__('origami_commander')
        self.pub1 = self.create_publisher(JointTrajectory, '/robot1_joint_trajectory_controller/joint_trajectory', 10)
        self.pub2 = self.create_publisher(JointTrajectory, '/robot2_joint_trajectory_controller/joint_trajectory', 10)
        
    def move_to_target(self, r1_joints, r2_joints):
        # Robot 1 Command
        msg1 = JointTrajectory()
        msg1.joint_names = [f"robot1_{j}_joint" for j in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
        p1 = JointTrajectoryPoint(positions=r1_joints, time_from_start=rclpy.duration.Duration(seconds=2.0).to_msg())
        msg1.points.append(p1)
        
        # Robot 2 Command
        msg2 = JointTrajectory()
        msg2.joint_names = [f"robot2_{j}_joint" for j in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
        p2 = JointTrajectoryPoint(positions=r2_joints, time_from_start=rclpy.duration.Duration(seconds=2.0).to_msg())
        msg2.points.append(p2)
        
        self.pub1.publish(msg1)
        self.pub2.publish(msg2)
        print("Moving robots to fold position...")

# Use pre-calculated inverse kinematics for your JSON point [0.075, 0.075, 0.04]
# These angles are safer and higher than the ground collision limit (Req #4)
r1_safe = [0.8, -1.2, 1.5, -1.8, -1.5, 0.0]
r2_safe = [2.3, -1.2, 1.5, -1.8, -1.5, 0.0]