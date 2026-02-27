import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class ForceFold(Node):
    def __init__(self):
        super().__init__('force_fold_node')
        # Connects to the controllers defined in your launch file
        self.pub1 = self.create_publisher(JointTrajectory, '/robot1_joint_trajectory_controller/joint_trajectory', 10)
        self.pub2 = self.create_publisher(JointTrajectory, '/robot2_joint_trajectory_controller/joint_trajectory', 10)
        time.sleep(1) # Wait for connection

    def execute_fold(self):
        # Goal: Position end-effectors at Z=0.04 (Above the 0.015 safety floor)
        # These angles move the robots to the center of the paper
        r1_angles = [0.78, -1.2, 1.57, -1.9, -1.57, 0.0] 
        r2_angles = [2.35, -1.2, 1.57, -1.9, -1.57, 0.0]

        msg1 = self.create_trajectory_msg("robot1", r1_angles)
        msg2 = self.create_trajectory_msg("robot2", r2_angles)

        self.pub1.publish(msg1)
        self.pub2.publish(msg2)
        self.get_logger().info('Sending robots to fold position...')

    def create_trajectory_msg(self, prefix, positions):
            msg = JointTrajectory()
            # Ensure these exact strings match your ur_controllers.yaml
            msg.joint_names = [
                f"{prefix}_shoulder_pan_joint",
                f"{prefix}_shoulder_lift_joint",
                f"{prefix}_elbow_joint",
                f"{prefix}_wrist_1_joint",
                f"{prefix}_wrist_2_joint",
                f"{prefix}_wrist_3_joint"
            ]
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in positions]
            point.time_from_start.sec = 2 # Give it 2 seconds to reach the goal
            msg.points.append(point)
            return msg

def main():
    rclpy.init()
    node = ForceFold()
    node.execute_fold()
    rclpy.spin_once(node, timeout_sec=5.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()