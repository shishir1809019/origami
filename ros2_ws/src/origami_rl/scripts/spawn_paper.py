import rclpy
from gazebo_msgs.srv import SpawnEntity

def main():
    rclpy.init()
    node = rclpy.create_node('paper_spawner')
    client = node.create_client(SpawnEntity, '/spawn_entity')
    
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Service not available, waiting...')

    # Simple URDF for a 15cm x 15cm white square
    paper_xml = """
    <robot name="origami_paper">
      <link name="link">
        <visual>
          <geometry>
            <box size="0.15 0.15 0.005"/>
          </geometry>
          <material name="white">
            <color rgba="1.0 1.0 1.0 1.0"/>
          </material>
        </visual>
        <collision>
          <geometry>
            <box size="0.15 0.15 0.005"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="0.01"/>
          <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
        </inertial>
      </link>
      <gazebo reference="link">
        <material>Gazebo/White</material>
      </gazebo>
    </robot>
    """

    request = SpawnEntity.Request()
    request.name = "origami_paper"
    request.xml = paper_xml
    request.initial_pose.position.z = 0.02 # Slightly above ground
    
    client.call_async(request)
    node.get_logger().info('Paper spawn requested!')
    rclpy.shutdown()

if __name__ == "__main__":
    main()