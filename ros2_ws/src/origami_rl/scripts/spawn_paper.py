import rclpy
from gazebo_msgs.srv import SpawnEntity

def main():
    rclpy.init()
    node = rclpy.create_node('paper_spawner')
    client = node.create_client(SpawnEntity, '/spawn_entity')

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for /spawn_entity...')

    paper_xml = """
    <robot name="origami_paper">
      <link name="link">
        <visual>
          <geometry><box size="0.15 0.15 0.002"/></geometry>
        </visual>
        <collision>
          <geometry><box size="0.15 0.15 0.002"/></geometry>
        </collision>
        <inertial>
          <mass value="0.005"/>
          <inertia ixx="0.0001" ixy="0" ixz="0"
                   iyy="0.0001" iyz="0" izz="0.0001"/>
        </inertial>
      </link>
      <gazebo reference="link">
        <material>Gazebo/White</material>
      </gazebo>
    </robot>
    """

    request = SpawnEntity.Request()
    request.name = "origami_paper"
    request.xml  = paper_xml
    request.initial_pose.position.x = 0.0
    request.initial_pose.position.y = 0.0
    request.initial_pose.position.z = 0.31

    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)

    if future.result() is not None:
        node.get_logger().info('Paper spawned!')
    else:
        node.get_logger().error('Failed to spawn paper!')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()