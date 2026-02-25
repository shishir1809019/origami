import numpy as np

def get_ee_position(joint_angles, robot_id=1):
    # Standard UR5 Kinematics
    d1, a2, a3 = 0.089159, -0.425, -0.39225
    theta1, theta2, theta3 = joint_angles[0:3]

    # Local position
    r = a2 * np.cos(theta2) + a3 * np.cos(theta2 + theta3)
    local_x = np.cos(theta1) * r
    local_y = np.sin(theta1) * r
    z = d1 + a2 * np.sin(theta2) + a3 * np.sin(theta2 + theta3)

    # Offset Application
    # IMPORTANT: Ensure your Gazebo world actually has them at +/- 0.5
    if robot_id == 1:
        world_x = local_x - 0.5 
    else:
        world_x = local_x + 0.5 
        
    return np.array([world_x, local_y, z])


# def get_ee_position(joint_angles):
#     """
#     Calculates the (x, y, z) of the UR5 end-effector using Forward Kinematics.
#     """
#     # Link lengths in meters
#     d1 = 0.089159
#     a2 = -0.425
#     a3 = -0.39225
    
#     # We use a simplified planar approximation for the project scope
#     theta1 = joint_angles[0]
#     theta2 = joint_angles[1]
#     theta3 = joint_angles[2]

#     # Calculate 2D reach (r) in the XY plane and height (z)
#     r = a2 * np.cos(theta2) + a3 * np.cos(theta2 + theta3)
#     x = np.cos(theta1) * r
#     y = np.sin(theta1) * r
#     z = d1 + a2 * np.sin(theta2) + a3 * np.sin(theta2 + theta3)
    
#     return np.array([x, y, z])

def calculate_reward(r1_ee, r2_ee, goal_pos, collision_risk):
    # Distance of both robots to the fold point
    dist_to_fold = np.linalg.norm(r1_ee - goal_pos) + np.linalg.norm(r2_ee - goal_pos)
    
    # Requirement #3: Autonomous Cooperation
    # Reward robots for being on opposite sides of the fold point
    dist_between_robots = np.linalg.norm(r1_ee - r2_ee)
    
    # We want them near the goal, but not exactly touching each other (avoiding collision)
    reward = -dist_to_fold 
    
    if dist_between_robots < 0.05:
        reward -= 20.0 # Collision prevention (Requirement #4)
        
    return reward