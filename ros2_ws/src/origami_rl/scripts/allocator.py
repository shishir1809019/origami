import numpy as np

class TaskAllocator:
    def __init__(self, robot1_base_xyz=[-0.5, 0, 0], robot2_base_xyz=[0.5, 0, 0]):
        self.bases = {
            "robot1": np.array(robot1_base_xyz), 
            "robot2": np.array(robot2_base_xyz)
        }

    def assign_task(self, target_point):
        """
        Assigns the fold to the robot that has the most 
        available 'reach' to that coordinate.
        """
        target = np.array(target_point)
        dist1 = np.linalg.norm(self.bases["robot1"] - target)
        dist2 = np.linalg.norm(self.bases["robot2"] - target)
        
        # If the target is in the middle, both robots work together
        if abs(dist1 - dist2) < 0.1:
            return "dual"
        
        return "robot1" if dist1 < dist2 else "robot2"