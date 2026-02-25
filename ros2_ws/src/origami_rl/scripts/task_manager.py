import json
import numpy as np

class TaskManager:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.stages = list(self.data.keys())
        self.current_stage_idx = 0
        self.current_step_idx = 0

    def get_current_stage(self):
        stage_key = self.stages[self.current_stage_idx]
        return self.data[stage_key]
    
    def get_current_step_data(self):
        stage = self.get_current_stage()
        step_keys = list(stage.keys())
        return stage[step_keys[self.current_step_idx]]

    def next_step(self):
        stage = self.get_current_stage()
        step_keys = list(stage.keys())

        if self.current_step_idx < len(step_keys) - 1:
            self.current_step_idx += 1
            return True # Successfully moved to next step
        else:
            # Move to next stage
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.current_step_idx = 0
                return True # Successfully moved to next stage
            return False # Entire dragon complete

    def is_goal_reached(self, r1_pos, r2_pos, threshold=0.03):
        """Check if end-effectors are close enough to the target fold."""
        goal = np.array(self.get_current_stage()['target'])
        dist1 = np.linalg.norm(r1_pos - goal)
        dist2 = np.linalg.norm(r2_pos - goal)
        
        # Goal is reached if at least one robot is at the fold point
        return dist1 < threshold or dist2 < threshold