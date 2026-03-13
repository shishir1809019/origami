import json
import numpy as np
import os

class TaskManager:
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.steps = []
        for stage in data.values():
            for step in stage.values():
                self.steps.append(step)
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0

    def get_current_step_data(self):
        return self.steps[self.current_idx]

    def get_target_r1(self):
        step = self.steps[self.current_idx]
        if "target_r1" in step:
            return np.array(step["target_r1"], dtype=np.float32)
        return np.array(step["target"], dtype=np.float32)

    def get_target_r2(self):
        step = self.steps[self.current_idx]
        if "target_r2" in step:
            return np.array(step["target_r2"], dtype=np.float32)
        return np.array(step["target"], dtype=np.float32)

    def next_step(self):
        self.current_idx += 1
        return self.current_idx < len(self.steps)

    def get_total_steps(self):
        return len(self.steps)

    def get_completed_steps(self):
        return self.current_idx

    @property
    def current_stage_idx(self):
        return self.current_idx
