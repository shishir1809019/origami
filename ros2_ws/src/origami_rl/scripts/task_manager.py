"""
task_manager.py — supports per-robot targets (target_r1 / target_r2)
Falls back to shared 'target' if only one target is defined.
"""
import json
import numpy as np

class TaskManager:
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.stages = json.load(f)

        self.stage_keys = list(self.stages.keys())
        self.current_stage_idx = 0
        self.current_step_idx  = 0
        self._build_flat_steps()

    def _build_flat_steps(self):
        """Flatten all stages/steps into one ordered list."""
        self._steps = []
        for stage_key in self.stage_keys:
            stage = self.stages[stage_key]
            for step_key in sorted(stage.keys()):
                self._steps.append(stage[step_key])
        self._total = len(self._steps)
        self._current = 0

    def get_current_step_data(self) -> dict:
        if self._current >= self._total:
            return self._steps[-1]
        return self._steps[self._current]

    def get_target_r1(self) -> np.ndarray:
        d = self.get_current_step_data()
        if "target_r1" in d:
            return np.array(d["target_r1"], dtype=np.float32)
        return np.array(d["target"], dtype=np.float32)

    def get_target_r2(self) -> np.ndarray:
        d = self.get_current_step_data()
        if "target_r2" in d:
            return np.array(d["target_r2"], dtype=np.float32)
        return np.array(d["target"], dtype=np.float32)

    def next_step(self) -> bool:
        """Advance to next step. Returns True if more steps remain."""
        self._current += 1
        # Update stage index for visual feedback
        completed = self._current
        step_count = 0
        for i, stage_key in enumerate(self.stage_keys):
            stage_steps = len(self.stages[stage_key])
            step_count += stage_steps
            if completed < step_count:
                self.current_stage_idx = i
                break
        return self._current < self._total

    def reset(self):
        self._current = 0
        self.current_stage_idx = 0
        self.current_step_idx  = 0

    def get_total_steps(self) -> int:
        return self._total

    def get_completed_steps(self) -> int:
        return self._current