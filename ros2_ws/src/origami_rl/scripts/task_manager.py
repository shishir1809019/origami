"""
task_manager.py  —  FIXED VERSION
===================================
Bug fixes applied:
  1. is_goal_reached() called self.get_current_stage()['target'] but stages
     are dicts of steps, not single targets → KeyError crash          [FIXED]
  2. Added get_total_steps() helper for curriculum training            [NEW]
  3. reset() method added so env.reset() can restart the fold sequence [NEW]
"""

import json
import numpy as np


class TaskManager:
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.stages            = list(self.data.keys())
        self.current_stage_idx = 0
        self.current_step_idx  = 0

    # ── Read helpers ─────────────────────────────────────────────────────────

    def get_current_stage(self) -> dict:
        stage_key = self.stages[self.current_stage_idx]
        return self.data[stage_key]

    def get_current_step_data(self) -> dict:
        stage     = self.get_current_stage()
        step_keys = list(stage.keys())
        return stage[step_keys[self.current_step_idx]]

    def get_total_steps(self) -> int:
        """Total fold steps across all stages."""
        return sum(len(self.data[s]) for s in self.stages)

    def get_completed_steps(self) -> int:
        """How many steps have been finished so far."""
        completed = 0
        for i in range(self.current_stage_idx):
            completed += len(self.data[self.stages[i]])
        completed += self.current_step_idx
        return completed

    # ── Advance ──────────────────────────────────────────────────────────────

    def next_step(self) -> bool:
        """
        Advance to the next step/stage.
        Returns True if there is a next step, False if the dragon is complete.
        """
        stage     = self.get_current_stage()
        step_keys = list(stage.keys())

        if self.current_step_idx < len(step_keys) - 1:
            self.current_step_idx += 1
            return True

        # Move to next stage
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.current_step_idx  = 0
            return True

        return False   # Entire dragon complete

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self):
        """Restart the fold sequence from the beginning."""
        self.current_stage_idx = 0
        self.current_step_idx  = 0

    # ── Goal check  (FIX #1) ─────────────────────────────────────────────────

    def is_goal_reached(
        self,
        r1_pos: np.ndarray,
        r2_pos: np.ndarray,
        threshold: float = 0.08,
    ) -> bool:
        """
        Returns True if at least one robot's EE is within
        `threshold` metres of the current step's target.
        
        FIX: previously called self.get_current_stage()['target'] which
        fails because a stage is a dict-of-steps, not a single target dict.
        Now correctly calls get_current_step_data()['target'].
        """
        step_data = self.get_current_step_data()
        goal      = np.array(step_data["target"], dtype=float)

        dist1 = np.linalg.norm(r1_pos - goal)
        dist2 = np.linalg.norm(r2_pos - goal)
        return dist1 < threshold or dist2 < threshold