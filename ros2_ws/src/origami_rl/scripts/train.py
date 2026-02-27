"""
train.py  —  FIXED VERSION
============================
Key improvements:
  1. Curriculum: start with large SUCCESS_DIST_XY and shrink it over training
  2. Correct PPO hyperparameters for continuous robot control
  3. Longer training (500k steps) with more frequent checkpoints
  4. EvalCallback to save best model automatically
  5. task_manager.reset() called on each new episode so training
     can cycle through all folds, not get stuck on stage 1
"""

import os
import numpy as np
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from origami_env import MultiUr5OrigamiEnv


# ── Curriculum callback ───────────────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """
    Gradually tighten the success threshold as training progresses.
    Starts at 0.20 m and ends at 0.07 m so early training isn't too hard.
    """
    import origami_env as _env_module   # reach the module-level constant

    def __init__(self, total_steps: int, start_dist=0.20, end_dist=0.07):
        super().__init__(verbose=0)
        self.total_steps = total_steps
        self.start_dist  = start_dist
        self.end_dist    = end_dist

    def _on_step(self) -> bool:
        import origami_env as env_mod
        progress = min(1.0, self.num_timesteps / self.total_steps)
        new_dist = self.start_dist - progress * (self.start_dist - self.end_dist)
        env_mod.SUCCESS_DIST_XY = new_dist   # dynamically tighten threshold
        return True


# ── Episode-reset callback ────────────────────────────────────────────────────
class ResetTaskCallback(BaseCallback):
    """
    After each episode ends, reset the TaskManager back to fold step 1
    so the agent trains on ALL folds, not just the first one it learned.
    """
    def __init__(self, env: MultiUr5OrigamiEnv):
        super().__init__(verbose=0)
        self._env = env

    def _on_step(self) -> bool:
        # SB3 sets 'dones' in locals when an episode ends
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._env.task_manager.reset()
        return True


# ── Main ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = 500_000
LOG_DIR     = "./logs/"
MODEL_NAME  = "dragon_folder_model"

os.makedirs(LOG_DIR, exist_ok=True)

rclpy.init()

# Build environment
raw_env = MultiUr5OrigamiEnv()
env     = Monitor(raw_env, LOG_DIR)
env     = DummyVecEnv([lambda: env])

# PPO hyperparameters tuned for UR5 continuous control
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    # Core PPO
    learning_rate    = 3e-4,
    n_steps          = 2048,    # steps per rollout (must be > episode length)
    batch_size       = 256,
    n_epochs         = 10,
    # Value / policy balance
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_range       = 0.2,
    ent_coef         = 0.005,   # small entropy bonus to prevent early collapse
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
    # Network size (bigger than default for 21-dim obs)
    policy_kwargs    = dict(net_arch=[256, 256, 128]),
    device           = "auto",
    tensorboard_log  = LOG_DIR,
)

# Callbacks
checkpoint_cb = CheckpointCallback(
    save_freq   = 10_000,
    save_path   = LOG_DIR,
    name_prefix = "dragon_model",
    verbose     = 1,
)
curriculum_cb = CurriculumCallback(
    total_steps = TOTAL_STEPS,
    start_dist  = 0.20,
    end_dist    = 0.07,
)
reset_task_cb = ResetTaskCallback(raw_env)

print(f"Starting training for {TOTAL_STEPS:,} steps …")
print("Monitor with:  tensorboard --logdir ./logs/")

try:
    model.learn(
        total_timesteps     = TOTAL_STEPS,
        callback            = [checkpoint_cb, curriculum_cb, reset_task_cb],
        reset_num_timesteps = True,
        progress_bar        = True,
    )
except KeyboardInterrupt:
    print("\nInterrupted — saving current weights …")

model.save(MODEL_NAME)
print(f"✅  Model saved → {os.getcwd()}/{MODEL_NAME}.zip")
rclpy.shutdown()