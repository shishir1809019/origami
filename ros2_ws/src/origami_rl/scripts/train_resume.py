"""
train_resume.py — continue from 120k checkpoint with fixed hyperparameters
"""
import os
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from origami_env import MultiUr5OrigamiEnv

TOTAL_STEPS    = 400_000   # additional steps on top of 120k already done
LOG_DIR        = "./logs_resume/"
MODEL_NAME     = "dragon_folder_model"
CHECKPOINT     = "./logs/dragon_model_120000_steps.zip"

os.makedirs(LOG_DIR, exist_ok=True)


class CurriculumCallback(BaseCallback):
    """Start from where curriculum left off at 120k/500k = 24% progress."""
    def __init__(self, env, total_steps):
        super().__init__(verbose=0)
        self._env   = env
        self._total = total_steps

    def _on_step(self):
        # Resume curriculum from 24% (where 120k checkpoint was)
        base_progress = 0.24
        extra = min(1.0, self.num_timesteps / self._total)
        progress = base_progress + extra * (1.0 - base_progress)
        self._env.success_dist = 0.13 - progress * (0.13 - 0.07)
        return True


class ResetTaskCallback(BaseCallback):
    def __init__(self, env):
        super().__init__(verbose=0)
        self._env = env
    def _on_step(self):
        if any(self.locals.get("dones", [False])):
            self._env.task_manager.reset()
        return True


rclpy.init()
raw_env = MultiUr5OrigamiEnv()
env = Monitor(raw_env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.0, clip_reward=10.0, gamma=0.99)

# Load from 120k checkpoint
print(f"Loading checkpoint: {CHECKPOINT}")
model = PPO.load(
    CHECKPOINT,
    env=env,
    device="cpu",
    # Override hyperparameters — fix the entropy issue
    custom_objects={
        "learning_rate": 1e-4,    # lower lr for fine-tuning
        "ent_coef": 0.001,        # KEY FIX: was 0.01, caused policy to go random
        "clip_range": 0.1,        # tighter clipping for stable fine-tuning
    }
)

print(f"Resuming training for {TOTAL_STEPS:,} more steps")
print(f"Curriculum: {raw_env.success_dist:.3f}m → 0.07m")

try:
    model.learn(
        total_timesteps     = TOTAL_STEPS,
        callback            = [
            CheckpointCallback(10_000, LOG_DIR, "dragon_resume", verbose=1),
            CurriculumCallback(raw_env, TOTAL_STEPS),
            ResetTaskCallback(raw_env),
        ],
        reset_num_timesteps = False,  # continue step count from 120k
        progress_bar        = True,
    )
except KeyboardInterrupt:
    print("\nSaving …")

model.save(MODEL_NAME)
env.save(MODEL_NAME + "_vecnormalize.pkl")
print(f"✅ Saved → {MODEL_NAME}.zip + vecnormalize.pkl")
rclpy.shutdown()