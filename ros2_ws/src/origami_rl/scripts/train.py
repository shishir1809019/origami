import os
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from origami_env import MultiUr5OrigamiEnv

TOTAL_STEPS = 500_000
LOG_DIR     = "./logs/"
MODEL_NAME  = "dragon_folder_model"
os.makedirs(LOG_DIR, exist_ok=True)


class CurriculumCallback(BaseCallback):
    """Shrink success threshold from 0.20m → 0.08m as training progresses."""
    def __init__(self, env, total_steps):
        super().__init__(verbose=0)
        self._env   = env
        self._total = total_steps

    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self._total)
        # 0.20m at start → 0.08m at end
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

model = PPO(
    "MlpPolicy", env,
    verbose          = 1,
    learning_rate    = 3e-4,
    n_steps          = 2048,
    batch_size       = 128,
    n_epochs         = 10,
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_range       = 0.2,
    ent_coef         = 0.01,
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
    policy_kwargs    = dict(net_arch=[256, 256, 128]),
    device           = "cpu",
    tensorboard_log  = LOG_DIR,
)

print(f"Starting training — success threshold 0.13m → 0.07m over {TOTAL_STEPS:,} steps")

try:
    model.learn(
        total_timesteps     = TOTAL_STEPS,
        callback            = [
            CheckpointCallback(10_000, LOG_DIR, "dragon_model", verbose=1),
            CurriculumCallback(raw_env, TOTAL_STEPS),
            ResetTaskCallback(raw_env),
        ],
        reset_num_timesteps = True,
        progress_bar        = True,
    )
except KeyboardInterrupt:
    print("\nSaving …")

model.save(MODEL_NAME)
env.save(MODEL_NAME + "_vecnormalize.pkl")
print(f"✅ Saved → {MODEL_NAME}.zip + vecnormalize.pkl")
rclpy.shutdown()