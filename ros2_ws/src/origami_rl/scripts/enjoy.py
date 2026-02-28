"""
enjoy.py — works with VecNormalize
"""
import time, sys
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from origami_env import MultiUr5OrigamiEnv

MODEL_PATH = "dragon_folder_model"
NORM_PATH  = "dragon_folder_model_vecnormalize.pkl"

def main():
    rclpy.init()
    raw_env = MultiUr5OrigamiEnv()
    env     = Monitor(raw_env)
    env     = DummyVecEnv([lambda: env])

    # Load normalizer (must match training)
    import os
    if os.path.exists(NORM_PATH):
        env = VecNormalize.load(NORM_PATH, env)
        env.training    = False   # don't update stats at inference
        env.norm_reward = False   # don't normalize rewards at inference
        print("✅  VecNormalize loaded")
    else:
        print("⚠️   No vecnormalize file — running without normalization")

    try:
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
    except FileNotFoundError:
        print(f"❌  No model at '{MODEL_PATH}.zip' — run train.py first")
        rclpy.shutdown(); sys.exit(1)

    print("✅  Model loaded. Starting …\n")
    total_folds = raw_env.task_manager.get_total_steps()
    episode = 0

    while True:
        episode += 1
        obs = env.reset()
        raw_env.task_manager.reset()
        step_count = 0
        ep_reward  = 0.0
        print(f"─── Episode {episode} ────────────────────────────")

        for _ in range(4000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward  += float(reward[0])
            step_count += 1

            rclpy.spin_once(raw_env.node, timeout_sec=0.01)
            time.sleep(0.05)

            if step_count % 50 == 0:
                completed = raw_env.task_manager.get_completed_steps()
                print(f"  step {step_count:4d} | fold {completed}/{total_folds}"
                      f" | reward {ep_reward:.1f}"
                      f" | R1goal {raw_env.goal_r1.round(3)}"
                      f" | R2goal {raw_env.goal_r2.round(3)}")

            if info[0].get("origami_complete"):
                print("🐉  Dragon COMPLETE!")
                break
            if done[0]:
                print(f"  Episode ended | steps={step_count} | reward={ep_reward:.1f}")
                break

    rclpy.shutdown()

if __name__ == "__main__":
    main()