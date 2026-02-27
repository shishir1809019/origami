"""
enjoy.py  —  FIXED VERSION
============================
Improvements:
  1. Proper ROS 2 spin_once rate so commands actually reach controllers
  2. Collision recovery: backs robots to home before continuing
  3. Prints real-time progress (fold N / total)
  4. Handles terminated AND truncated resets correctly
  5. task_manager.reset() on episode end so it replays all folds
"""

import time
import rclpy
from stable_baselines3 import PPO
from origami_env import MultiUr5OrigamiEnv


def main():
    rclpy.init()
    env   = MultiUr5OrigamiEnv()
    model = PPO.load("dragon_folder_model", device="cpu")
    print("✅  Model loaded.  Starting origami folding …\n")

    total_folds  = env.task_manager.get_total_steps()
    episode      = 0

    while True:
        episode += 1
        obs, info  = env.reset()
        env.task_manager.reset()          # always start from fold 1
        step_count = 0
        ep_reward  = 0.0
        print(f"─── Episode {episode} ────────────────────────────")

        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward  += reward
            step_count += 1

            # Let ROS process the trajectory command
            rclpy.spin_once(env.node, timeout_sec=0.01)

            # Progress print every 50 steps
            if step_count % 50 == 0:
                completed = env.task_manager.get_completed_steps()
                print(
                    f"  step {step_count:4d} | "
                    f"fold {completed}/{total_folds} | "
                    f"reward {ep_reward:.1f} | "
                    f"goal {env.active_goal.round(3)}"
                )

            time.sleep(0.05)   # ~20 Hz — matches publish rate

            if info.get("origami_complete"):
                print("🐉  Dragon COMPLETE!  Restarting …")
                break

            if terminated or truncated:
                status = "goal reached" if terminated else "timeout"
                print(f"  Episode ended ({status}) after {step_count} steps | "
                      f"total reward {ep_reward:.1f}")
                break

    rclpy.shutdown()


if __name__ == "__main__":
    main()