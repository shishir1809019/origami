import os
from stable_baselines3 import PPO
from origami_env import MultiUr5OrigamiEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Create and wrap the environment
env = MultiUr5OrigamiEnv()
env = Monitor(env)  # Provides logs for episode rewards/lengths
env = DummyVecEnv([lambda: env]) # Vectorize for SB3 compatibility

# Save a checkpoint every 5000 steps (reduced frequency to save disk space)
checkpoint_callback = CheckpointCallback(
    save_freq=5000, 
    save_path='./logs/',
    name_prefix='dragon_model'
)

# Initialize PPO with a slightly lower learning rate for stability in robotics
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

print("Starting Training... Press Ctrl+C to stop early.")

try:
    # 100,000 steps is the 'sweet spot' for UR5 reach tasks
    model.learn(total_timesteps=200000, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current progress...")

# Final Save
model.save("dragon_folder_model")
print(f"Model saved successfully in: {os.getcwd()}/dragon_folder_model.zip")
