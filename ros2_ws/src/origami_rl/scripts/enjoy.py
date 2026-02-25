import time
import rclpy
from stable_baselines3 import PPO
from origami_env import MultiUr5OrigamiEnv

def main():
    rclpy.init()
    # 1. Load the Environment
    env = MultiUr5OrigamiEnv()
    
    # 2. Load the Trained Model
    model = PPO.load("dragon_folder_model")
    print("Model Loaded. Starting Origami Folding...")

    obs, info = env.reset() # Reset now returns 2 values
    for i in range(1000):
        # 3. Predict the next move based on the trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # 4. Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # NEW: Allow ROS 2 to process the message queue
        rclpy.spin_once(env.node, timeout_sec=0.01)
        
        # 5. Slow down the loop so we can watch it in Gazebo
        time.sleep(0.5) 
        
        if terminated or truncated:
            obs, info = env.reset()
            print("Sequence Reset: Goal reached or safety violation prevented.")

    rclpy.shutdown()

if __name__ == "__main__":
    main()