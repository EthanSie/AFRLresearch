import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np

# 1. Define Action Descriptions
ACTION_DESCRIPTIONS = {
    0: "Move Left",
    1: "Move Down",
    2: "Move Right",
    3: "Move Up"
}

def map_action_to_description(action):
    return ACTION_DESCRIPTIONS.get(action, "Unknown Action")

# 2. Initialize the Environment
env = gym.make('FrozenLake-v1', is_slippery=False)  # Set is_slippery=False for deterministic environment

# 3. Initialize the DQN Agent
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02
)

# 4. Train the Agent
TIMESTEPS = 10000  # Adjust as needed
model.learn(total_timesteps=TIMESTEPS)

# 5. Save the Trained Model (Optional)
model.save("dqn_frozenlake")
# To load: model = DQN.load("dqn_frozenlake")

# 6. Evaluate the Agent with Move Descriptions
def evaluate_agent(env, model, num_episodes=5, max_steps=100):
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        print(f"\n--- Episode {episode + 1} ---")
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            
            # Ensure action is an integer
            if isinstance(action, np.ndarray):
                action = int(action.item())  # Extract scalar value
            else:
                action = int(action)
            
            action_desc = map_action_to_description(action)
            print(f"Step {step + 1}: {action_desc}")
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            env.render()
            
            if done or truncated:
                if reward == 1:
                    print("Result: Success! Reached the goal.")
                else:
                    print("Result: Failed! Fell into a hole or exceeded time.")
            
            # Optional: Break if max_steps is exceeded
            if step >= max_steps:
                print(f"Result: Exceeded maximum steps ({max_steps}).")
                break
    env.close()

# 7. Run Evaluation
evaluate_agent(env, model, num_episodes=5, max_steps=100)
