import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import imageio
from PIL import Image, ImageDraw, ImageFont

################################################################################
# 1. Action Descriptions
################################################################################
ACTION_DESCRIPTIONS = {
    0: "Move Left",
    1: "Move Down",
    2: "Move Right",
    3: "Move Up"
}

def map_action_to_description(action):
    return ACTION_DESCRIPTIONS.get(action, "Unknown Action")


################################################################################
# 2. Initialize the Environment
#    Note: Using render_mode='rgb_array' so we can grab frames for our GIF/video.
################################################################################
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')


################################################################################
# 3. Initialize and Train the DQN Agent
################################################################################
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

TIMESTEPS = 10000
model.learn(total_timesteps=TIMESTEPS)

# Save trained model
model.save("dqn_frozenlake")


################################################################################
# 4. Evaluate the Agent (Text Output)
################################################################################
def evaluate_agent(env, model, num_episodes=5, max_steps=100):
    """
    Simple function to evaluate the agent with a text-based output,
    printing actions and step results. No GIF/video is saved here.
    """
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
                action = int(action.item())
            else:
                action = int(action)

            action_desc = map_action_to_description(action)
            print(f"Step {step + 1}: {action_desc}")

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1


            if done or truncated:
                if reward == 1:
                    print("Result: Success! Reached the goal.")
                else:
                    print("Result: Failed! Fell into a hole or exceeded time.")

            if step >= max_steps:
                print(f"Result: Exceeded maximum steps ({max_steps}).")
                break
    env.close()

# Uncomment to do a quick text-based evaluation:
# evaluate_agent(env, model, num_episodes=5, max_steps=100)


################################################################################
# 5. Generate a GIF for a Fixed Number of Episodes
################################################################################
def generate_video(env, model, num_episodes=1, max_steps=50, output_filename="frozenlake_run.gif"):
    """
    Runs a few episodes in 'rgb_array' mode, capturing frames and Q-values,
    and saves to an animated GIF.
    """
    frames = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0

        while not (done or truncated) and step < max_steps:
            # 1) Get Q-values for the current obs from the DQN
            obs_tensor = torch.tensor([obs]).long().to(model.device)
            with torch.no_grad():
                q_values = model.q_net(obs_tensor).cpu().numpy().flatten()

            # 2) Choose the best action (deterministic)
            action = np.argmax(q_values)
            action_desc = map_action_to_description(action)

            # 3) Render the environment to get the raw frame (as a NumPy array)
            frame = env.render()

            # 4) Convert the NumPy frame to a PIL image for easy text overlay
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.load_default()

            text_lines = [
                f"Episode: {episode+1}",
                f"Step: {step}",
                f"Action: {action_desc} (#{action})",
                f"Q-values: {np.round(q_values, 3).tolist()}"
            ]

            text_y = 10
            for line in text_lines:
                draw.text((10, text_y), line, fill=(0, 0, 0), font=font)
                text_y += 15

            annotated_frame = np.array(pil_image)
            frames.append(annotated_frame)

            # 5) Step the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

        print(f"Episode {episode+1} ended with reward {total_reward}.")

    # 6) Save frames to a GIF
    imageio.mimsave(output_filename, frames, fps=3)
    print(f"Saved GIF to {output_filename}")


################################################################################
# 6. Generate a GIF Until Success
################################################################################
def generate_video_until_success(
    env, 
    model, 
    max_episodes=50, 
    max_steps=50, 
    output_filename="frozenlake_until_success.gif"
):
    """
    Runs multiple episodes until the agent succeeds at least once (reward == 1),
    capturing frames and Q-values. Saves an animated GIF upon success or
    after max_episodes if no success is found.
    """
    
    frames = []
    success = False

    for episode in range(max_episodes):
        # Reset environment
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0
        
        while not (done or truncated) and step < max_steps:
            # 1) Get Q-values
            obs_tensor = torch.tensor([obs]).long().to(model.device)
            with torch.no_grad():
                q_values = model.q_net(obs_tensor).cpu().numpy().flatten()

            # 2) Choose best action (deterministic)
            action = np.argmax(q_values)
            action_desc = map_action_to_description(action)

            # 3) Render environment as a frame
            frame = env.render()  # returns a NumPy array (rgb_array)
            
            # 4) Annotate the frame
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.load_default()

            text_lines = [
                f"Episode: {episode+1}",
                f"Step: {step}",
                f"Action: {action_desc} (#{action})",
                f"Q-values: {np.round(q_values, 3).tolist()}",
                f"Total Reward (so far): {total_reward}"
            ]
            
            text_y = 10
            for line in text_lines:
                draw.text((10, text_y), line, fill=(0, 0, 0), font=font)
                text_y += 15
            
            annotated_frame = np.array(pil_image)
            frames.append(annotated_frame)

            # 5) Take action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
        
        # Check if success occurred
        if total_reward > 0:
            print(f"SUCCESS on episode {episode+1} with reward {total_reward}!")
            success = True
            break
        else:
            print(f"Episode {episode+1} ended with reward {total_reward} (no success).")

    # Save frames to a GIF
    imageio.mimsave(output_filename, frames, fps=3)
    if success:
        print(f"Saved GIF to {output_filename} - success observed!")
    else:
        print(f"Saved GIF to {output_filename} - no success within {max_episodes} episodes.")



if __name__ == "__main__":
    # Example: Evaluate with text-based debug (uncomment to use)
    # evaluate_agent(env, model, num_episodes=5, max_steps=100)

    # Example: Generate a short GIF for 2 episodes
    generate_video(env, model, num_episodes=2, max_steps=20, output_filename="frozenlake_run.gif")

    # Example: Generate a GIF until success or max_episodes
    generate_video_until_success(
        env, 
        model, 
        max_episodes=200, 
        max_steps=200, 
        output_filename="frozenlake_until_success.gif"
    )
