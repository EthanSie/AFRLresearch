import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# 1. Register ALE environments
gym.register_envs(ale_py)

# 2. Load the trained model
model = DQN.load("dqn_breakout")  # loads dqn_breakout.zip

# 3. Create the environment **with** render_mode="human"
env_id = "ALE/Breakout-v5"
play_env = gym.make(env_id, render_mode="human")
play_env = AtariWrapper(
    play_env,
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    terminal_on_life_loss=True,
    clip_reward=False,  # show real scores now
    # scale_obs=False
)

obs, info = play_env.reset()
done = False

while not done:
    # Use the model to predict the best action
    action, _states = model.predict(obs, deterministic=True)

    # Step the environment
    obs, reward, done, truncated, info = play_env.step(action)

    # Render the environment on each step
    play_env.render()

    if done or truncated:
        print("Episode finished!")
        break

play_env.close()
