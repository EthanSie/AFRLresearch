import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# 1. Register the ALE envs if needed
gym.register_envs(ale_py)

# 2. Create the environment ID
env_id = "ALE/Breakout-v5"

# 3. Create the training environment (usually no window)
train_env = gym.make(env_id)
train_env = AtariWrapper(
    train_env,
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    terminal_on_life_loss=True,
    clip_reward=True,
    # scale_obs=False (default), so we get standard [0..255] images
)

# 4. Create the DQN model
model = DQN(
    "CnnPolicy",
    train_env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=100000,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
)

# 5. Train the model (example: 1e5 steps)
model.learn(total_timesteps=100000)

# 6. Save the model so we can load it later
model.save("dqn_breakout")

train_env.close()
