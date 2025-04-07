import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

env_id = "ALE/Breakout-v5"

# Training env
train_env = gym.make(env_id)
train_env = AtariWrapper(
    train_env,
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    terminal_on_life_loss=True,
    clip_reward=True,   # Discretizes reward
    # "scale_obs" is typically not needed here
)

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

model.learn(total_timesteps=100000)
train_env.close()

# Playing
play_env = gym.make(env_id, render_mode="human")
play_env = AtariWrapper(
    play_env,
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    terminal_on_life_loss=True,
    clip_reward=False,  # get real scores
)
obs, info = play_env.reset()

done = False
while not done:
    # predict
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = play_env.step(action)
    play_env.render()
    if done or truncated:
        break

play_env.close()
