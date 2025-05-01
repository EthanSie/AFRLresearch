#!/usr/bin/env python
"""
DOUBLE-DQN MODEL EVALUATION SCRIPT
Integrates directly with your ADVANCED DOUBLE-DQN TRAINER.
Save your trainer script as `train_double_dqn.py` in the same directory.
"""

import gzip
import torch

# import classes and constants from your training module
from ENDOFAPRIL import RemoteAFSIM, DuelingDQN, ACTION_MAPPING, DEVICE

# Path to your trained model checkpoint
MODEL_PATH   = "double_dqn_YYYYMMDD_HHMMSS.pt.gz"
# Evaluation parameters
NUM_EPISODES = 100
MAX_STEPS    = 200


def load_model(path: str) -> DuelingDQN:
    """
    Load a trained DuelingDQN from a gzip-compressed state dict.
    """
    with gzip.open(path, "rb") as f:
        state_dict = torch.load(f, map_location=DEVICE)
    model = DuelingDQN().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_episode(model: torch.nn.Module) -> float:
    """
    Run one evaluation episode using greedy action selection.
    Returns the cumulative reward.
    """
    env = RemoteAFSIM()
    state = env.state()
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        state_tensor = torch.tensor([state], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            q_vals = model(state_tensor)
        action = int(q_vals.argmax(dim=1).item())

        if not env.step(action):
            break
        total_reward += env.reward()
        state = env.state()
        if env.done():
            break

    return total_reward


def evaluate(model: torch.nn.Module, episodes: int = NUM_EPISODES) -> list[float]:
    """
    Evaluate the model for a given number of episodes, print per-episode rewards,
    and return the list of rewards.
    """
    rewards = []
    for ep in range(1, episodes + 1):
        r = run_episode(model)
        rewards.append(r)
        print(f"Episode {ep}/{episodes}: Reward = {r:.2f}")

    avg_reward = sum(rewards) / len(rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    return rewards


if __name__ == "__main__":
    net = load_model(MODEL_PATH)
    evaluate(net)
