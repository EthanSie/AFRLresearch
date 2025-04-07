import json
import matplotlib.pyplot as plt

def load_rewards(filename):
    """Load the training rewards from a JSON file."""
    with open(filename, 'r') as f:
        rewards = json.load(f)
    return rewards

def plot_rewards(rewards):
    """Plot training rewards over episodes with enhanced visibility."""
    episodes = range(1, len(rewards) + 1)
    plt.figure(figsize=(14, 7))
    plt.plot(episodes, rewards, marker='o', linestyle='-', linewidth=2, markersize=8, color='blue')
    plt.title('Training Rewards Over Episodes', fontsize=18)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Total Reward', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_reward_distribution(rewards):
    """Plot a histogram of training reward distribution."""
    plt.figure(figsize=(12, 6))
    plt.hist(rewards, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Training Rewards", fontsize=18)
    plt.xlabel("Reward", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def load_q_table(filename):
    """Load the Q–table from a JSON file."""
    with open(filename, 'r') as f:
        q_table = json.load(f)
    return q_table

def print_q_table(q_table):
    """Print the Q–table in a human–readable format."""
    print("=== Q–Table ===")
    for state, actions in q_table.items():
        print(f"State: {state}")
        for action, value in actions.items():
            print(f"   {action}: {value:.2f}")
        print("-" * 30)

if __name__ == "__main__":
    rewards = load_rewards("training_rewards.json")
    # Plot rewards over episodes.
    plot_rewards(rewards)
    # Plot the distribution of rewards.
    plot_reward_distribution(rewards)
    
    q_table = load_q_table("q_table.json")
    print_q_table(q_table)
