import pygame
import sys
import numpy as np
import random
import time
import os
from collections import deque
import pickle

# PyTorch imports for DQN
import torch
import torch.nn as nn
import torch.optim as optim

# Import tqdm for progress bar
from tqdm import tqdm

# ---------------------------
# Global Constants & Colors
# ---------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE  # extra row for drop preview

# Colors (RGB)
BLUE   = (0, 0, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE  = (255, 255, 255)

# Model persistence filenames
DQN_MODEL_FILE = "dqn_connect4_model.pth"
Q_MODEL_FILE   = "connect4_model.pkl"

# ---------------------------
# Connect 4 Environment
# ---------------------------
class Connect4Env:
    def __init__(self):
        self.rows = ROW_COUNT
        self.cols = COLUMN_COUNT
        self.board = np.zeros((self.rows, self.cols), dtype=int)

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        return self.board.copy()

    def get_available_actions(self, board):
        """Return a list of columns (actions) that are not full."""
        return [col for col in range(self.cols) if board[0, col] == 0]

    def step(self, action, player):
        """
        Drop a piece in the given column for the given player.
        Returns (new_board, reward, done, info).

        - Drops the piece into the lowest available row.
        - Returns a reward of +1 for a win,
          0 for a draw,
          and a small penalty (-0.01) for each non–terminal move.
        - Illegal moves yield -10 and end the game.
        """
        if action not in self.get_available_actions(self.board):
            return self.board.copy(), -10, True, {"error": "Invalid move"}
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                break
        if self.check_win(self.board, player):
            return self.board.copy(), 1, True, {"winner": player}
        if len(self.get_available_actions(self.board)) == 0:
            return self.board.copy(), 0, True, {"winner": 0}
        return self.board.copy(), -0.01, False, {}

    def check_win(self, board, player):
        """Return True if the player has four in a row (horizontally, vertically, or diagonally)."""
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r, c+1] == player and board[r, c+2] == player and board[r, c+3] == player:
                    return True
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if board[r, c] == player and board[r+1, c] == player and board[r+2, c] == player and board[r+3, c] == player:
                    return True
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r+1, c+1] == player and board[r+2, c+2] == player and board[r+3, c+3] == player:
                    return True
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r-1, c+1] == player and board[r-2, c+2] == player and board[r-3, c+3] == player:
                    return True
        return False

# ---------------------------
# Canonical State Function
# ---------------------------
def canonical_state(board, player):
    """
    Return a canonical representation of the board from the perspective of the current player.
    If player==1, returns the board unchanged; if player==-1, returns board * -1.
    (Returned as a tuple of tuples.)
    """
    if player == 1:
        canon = board.copy()
    else:
        canon = board * -1
    return tuple(tuple(int(x) for x in row) for row in canon)

# ---------------------------
# Deep Q–Network Architecture (for DQN Agent)
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        """
        A simple fully connected network:
          - input_size: number of inputs (here, 6*7 = 42)
          - output_size: number of actions (7 columns)
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------
# DQN Agent with Experience Replay & Target Network
# ---------------------------
class DQNAgent:
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99995,
                 memory_size=10000, batch_size=64, target_update=1000):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, available_actions):
        """
        Given the current state (a numpy array of shape (6,7)) and available actions,
        select an action using epsilon–greedy exploration.
        """
        self.steps_done += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            available_q = {a: q_values[a] for a in available_actions}
            return max(available_q, key=available_q.get)

    def store_transition(self, state, action, reward, next_state, done, next_avail_actions):
        self.memory.append((state, action, reward, next_state, done, next_avail_actions))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_avail_actions = zip(*transitions)
        states_np = np.array([s.flatten() for s in states])
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_states_np = np.array([s.flatten() for s in next_states])
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values_np = next_q_values.cpu().numpy()
            target_q_list = []
            for i in range(self.batch_size):
                avail = next_avail_actions[i]
                max_q = max([next_q_values_np[i][a] for a in avail]) if avail else 0.0
                target_q_list.append(max_q)
            target_q = torch.FloatTensor(target_q_list).unsqueeze(1).to(self.device)
            expected_q = rewards + (1 - dones) * self.gamma * target_q
        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# ---------------------------
# Q–Learning Agent (Table–based)
# ---------------------------
class QLearningAgentTable:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q = {}  # dictionary for Q-values
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values) if q_values else 0.0
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_avail_actions, done):
        current_q = self.get_q(state, action)
        if done or next_state is None:
            target = reward
        else:
            next_q_values = [self.get_q(next_state, a) for a in next_avail_actions]
            max_next_q = max(next_q_values) if next_q_values else 0.0
            target = reward + self.gamma * max_next_q
        self.q[(state, action)] = current_q + self.alpha * (target - current_q)

# ---------------------------
# Training function for Q–Learning Agent (Table–based)
# ---------------------------
def train_q_agent(episodes=100000, agent=None):
    env = Connect4Env()
    if agent is None:
        agent = QLearningAgentTable(alpha=0.1, gamma=0.9, epsilon=0.1)
    print(f"Training Q-learning agent for {episodes} episodes...")
    for episode in tqdm(range(episodes), desc="Training Q-learning Agent"):
        board = env.reset()
        current_player = 1
        done = False
        while not done:
            state = canonical_state(board, current_player)
            available_actions = env.get_available_actions(board)
            action = agent.choose_action(state, available_actions)
            new_board, reward, done, info = env.step(action, current_player)
            if not done:
                next_state = canonical_state(new_board, -current_player)
                next_avail = env.get_available_actions(new_board)
            else:
                next_state = None
                next_avail = []
            agent.update(state, action, reward, next_state, next_avail, done)
            board = new_board
            current_player *= -1
    print("Q-learning training complete!")
    return agent

def save_model_q(agent, filename=Q_MODEL_FILE):
    with open(filename, "wb") as f:
        pickle.dump(agent.q, f)
    print(f"Q-learning model saved to {filename}")

def load_model_q(agent, filename=Q_MODEL_FILE):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            agent.q = pickle.load(f)
        print(f"Loaded Q-learning model from {filename}")
    else:
        print("No saved Q-learning model found; starting fresh.")

# ---------------------------
# DQN Training Function (Self–play)
# ---------------------------
def train_dqn_agent(episodes=100000, agent=None):
    env = Connect4Env()
    if agent is None:
        agent = DQNAgent(input_size=ROW_COUNT*COLUMN_COUNT, output_size=COLUMN_COUNT)
    print(f"Training DQN agent for {episodes} episodes...")
    for episode in tqdm(range(episodes), desc="Training DQN Agent"):
        board = env.reset()
        current_player = 1
        done = False
        while not done:
            state = np.array(canonical_state(board, current_player), dtype=np.float32)
            available_actions = env.get_available_actions(board)
            action = agent.select_action(state, available_actions)
            new_board, reward, done, info = env.step(action, current_player)
            if not done:
                next_state = np.array(canonical_state(new_board, -current_player), dtype=np.float32)
                next_avail = env.get_available_actions(new_board)
            else:
                next_state = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.float32)
                next_avail = []
            agent.store_transition(state, action, reward, next_state, done, next_avail)
            agent.optimize_model()
            board = new_board
            current_player *= -1
        if (episode + 1) % 5000 == 0:
            print(f"Episode {episode+1}/{episodes} completed.")
    print("DQN training complete!")
    return agent

def save_model_dqn(agent, filename=DQN_MODEL_FILE):
    torch.save(agent.policy_net.state_dict(), filename)
    print(f"DQN model saved to {filename}")

def load_model_dqn(agent, filename=DQN_MODEL_FILE):
    if os.path.exists(filename):
        agent.policy_net.load_state_dict(torch.load(filename, map_location=agent.device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Loaded DQN model from {filename}")
    else:
        print("No saved DQN model found; starting fresh.")

# ---------------------------
# Pygame Drawing Function
# ---------------------------
def draw_board_pygame(board, screen):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == -1:
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

# ---------------------------
# Main Game Loop (Pygame)
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect 4 – Combined Agents")
    font = pygame.font.SysFont("monospace", 75)

    # Menu Options:
    # 1: Human vs Human
    # 2: Human vs DQN Agent (with additional training)
    # 3: DQN Agent vs DQN Agent (with additional training)
    # 4: Continuous Improvement DQN (Self–play training)
    # 5: Load and Play DQN (Bot vs Bot, no training)
    # 6: Human vs DQN Agent (no additional training)
    # 7: Continuous Improvement Q–Learning (Self–play training)
    # 8: Human vs Model (choose between Q–Learning and DQN; no additional training)
    print("Choose game mode:")
    print("1: Human vs Human")
    print("2: Human vs DQN Agent (with training)")
    print("3: DQN Agent vs DQN Agent (with training)")
    print("4: Continuous Improvement DQN (Self–play training)")
    print("5: Load and Play DQN (Bot vs Bot, no training)")
    print("6: Human vs DQN Agent (no training)")
    print("7: Continuous Improvement Q–Learning (Self–play training)")
    print("8: Human vs Model (choose Q–Learning or DQN, no training)")
    mode = input("Enter mode (1-8): ").strip()
    while mode not in [str(i) for i in range(1,9)]:
        mode = input("Enter mode (1-8): ").strip()
    mode = int(mode)

    env = Connect4Env()
    board = env.reset()
    game_over = False
    current_player = 1  # 1 = Red (human when playing) and -1 = Yellow (agent)
    agent = None

    # Modes involving an agent:
    if mode in [2, 3, 4, 5, 6]:
        # Use DQN Agent for these modes
        agent = DQNAgent(
            input_size=ROW_COUNT*COLUMN_COUNT,
            output_size=COLUMN_COUNT,
            lr=0.005,               # higher learning rate
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,       # lower minimum epsilon
            epsilon_decay=0.9995,   # faster decay
            memory_size=50000,      # larger replay buffer
            batch_size=128,         # larger batch size
            target_update=500       # more frequent target updates
        )

        load_model_dqn(agent, DQN_MODEL_FILE)
        if mode == 2:
            agent = train_dqn_agent(episodes=10000, agent=agent)
            save_model_dqn(agent, DQN_MODEL_FILE)
            print("Training complete. You are playing as Red; Agent is Yellow.")
        elif mode == 3:
            agent = train_dqn_agent(episodes=10000, agent=agent)
            save_model_dqn(agent, DQN_MODEL_FILE)
            print("Training complete. Bot vs Bot (DQN Agent vs DQN Agent).")
        elif mode == 4:
            agent = train_dqn_agent(episodes=100000, agent=agent)
            save_model_dqn(agent, DQN_MODEL_FILE)
            print("Continuous improvement training complete (DQN). Bot vs Bot match.")
        elif mode == 5:
            print("Loaded DQN model. No additional training. Bot vs Bot (DQN).")
        elif mode == 6:
            print("Loaded DQN model. No additional training. You are playing as Red; Agent is Yellow.")

    elif mode == 7:
        # Continuous Improvement for Q-Learning agent
        agent = QLearningAgentTable(alpha=0.1, gamma=0.9, epsilon=0.1)
        load_model_q(agent, Q_MODEL_FILE)
        agent = train_q_agent(episodes=1000000, agent=agent)
        save_model_q(agent, Q_MODEL_FILE)
        print("Continuous improvement training complete (Q–Learning). Bot vs Bot match.")
    elif mode == 8:
        # Human vs Model (choose between Q-learning and DQN; no training)
        choice = input("Choose model to play against: 1 for Q–Learning, 2 for DQN: ").strip()
        while choice not in ["1", "2"]:
            choice = input("Enter 1 for Q–Learning or 2 for DQN: ").strip()
        if choice == "1":
            agent = QLearningAgentTable(alpha=0.1, gamma=0.9, epsilon=0.1)
            load_model_q(agent, Q_MODEL_FILE)
            print("Loaded Q–Learning model. No additional training. You are playing as Red; Agent is Yellow.")
        else:
            agent = DQNAgent(input_size=ROW_COUNT*COLUMN_COUNT, output_size=COLUMN_COUNT)
            load_model_dqn(agent, DQN_MODEL_FILE)
            print("Loaded DQN model. No additional training. You are playing as Red; Agent is Yellow.")

    draw_board_pygame(board, screen)

    # Main game loop.
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                # For Human vs Human and for Human vs Model (modes 2 or 6 or 8 when agent is DQN/Q-learning)
                if mode == 1 or (mode in [2,6,8] and current_player == 1):
                    color = RED if current_player == 1 else YELLOW
                    pygame.draw.circle(screen, color, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                if mode == 1 or (mode in [2,6,8] and current_player == 1):
                    posx = event.pos[0]
                    col = int(posx // SQUARESIZE)
                    available_actions = env.get_available_actions(board)
                    if col not in available_actions:
                        print("Invalid move. Try another column.")
                        continue
                    board, reward, game_over, info = env.step(col, current_player)
                    draw_board_pygame(board, screen)
                    if game_over:
                        pygame.time.wait(500)
                        if "winner" in info:
                            if info["winner"] == 1:
                                label = font.render("Red wins!", 1, RED)
                            elif info["winner"] == -1:
                                label = font.render("Yellow wins!", 1, YELLOW)
                            else:
                                label = font.render("Draw!", 1, WHITE)
                            screen.blit(label, (40, 10))
                            pygame.display.update()
                            time.sleep(3)
                        break
                    current_player *= -1
        # Process agent moves for modes where an agent is involved.
        if (mode in [2,6,8] and current_player == -1) or (mode in [3,4,5,7] and not game_over):
            pygame.time.wait(500)
            if mode in [2,6,8]:
                # For human vs model modes (agent is either DQN or Q-learning)
                if isinstance(agent, DQNAgent):
                    state = np.array(canonical_state(board, current_player), dtype=np.float32)
                    available_actions = env.get_available_actions(board)
                    action = agent.select_action(state, available_actions)
                else:
                    state = canonical_state(board, current_player)
                    available_actions = env.get_available_actions(board)
                    action = agent.choose_action(state, available_actions)
            else:
                # For Bot vs Bot modes
                if isinstance(agent, DQNAgent):
                    state = np.array(canonical_state(board, current_player), dtype=np.float32)
                    available_actions = env.get_available_actions(board)
                    action = agent.select_action(state, available_actions)
                else:
                    state = canonical_state(board, current_player)
                    available_actions = env.get_available_actions(board)
                    action = agent.choose_action(state, available_actions)
            print(f"Agent ({'Red' if current_player==1 else 'Yellow'}) chooses column:", action)
            board, reward, game_over, info = env.step(action, current_player)
            draw_board_pygame(board, screen)
            if game_over:
                pygame.time.wait(500)
                if "winner" in info:
                    if info["winner"] == 1:
                        label = font.render("Red wins!", 1, RED)
                    elif info["winner"] == -1:
                        label = font.render("Yellow wins!", 1, YELLOW)
                    else:
                        label = font.render("Draw!", 1, WHITE)
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    time.sleep(3)
                break
            current_player *= -1

if __name__ == "__main__":
    if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
            print("CUDA is not available.")
    main()
