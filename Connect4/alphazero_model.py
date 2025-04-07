import pygame
import sys
import numpy as np
import random
import time
import os
from collections import deque
import pickle

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

# ------------------------------------------
# Globals & Colors (from your original code)
# ------------------------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

BLUE   = (0, 0, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE  = (255, 255, 255)

# ------------------------------------------
# Connect4Env (original custom environment)
# ------------------------------------------
class Connect4Env:
    def __init__(self):
        self.rows = ROW_COUNT
        self.cols = COLUMN_COUNT
        self.board = np.zeros((self.rows, self.cols), dtype=int)
    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        return self.board.copy()
    def get_available_actions(self, board):
        return [col for col in range(self.cols) if board[0, col] == 0]
    def step(self, action, player):
        # place piece in the board
        if action not in self.get_available_actions(self.board):
            return self.board.copy(), -10, True, {"error": "Invalid move"}
        for row in range(self.rows-1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                break
        # check win
        if self.check_win(self.board, player):
            return self.board.copy(), 1, True, {"winner": player}
        # check draw
        if len(self.get_available_actions(self.board)) == 0:
            return self.board.copy(), 0, True, {"winner": 0}
        # else ongoing
        return self.board.copy(), -0.01, False, {}
    def check_win(self, board, player):
        # same code from your original to check 4 in a row
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if (board[r,c] == player and
                    board[r,c+1] == player and
                    board[r,c+2] == player and
                    board[r,c+3] == player):
                    return True
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if (board[r,c] == player and
                    board[r+1,c] == player and
                    board[r+2,c] == player and
                    board[r+3,c] == player):
                    return True
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if (board[r,c] == player and
                    board[r+1,c+1] == player and
                    board[r+2,c+2] == player and
                    board[r+3,c+3] == player):
                    return True
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if (board[r,c] == player and
                    board[r-1,c+1] == player and
                    board[r-2,c+2] == player and
                    board[r-3,c+3] == player):
                    return True
        return False

# ------------------------------------------
# Pygame board drawing function (unchanged)
# ------------------------------------------
def draw_board_pygame(board, screen):
    # Simple approach: row=0 at the top
    # We'll use row 'r' as the vertical offset
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            # draw background cell
            pygame.draw.rect(
                screen, BLUE,
                (c * SQUARESIZE, (r+1) * SQUARESIZE, SQUARESIZE, SQUARESIZE)
            )
            pygame.draw.circle(
                screen, BLACK,
                (c*SQUARESIZE + SQUARESIZE//2, (r+1)*SQUARESIZE + SQUARESIZE//2),
                RADIUS
            )
    # now draw the pieces
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            piece = board[r][c]
            if piece == 1:
                color = RED
            elif piece == -1:
                color = YELLOW
            else:
                color = None
            if color:
                pygame.draw.circle(
                    screen, color,
                    (c*SQUARESIZE + SQUARESIZE//2, (r+1)*SQUARESIZE + SQUARESIZE//2),
                    RADIUS
                )
    pygame.display.update()
# ------------------------------------------
# Now: AlphaZero for Connect4
# ------------------------------------------

class AlphaZeroNet(nn.Module):
    """Simple MLP for Connect4, policy + value."""
    def __init__(self, rows=ROW_COUNT, cols=COLUMN_COUNT, hidden_size=128):
        super(AlphaZeroNet, self).__init__()
        self.rows = rows
        self.cols = cols
        self.input_size = rows * cols

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.policy_head = nn.Linear(hidden_size, cols)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch_size, input_size]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))  # in [-1,1]
        return policy_logits, value


class MCTSNode:
    def __init__(self, parent=None, prior_prob=0.0):
        self.parent = parent
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.total_value = 0.0
        self.children = {}

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

class MCTS:
    def __init__(self, net: AlphaZeroNet, n_simulations=50, c_puct=1.0):
        self.net = net
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def search(self, env: Connect4Env, current_player, temp=1.0):
        """Perform MCTS from the current environment state, returning a distribution over columns."""
        root = MCTSNode(parent=None, prior_prob=1.0)

        # Evaluate root
        policy_probs, _ = self._policy_value_fn(env.board, current_player)
        legal_moves = env.get_available_actions(env.board)
        for a in range(COLUMN_COUNT):
            if a in legal_moves:
                child = MCTSNode(parent=root, prior_prob=policy_probs[a])
                root.children[a] = child
            else:
                policy_probs[a] = 0

        # run simulations
        for _ in range(self.n_simulations):
            sim_env = self._copy_env(env)
            self._simulate(root, sim_env, current_player)

        # build distribution from children
        counts = np.array([root.children[a].visit_count if a in root.children else 0
                           for a in range(COLUMN_COUNT)])
        if temp <= 1e-6:
            # pick best
            best_a = np.argmax(counts)
            probs = np.zeros_like(counts, dtype=np.float32)
            probs[best_a] = 1.0
        else:
            counts_exp = counts ** (1.0/temp)
            probs = counts_exp / np.sum(counts_exp)
        return probs

    def _simulate(self, node: MCTSNode, env: Connect4Env, current_player):
        # 1) selection
        while node.children:
            action, node = self._select_child(node)
            # apply action
            env.step(action, current_player)
            # check terminal
            if env.check_win(env.board, current_player):
                break
            if len(env.get_available_actions(env.board)) == 0:
                # draw
                break
            # switch player
            current_player *= -1

        # 2) expansion + evaluation
        if (not env.check_win(env.board, current_player)
           and len(env.get_available_actions(env.board)) > 0):
            policy_probs, value_est = self._policy_value_fn(env.board, current_player)
            legal_moves = env.get_available_actions(env.board)
            for a in range(COLUMN_COUNT):
                if a in legal_moves:
                    child = MCTSNode(parent=node, prior_prob=policy_probs[a])
                    node.children[a] = child
                else:
                    policy_probs[a] = 0
        else:
            # game ended
            if env.check_win(env.board, current_player):
                value_est = 1.0  # from current_player perspective
            else:
                value_est = 0.0  # draw

        # 3) backprop
        self._backprop(node, -value_est)

    def _select_child(self, node: MCTSNode):
        best_score = -float('inf')
        best_action, best_child = None, None
        sum_visits = sum(ch.visit_count for ch in node.children.values())

        for action, child in node.children.items():
            q = child.q_value
            u = self.c_puct * child.prior_prob * np.sqrt(node.visit_count + 1) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backprop(self, node: MCTSNode, value):
        # alternate sign each step up
        current = node
        sign = 1
        while current is not None:
            current.visit_count += 1
            current.total_value += value * sign
            current = current.parent
            sign = -sign

    def _policy_value_fn(self, board, player):
        # Flip board so that "player" sees themselves as +1
        input_board = board.copy() * player
        inp = torch.FloatTensor(input_board.flatten()).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(inp)
            policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return policy, value.item()

    def _copy_env(self, env: Connect4Env):
        import copy
        return copy.deepcopy(env)

def self_play(env: Connect4Env, net: AlphaZeroNet, n_games=5, n_sims=50, temp=1.0):
    """Generate data from self-play."""
    mcts = MCTS(net, n_simulations=n_sims, c_puct=1.0)
    dataset = []
    for _ in range(n_games):
        env.reset()
        game_history = []
        current_player = 1
        done = False
        while not done:
            state = env.board.copy() * current_player
            pi = mcts.search(env, current_player, temp=temp)
            action = np.random.choice(range(COLUMN_COUNT), p=pi)

            game_history.append((state, pi, current_player))

            # step
            new_board, reward, done, info = env.step(action, current_player)
            if done:
                if "winner" in info and info["winner"] != 0:
                    winner = info["winner"]  # 1 or -1
                    for s, p, pl in game_history:
                        outcome = 1.0 if (pl == winner) else -1.0
                        dataset.append((s, p, outcome))
                else:
                    # draw
                    for s, p, pl in game_history:
                        dataset.append((s, p, 0.0))
            current_player *= -1
    return dataset

def train_alphazero(env: Connect4Env, net: AlphaZeroNet,
                    num_iterations=5, games_per_iter=10,
                    batch_size=32, lr=1e-3, n_sims=50):
    """Basic training loop with progress bars via tqdm."""
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for iteration in tqdm(range(num_iterations), desc="Training Iterations"):
        # gather data with self-play
        dataset = self_play(env, net, n_games=games_per_iter, n_sims=n_sims, temp=1.0)
        random.shuffle(dataset)
        states, pis, outcomes = zip(*dataset)

        states_tensor = torch.FloatTensor([s.flatten() for s in states])
        pis_tensor    = torch.FloatTensor(pis)
        outcomes_tensor = torch.FloatTensor(outcomes).unsqueeze(1)

        # Process batches with a progress bar
        for start_idx in tqdm(range(0, len(states_tensor), batch_size), desc="Training Batches", leave=False):
            end_idx = start_idx + batch_size
            batch_states = states_tensor[start_idx:end_idx]
            batch_pis = pis_tensor[start_idx:end_idx]
            batch_outcomes = outcomes_tensor[start_idx:end_idx]

            logits, value = net(batch_states)
            policy = torch.softmax(logits, dim=1)

            policy_loss = -torch.mean(torch.sum(batch_pis * torch.log(policy + 1e-8), dim=1))
            value_loss  = torch.mean((value - batch_outcomes)**2)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tqdm.write(f"Iteration {iteration+1}/{num_iterations} done. Data size = {len(dataset)}")

    return net

# helper: save & load
def save_alphazero_model(net: AlphaZeroNet, filename="alphazero_connect4.pth"):
    torch.save(net.state_dict(), filename)
    print(f"AlphaZero model saved to {filename}")

def load_alphazero_model(net: AlphaZeroNet, filename="alphazero_connect4.pth"):
    if os.path.exists(filename):
        net.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        print(f"Loaded AlphaZero model from {filename}")
    else:
        print(f"No model found at {filename}; starting fresh.")


def play_human_vs_alphazero(env, net, screen, font, n_sims=300):
    """
    Example function to let a human (player=1, red) play vs AlphaZero (player=-1, yellow).
    With hovering on the top row, no flipping, more MCTS sims for stronger AI.
    """
    mcts = MCTS(net, n_simulations=n_sims, c_puct=1.0)

    env.reset()
    current_player = 1
    game_over = False

    while not game_over:
        draw_board_pygame(env.board, screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # -- HOVERING LOGIC --
            if event.type == pygame.MOUSEMOTION:
                # clear the top area
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                x_pos = event.pos[0]
                # draw a preview piece
                if current_player == 1:
                    pygame.draw.circle(
                        screen, RED,
                        (x_pos, SQUARESIZE//2),
                        RADIUS
                    )
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Human turn
                if current_player == 1:
                    x_pos = event.pos[0]
                    col = x_pos // SQUARESIZE
                    available_cols = env.get_available_actions(env.board)
                    if col in available_cols:
                        new_board, reward, done, info = env.step(col, current_player)
                        if done:
                            if "winner" in info and info["winner"] == 1:
                                label = font.render("You win!", 1, RED)
                            elif "winner" in info and info["winner"] == -1:
                                label = font.render("AI wins!", 1, YELLOW)
                            else:
                                label = font.render("Draw!", 1, WHITE)
                            screen.blit(label, (40, 10))
                            pygame.display.update()
                            time.sleep(3)
                            game_over = True
                        current_player *= -1

        # if it's AI's turn and not game_over
        if current_player == -1 and not game_over:
            pygame.time.wait(500)
            pi = mcts.search(env, current_player, temp=0.0)
            action = np.argmax(pi)
            new_board, reward, done, info = env.step(action, current_player)
            if done:
                if "winner" in info and info["winner"] == -1:
                    label = font.render("AI wins!", 1, YELLOW)
                elif "winner" in info and info["winner"] == 1:
                    label = font.render("You win!", 1, RED)
                else:
                    label = font.render("Draw!", 1, WHITE)
                screen.blit(label, (40, 10))
                pygame.display.update()
                time.sleep(3)
                game_over = True
            current_player *= -1

    env.reset()
# ------------------------------------------
# Modify the main() to add an AlphaZero mode
# ------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect 4 – Combined Agents")
    font = pygame.font.SysFont("monospace", 75)

    print("Choose game mode:")
    print("1: Human vs Human")
    print("2: Human vs DQN Agent (with training)")
    print("3: DQN Agent vs DQN Agent (with training)")
    print("4: Continuous Improvement DQN (Self–play training)")
    print("5: Load and Play DQN (Bot vs Bot, no training)")
    print("6: Human vs DQN Agent (no training)")
    print("7: Continuous Improvement Q–Learning (Self–play training)")
    print("8: Human vs Model (Q–Learning or DQN, no training)")
    print("9: OpenSpiel Random Self-Play (console output)")
    print("10: Visual OpenSpiel Play (Human vs Random Agent)")
    print("11: Continuous Training OpenSpiel DQN & Visual Play (Human vs Trained Agent)")
    print("12: AlphaZero Connect4 (Play against trained model)")
    print("13: AlphaZero Connect4 (Extensive training mode)")

    mode = input("Enter mode (1-13): ").strip()
    while mode not in [str(i) for i in range(1, 14)]:
        mode = input("Enter mode (1-13): ").strip()
    mode = int(mode)

    if mode == 12:
        # Mode 12: Load the trained model and let the human play against it.
        env = Connect4Env()
        net = AlphaZeroNet(rows=ROW_COUNT, cols=COLUMN_COUNT, hidden_size=128)
        load_alphazero_model(net, "alphazero_connect4.pth")  # Load trained model if exists
        print("Loaded model. Starting human vs. AlphaZero play.")
        play_human_vs_alphazero(env, net, screen, font, n_sims=50)
        return

    if mode == 13:
        # Mode 13: Extensive training with progress bars.
        env = Connect4Env()
        net = AlphaZeroNet(rows=ROW_COUNT, cols=COLUMN_COUNT, hidden_size=128)
        load_alphazero_model(net, "alphazero_connect4.pth")  # Load existing model (if any)
        print("Starting extensive training. This may take a while...")

        net = train_alphazero(
            env, net,
            num_iterations=1,    # Increase number of iterations
            games_per_iter=50,   # More self-play games per iteration
            n_sims=200,           # Higher number of MCTS simulations
            batch_size=64,
            lr=1e-3
        )
        save_alphazero_model(net, "alphazero_connect4.pth")
        print("Extensive training complete and model saved.")
        return

    # Other modes (1-11) can be handled here as needed.
    print("Other modes not shown here. Exiting.")
    sys.exit()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")
    main()
