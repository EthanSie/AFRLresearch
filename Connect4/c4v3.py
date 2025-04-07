import pygame
import sys
import numpy as np
import random
import time

# ---------------------------
# Global Constants & Colors
# ---------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE  # extra row on top for drop preview

# Define colors (RGB)
BLUE   = (0, 0, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE  = (255, 255, 255)

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
        """Return a list of columns that are not full."""
        return [col for col in range(self.cols) if board[0, col] == 0]

    def step(self, action, player):
        """
        Drop a piece in the given column for the given player.
        Returns (new_board, reward, done, info).
        """
        # Illegal move: column full or out of bounds.
        if action < 0 or action >= self.cols or self.board[0, action] != 0:
            return self.board.copy(), -10, True, {"error": "Invalid move"}

        # Find the lowest available row.
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                break

        # Check for win.
        if self.check_win(self.board, player):
            return self.board.copy(), 1, True, {"winner": player}
        # Check for draw.
        if len(self.get_available_actions(self.board)) == 0:
            return self.board.copy(), 0, True, {"winner": 0}

        return self.board.copy(), 0, False, {}

    def check_win(self, board, player):
        """Return True if the player has four in a row."""
        # Horizontal check.
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r, c+1] == player and \
                   board[r, c+2] == player and board[r, c+3] == player:
                    return True
        # Vertical check.
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if board[r, c] == player and board[r+1, c] == player and \
                   board[r+2, c] == player and board[r+3, c] == player:
                    return True
        # Positive diagonal.
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r+1, c+1] == player and \
                   board[r+2, c+2] == player and board[r+3, c+3] == player:
                    return True
        # Negative diagonal.
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r-1, c+1] == player and \
                   board[r-2, c+2] == player and board[r-3, c+3] == player:
                    return True

        return False

# ---------------------------
# Canonical State Function
# ---------------------------
def canonical_state(board, player):
    """
    Transform the board so that the current player's pieces are +1.
    Returns a tuple of tuples (hashable for use as Q–table keys).
    """
    if player == 1:
        canon = board.copy()
    else:
        canon = board * -1
    return tuple(tuple(int(x) for x in row) for row in canon)

# ---------------------------
# Q–Learning Agent
# ---------------------------
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q = {}  # Q–table with keys (state, action)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        """Return the Q–value for a given (state, action) pair; default is 0.0."""
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        """Epsilon–greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        # Randomly break ties.
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_available_actions, done):
        """
        Update the Q–value using:
            Q(s,a) <- Q(s,a) + α * [reward - γ * max_a' Q(s', a') - Q(s,a)]
        (Future Q is subtracted because of the zero–sum self–play formulation.)
        """
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_q_values = [self.get_q(next_state, a) for a in next_available_actions]
            max_next_q = max(next_q_values) if next_q_values else 0.0
            target = reward - self.gamma * max_next_q
        self.q[(state, action)] = current_q + self.alpha * (target - current_q)

# ---------------------------
# Training Function
# ---------------------------
def train_agent(episodes=100000):
    """
    Train the Q–Learning agent using self–play.
    The default is set to 100,000 episodes for an extreme training regime.
    """
    env = Connect4Env()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)
    print(f"Training agent for {episodes} episodes. This may take a while...")
    for episode in range(episodes):
        board = env.reset()
        current_player = 1  # start with player 1
        done = False
        while not done:
            state = canonical_state(board, current_player)
            available_actions = env.get_available_actions(board)
            action = agent.choose_action(state, available_actions)
            new_board, reward, done, info = env.step(action, current_player)
            if not done:
                next_state = canonical_state(new_board, -current_player)
                next_available_actions = env.get_available_actions(new_board)
            else:
                next_state = None
                next_available_actions = []
            agent.update(state, action, reward, next_state, next_available_actions, done)
            board = new_board
            current_player *= -1  # switch players

        # Optionally, print progress every 5000 episodes.
        if (episode + 1) % 5000 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")
    print("Training complete!")
    return agent

# ---------------------------
# Pygame Drawing Function
# ---------------------------
def draw_board_pygame(board, screen):
    """Draw the Connect 4 board using Pygame."""
    # Draw the board background and holes.
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE,
                             (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARESIZE + SQUARESIZE / 2),
                                int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)),
                               RADIUS)
    # Draw the pieces.
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    height - int(r * SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)
            elif board[r][c] == -1:
                pygame.draw.circle(screen, YELLOW,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    height - int(r * SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)
    pygame.display.update()

# ---------------------------
# Main Game Loop (Pygame)
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect 4 Extreme")
    font = pygame.font.SysFont("monospace", 75)

    print("Choose game mode:")
    print("1: Human vs Human")
    print("2: Human vs Q–Learning Agent")
    print("3: Q–Learning Agent vs Q–Learning Agent")
    mode = input("Enter mode (1, 2, or 3): ").strip()
    while mode not in ["1", "2", "3"]:
        mode = input("Enter mode (1, 2, or 3): ").strip()
    mode = int(mode)

    env = Connect4Env()
    board = env.reset()
    game_over = False
    current_player = 1  # 1 is Red, -1 is Yellow.
    agent = None

    # For modes 2 and 3, we train the agent.
    if mode in [2, 3]:
        agent = train_agent(episodes=100000)  # extreme training episodes
        if mode == 2:
            print("Training complete. You are playing as Red (1). The Agent is Yellow (-1).")
        else:
            print("Training complete. Agent vs Agent mode.")

    draw_board_pygame(board, screen)

    # Main game loop.
    while not game_over:
        # Process human events (only in modes 1 and 2 when it's human's turn).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                # In mode 1, both players are human.
                if mode == 1:
                    color = RED if current_player == 1 else YELLOW
                    pygame.draw.circle(screen, color, (posx, int(SQUARESIZE / 2)), RADIUS)
                # In mode 2, only the human (Red) controls input.
                elif mode == 2 and current_player == 1:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                # For Human vs Human or Human vs Agent (when it's human's turn).
                if mode == 1 or (mode == 2 and current_player == 1):
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

        # Agent move for modes 2 and 3:
        # - Mode 2: Agent plays when it's the agent's (Yellow) turn.
        # - Mode 3: Both players are controlled by the agent.
        if (mode == 2 and current_player == -1 or mode == 3) and not game_over:
            pygame.time.wait(500)  # pause for visual effect
            state = canonical_state(board, current_player)
            available_actions = env.get_available_actions(board)
            col = agent.choose_action(state, available_actions)
            print(f"Agent ({'Red' if current_player == 1 else 'Yellow'}) chooses column:", col)
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

if __name__ == "__main__":
    main()
