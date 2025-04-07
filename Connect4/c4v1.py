import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# ---------------------------
# Global game constants
# ---------------------------
ROWS = 6
COLS = 7

# ---------------------------
# Connect 4 Environment
# ---------------------------
class Connect4Env:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.board = np.zeros((self.rows, self.cols), dtype=int)

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        return self.board.copy()

    def get_available_actions(self, board):
        """Return a list of column indices where a piece can be dropped."""
        return [col for col in range(self.cols) if board[0, col] == 0]

    def step(self, action, player):
        """
        Given a column (action) and current player's marker,
        drop the piece in that column (if legal), update the board,
        and return (new_board, reward, done, info).
        """
        # Check for a legal move: if the top cell is not empty, the move is illegal.
        if action < 0 or action >= self.cols or self.board[0, action] != 0:
            return self.board.copy(), -10, True, {"error": "Invalid move"}

        # Find the lowest available row in the column and drop the piece.
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                break

        # Check for a win
        if self.check_win(self.board, player):
            return self.board.copy(), 1, True, {"winner": player}
        # Check for a draw (no available moves)
        if len(self.get_available_actions(self.board)) == 0:
            return self.board.copy(), 0, True, {"winner": 0}
        # Otherwise, continue the game.
        return self.board.copy(), 0, False, {}

    def check_win(self, board, player):
        """Return True if the player has four in a row (horiz, vert, or diag)."""
        # Check horizontal locations for win
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r, c+1] == player \
                   and board[r, c+2] == player and board[r, c+3] == player:
                    return True

        # Check vertical locations for win
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if board[r, c] == player and board[r+1, c] == player \
                   and board[r+2, c] == player and board[r+3, c] == player:
                    return True

        # Check positively sloped diagonals
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r+1, c+1] == player \
                   and board[r+2, c+2] == player and board[r+3, c+3] == player:
                    return True

        # Check negatively sloped diagonals
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r-1, c+1] == player \
                   and board[r-2, c+2] == player and board[r-3, c+3] == player:
                    return True

        return False

# ---------------------------
# Canonical State Function
# ---------------------------
def canonical_state(board, player):
    """
    Transform the board into a canonical form from the perspective of the
    current player. That is, if player == 1, return the board as is;
    if player == -1, return board * -1 so that the current player's pieces
    always appear as +1.
    The returned state is a tuple of tuples (so it can be used as a key).
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
        # Q–table: keys are (state, action) pairs; values are Q–values.
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        """Return the Q–value for a (state, action) pair (default 0.0)."""
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        """Choose an action using epsilon–greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        # Otherwise, choose the action with the highest Q–value.
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        # In case of ties, randomly choose among the best.
        best_actions = [a for a, q_val in zip(available_actions, q_values) if q_val == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_available_actions, done):
        """
        Update the Q–value for the given (state, action) pair using the
        modified update rule for self–play in a zero–sum game:
            Q(s,a) <- Q(s,a) + alpha * (reward - gamma * max_{a'} Q(next_state, a') - Q(s,a))
        When the game is over (done==True) no future term is added.
        """
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_q_values = [self.get_q(next_state, a) for a in next_available_actions]
            max_next_q = max(next_q_values) if next_q_values else 0.0
            # Note the subtraction: the next state's value (for the opponent) counts negatively.
            target = reward - self.gamma * max_next_q
        self.q[(state, action)] = current_q + self.alpha * (target - current_q)

# ---------------------------
# Visualization Function
# ---------------------------
def draw_board(board):
    """
    Draw the Connect 4 board using matplotlib.
    The board is drawn with a blue background and circles in each cell:
      - Red for player 1
      - Yellow for player -1
      - White for empty.
    """
    fig, ax = plt.subplots()
    # Draw the blue board background.
    ax.add_patch(Rectangle((0, 0), COLS, ROWS, color='blue'))
    # Draw circles for each cell.
    for r in range(ROWS):
        for c in range(COLS):
            center = (c + 0.5, ROWS - r - 0.5)  # so that row 0 is at the top
            if board[r, c] == 1:
                color = 'red'
            elif board[r, c] == -1:
                color = 'yellow'
            else:
                color = 'white'
            circle = Circle(center, 0.4, color=color, ec='black')
            ax.add_patch(circle)
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

# ---------------------------
# Training Loop (Self–Play)
# ---------------------------
def train_agent(episodes=10000):
    env = Connect4Env()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)

    for episode in range(episodes):
        board = env.reset()
        current_player = 1  # We start with player 1 (which will see pieces as +1)
        done = False

        while not done:
            # Get the canonical state (from the current player's point of view).
            state = canonical_state(board, current_player)
            available_actions = env.get_available_actions(board)
            action = agent.choose_action(state, available_actions)

            # Take the action in the environment.
            new_board, reward, done, info = env.step(action, current_player)

            # If the game is not over, compute the next state's canonical form
            # from the opponent's perspective.
            if not done:
                next_state = canonical_state(new_board, -current_player)
                next_available_actions = env.get_available_actions(new_board)
            else:
                next_state = None
                next_available_actions = []

            # Update Q–value using our modified update (note the minus sign on the future value)
            agent.update(state, action, reward, next_state, next_available_actions, done)

            board = new_board
            # Switch players: multiply by -1 to flip between 1 and -1.
            current_player *= -1

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")

    return agent

# ---------------------------
# Play a Game (with Visuals)
# ---------------------------
def play_game(agent):
    """
    Play one game of Connect 4 using the trained Q–learning agent.
    The agent will play as red (player 1) and choose moves based on its Q–table.
    The opponent (yellow, player -1) simply selects random legal moves.
    After each move, the board is displayed.
    """
    env = Connect4Env()
    board = env.reset()
    current_player = 1
    done = False

    while not done:
        draw_board(board)
        available_actions = env.get_available_actions(board)
        state = canonical_state(board, current_player)

        if current_player == 1:
            # Agent (red) chooses its move.
            action = agent.choose_action(state, available_actions)
            print("Agent (red) chooses column:", action)
        else:
            # Opponent (yellow) uses a random move.
            action = random.choice(available_actions)
            print("Opponent (yellow) chooses column:", action)

        board, reward, done, info = env.step(action, current_player)
        current_player *= -1

    # Show final board
    draw_board(board)
    if "winner" in info:
        if info["winner"] == 1:
            print("Red (Agent) wins!")
        elif info["winner"] == -1:
            print("Yellow (Opponent) wins!")
        else:
            print("It's a draw!")
    else:
        print("Game over.")

# ---------------------------
# Main: Train and then Play
# ---------------------------
if __name__ == "__main__":
    print("Training agent with self–play...")
    trained_agent = train_agent(episodes=5000)  # try 5000 or more episodes for improved play
    print("Training complete. Now playing a game...")
    play_game(trained_agent)
