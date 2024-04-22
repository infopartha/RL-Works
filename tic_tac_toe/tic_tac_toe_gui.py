import argparse
import pygame
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tic_tac_toe_env import *
from tic_tac_toe_agents import *

class TicTacToeGUI:
    def __init__(self, agent=RandomAgent()):
        self.board = np.zeros((3, 3))  # Initialize empty 3x3 game board
        self.current_player = 1  # Player 1 starts the game
        self.player_desc = [' ', 'X', 'O'] # 0 ( ) 1 (X) or -1 (O)
        self.WIDTH = 300
        self.HEIGHT = 300
        self.SQUARE_SIZE = self.WIDTH // 3
        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe")
        self.agent = agent # Create an instance of the random agent
        self.game_over = False  # Flag to indicate if the game is over
        self.winner = None  # Variable to store the winner (1 for agent, -1 for human, 0 for draw)
        # print('Initiated')

    def draw_board(self):
        self.SCREEN.fill((255, 255, 255))  # Fill the screen with white color

        # Draw horizontal lines
        for i in range(1, 3):
            pygame.draw.line(self.SCREEN, (0, 0, 0), (0, i * self.SQUARE_SIZE), (self.WIDTH, i * self.SQUARE_SIZE), 2)

        # Draw vertical lines
        for i in range(1, 3):
            pygame.draw.line(self.SCREEN, (0, 0, 0), (i * self.SQUARE_SIZE, 0), (i * self.SQUARE_SIZE, self.HEIGHT), 2)

        # Draw X and O marks
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 1:
                    pygame.draw.line(self.SCREEN, (0, 0, 0), (col * self.SQUARE_SIZE + 10, row * self.SQUARE_SIZE + 10),
                                     ((col + 1) * self.SQUARE_SIZE - 10, (row + 1) * self.SQUARE_SIZE - 10), 2)
                    pygame.draw.line(self.SCREEN, (0, 0, 0), ((col + 1) * self.SQUARE_SIZE - 10, row * self.SQUARE_SIZE + 10),
                                     (col * self.SQUARE_SIZE + 10, (row + 1) * self.SQUARE_SIZE - 10), 2)
                elif self.board[row, col] == -1:
                    pygame.draw.circle(self.SCREEN, (0, 0, 0),
                                       (col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                                        row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2), self.SQUARE_SIZE // 2 - 10, 2)

        pygame.display.flip()  # Update the display


    def render(self):
        for row in self.board:
            # print('| ' + ' | '.join(['X' if cell == 1 else 'O' if cell == -1 else ' ' for cell in row]) + ' |')
            print('| ' + ' | '.join([self.player_desc[int(cell)] for cell in row]) + ' |')
            # print('-' * 13)

    def get_human_move(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    row = y // self.SQUARE_SIZE
                    col = x // self.SQUARE_SIZE
                    action = row * 3 + col
                    return action

    def check_win(self, player):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def check_draw(self):
        # Check if the board is full (no more empty cells)
        return not np.any(self.board == 0)

    def play(self):
        # print('asdlkfa;sldkfja')
        pygame.init()
        self.draw_board()

        while not self.game_over:

            # print(f'curr player: {self.current_player}')
            if self.current_player == 1:  # Agent's turn
                # print('agents turn')
                agent_action = self.agent.get_action(self.board)
                row = agent_action // 3
                col = agent_action % 3
                if self.board[row, col] == 0:  # Check if the selected cell is empty
                    self.board[row, col] = 1  # Place "X" for agent
                    self.draw_board()
                    # Check for a win or draw
                    if self.check_win(1):
                        self.game_over = True
                        self.winner = 1
                    elif self.check_draw():
                        self.game_over = True
                        self.winner = 0
                    self.current_player = -1  # Switch to human player
            else:  # Human's turn
                human_action = self.get_human_move()
                row = human_action // 3
                col = human_action % 3
                if self.board[row, col] == 0:  # Check if the selected cell is empty
                    self.board[row, col] = -1  # Place "O" for human
                    self.draw_board()
                    # Check for a win or draw
                    if self.check_win(-1):
                        self.game_over = True
                        self.winner = -1
                    elif self.check_draw():
                        self.game_over = True
                        self.winner = 0
                    self.current_player = 1  # Switch to agent
            
            # self.render()
            # print(self.board)

        # Game over, print the result
        winner_text = "Agent wins!" if self.winner == 1 else "Human wins!" if self.winner == -1 else "It's a draw!"
        # messagebox.showinfo("Game Over", winner_text + "\nDo you want to play again?")
        
        play_again = messagebox.askyesno("Game Over", winner_text + "\nDo you want to play again?")
        
        if(play_again):
            self.reset()

    def reset(self):
        self.board = np.zeros((3, 3))  # Reset the game board
        self.game_over = False  # Reset the game over flag
        self.winner = None  # Reset the winner
        self.play()  # Start a new game

# Test the environment
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script.")
    
    # Add required argument for model name
    parser.add_argument('algorithm', type=str, help="The name of the algorithm (q_learn/sarsa/random).")

    # Add optional argument for Q-table pickle file
    parser.add_argument('--pickle_file', type=str, help="Optional Q-table pickle file.", default=None)

    # Parse arguments
    args = parser.parse_args()

    # Access the arguments
    algorithm = args.algorithm
    pickle_file = args.pickle_file

    print(f'{algorithm=}')
    print(f'{pickle_file=}')

    if algorithm == 'q_learn':
        agent = QLearningAgent()
    elif algorithm == 'sarsa':
        agent = SarsaAgent()
    else:
        agent = RandomAgent()
    
    if pickle_file is not None:
        agent.load_q_table(pickle_file)

    gui = TicTacToeGUI(agent)
    gui.play()