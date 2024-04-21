

import gym
import pandas as pd
import numpy as np
from gym import spaces


class TicTacToeEnv(gym.Env):
    """
    Custom Environment that follows gym interface for playing Tic-Tac-Toe.
    This is a simple board game where the environment is a 3x3 grid.
    """
    def __init__(self, verbose=False):
        """
        Initialize the Tic-Tac-Toe environment.

        Parameters:
        - verbose (bool): If True, print additional output to aid debugging.
        """
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=int)  # 3x3 grid with values {-1, 0, 1}
        self.action_space = spaces.Discrete(9)  # 9 possible actions (0-8) corresponding to the grid positions
        self.player_desc = [' ', 'X', 'O'] # 0( ) 1(X) or -1(O)

        # Initialize the game state
        self.reset()

        self.rewards = {
            'valid_move': 1, #1
            'invalid_move': 0, #-10, #-5,
            'win': 20,
            'lose': -20,
            'draw': -5, #0
        }

        self.verbose = verbose

    def reset(self):
        """ Reset the game """
        self.board = np.zeros((3, 3)) # Initialize empty 3x3 game board (Initial State)
        self.current_player = 1 # Player 1(X) starts the game. 1(X) or -1(O)
        self.winner = None
        self.done = False
        self.invalid_moves = {1: 0, -1:0}
        return self.board

    def step(self, action: int):
        """
        Execute one time step within the environment.

        Parameters:
        - action (int): The action to take.

        Returns:
        - tuple: (observation (np.array), reward (float), done (bool))
        """
        if self.done:
            return self.board, 0, True

        if self.verbose:
            print(f'Action: {action}')
        row = action // 3
        col = action % 3

        if self.board[row, col] != 0:
            # Invalid move, penalize
            self.invalid_moves[self.current_player] += 1
            if self.verbose:
                print('Invalid move')
            return self.board, self.rewards['invalid_move'], False

        # Execute the move
        self.board[row, col] = self.current_player
        reward = self.is_game_over()

        # Switch to the next player
        self.current_player *= -1
        if self.verbose:
            print(f'Current Player: {self.current_player}')

        return self.board, reward, self.done
    
    def update_rewards(self, **kwargs):
        """
        Update the reward values according to provided keyword arguments.

        Parameters:
        - **kwargs: Reward components to update.
        """
        for key, val in kwargs.items():
            self.rewards[key] = val

    def display_reward_function(self):
        """Display the current reward structure."""
        print(pd.DataFrame.from_dict(self.rewards, orient='index', columns=['Reward']))
    
    def is_game_over(self):
        """
        Check if the game is over and update the game state.

        Returns:
        - float: The reward for the action taken.
        """
        # Check for winner
        for player in [1, -1]:
            # Check rows, columns, and diagonals for a win
            if (
                    np.any(np.all(self.board == player, axis=0)) 
                    or np.any(np.all(self.board == player, axis=1)) 
                    or np.all(np.diag(self.board) == player) 
                    or np.all(np.diag(np.fliplr(self.board)) == player)
            ):
                self.winner = player
                if self.verbose:
                    print(f'{self.player_desc[self.winner]} is winner')
                self.done = True

                return self.rewards['win'] if player == 1 else self.rewards['lose']

        # Check for draw
        if np.all(self.board != 0):
            if self.verbose:
                print('Match draw')
            self.done = True
            # No rewards for Match Draw
            return self.rewards['draw']
        
        return self.rewards['valid_move']


    def render(self):
        """Render the board in a human-readable format."""
        for row in self.board:
            print('| ' + ' | '.join([self.player_desc[int(cell)] for cell in row]) + ' |')
