
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from abc import ABC, abstractmethod


ROOT_PATH = os.getcwd() + os.path.sep
MODEL_PATH = ROOT_PATH + 'models' + os.path.sep
RESULTS_PATH = ROOT_PATH + 'results' + os.path.sep
LOG_PATH = ROOT_PATH + 'logs' + os.path.sep

np.random.seed(42)


class RandomAgent:
    """
    A simple agent that selects actions randomly, used as a baseline or for testing.
    """
    def __init__(self, num_actions=9, *args, **kwargs) -> None:
        """
        Initialize the RandomAgent with the number of possible actions.

        Parameters:
        - num_actions (int): The number of actions the agent can choose from.
        """
        self.num_actions = num_actions
        self.algorithm = 'random'

    def get_action(self, *args) -> int:
        """
        Select a random action.

        Returns:
        - int: A randomly selected action index.
        """
        action = np.random.randint(self.num_actions)
        return action

    def display_hyperparameters(self):
        """Display the hyperparameters and configurations of the agent."""
        for k, v in self.__dict__.items():
            if k == 'q_table':
                continue
            print(f'{k}\t: {v}')


class TDAgent(ABC):
    """
    Abstract base class for Temporal Difference learning agents to play Tic Tac Toe.
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon_start=0.1, epsilon_min=0.01, epsilon_decay=0.995, *args, **kwargs) -> None:
        """
        Initialize the agent with learning  and exploration parameters.
        
        Parameters:
        - alpha (float): Learning rate.
        - gamma (float): Discount factor, indicating the importance of future rewards.
        - epsilon_start (float): Initial exploration rate.
        - epsilon_min (float): Minimum exploration rate.
        - epsilon_decay (float): Factor by which the exploration rate decays.
        - *args, **kwargs: Additional arguments allowing for flexible parameter input.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = kwargs.get('epsilon', epsilon_start)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # self.exploration_approach = exploration_approach
        self.num_states = kwargs.get('num_states', 9)
        self.num_actions = kwargs.get('num_actions', 9)
        # self.c = kwargs['c'] if 'c' in kwargs else 0.5 
        self.action_counts = [0] * self.num_actions
        self.q_table = {}
        self.algorithm = None

    def get_valid_random_action(self, state) -> int:
        """Return a valid random action from the current state."""
        state = state.ravel().tolist()
        valid_actions = [index for index, value in enumerate(state) if value == 0]
        return np.random.choice(valid_actions) if valid_actions else None

    def decay_epsilon(self) -> None:
        """Apply exponential decay to the epsilon value based on the decay rate, down to a minimum."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def get_action(self, state) -> int:
        """
        Choose an action based on epsilon-greedy exploration/exploitation strategy.

        Parameters:
        - state (np.array): The current state from which to choose an action.

        Returns:
        - int: The chosen action index.
        """
        state_str = str(state.ravel().tolist())
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.get_valid_random_action(state)
        else:
            if state_str in self.q_table:
                action = np.argmax(self.q_table[state_str])
            else:
                action = self.get_valid_random_action(state)
        return action

    @abstractmethod
    def update_q_table(self, state, action, reward, next_state, next_action=None, *args):
        pass

    def load_q_table(self, pickle_file='tic_tac_toe_q_table.pkl', pickle_folder=None) -> None:
        """
        Load the Q-table from a pickle file.

        Parameters:
        - pickle_file (str): Filename of the pickle file to load.
        - pickle_folder (str, optional): Directory path where the pickle file is stored.
        """
        folder = pickle_folder + os.path.sep if pickle_folder else MODEL_PATH
        with open(folder + pickle_file, 'rb') as fo:
            self.q_table = pickle.load(fo)

    def save_q_table(self, pickle_file='tic_tac_toe_q_table.pkl', pickle_folder=None) -> None:
        """
        Save the Q-table to a pickle file.

        Parameters:
        - pickle_file (str): Filename for the pickle file to save.
        - pickle_folder (str, optional): Directory path where the pickle file should be saved.
        """
        folder = pickle_folder + os.path.sep if pickle_folder else MODEL_PATH
        with open(folder + pickle_file, 'wb') as fo:
            pickle.dump(self.q_table, fo)

    def update_hyperparameters(self, **kwargs) -> None:
        """
        Update agent hyperparameters dynamically.

        Parameters:
        - **kwargs: Keyword arguments for hyperparameters such as 'epsilon', 'alpha', and 'gamma'.
        """
        self.epsilon = kwargs.get('epsilon', self.epsilon)
        self.alpha = kwargs.get('alpha', self.alpha)
        self.gamma = kwargs.get('gamma', self.gamma)

    def display_hyperparameters(self) -> None:
        """Print the current values of hyperparameters."""
        for k, v in self.__dict__.items():
            if k in ('q_table', 'num_states', 'num_actions', 'action_counts', 'num_steps'):
                continue
            print(f'{k}\t: {v}')

    def train(self, env, num_episodes=10000, save_per_episodes=2500, verbose=False, model_name=None, tensorboard_monitoring=True, **kwargs):
        """
        Train the agent using the provided environment through multiple episodes.

        Parameters:
        - env: The environment in which the agent interacts.
        - num_episodes (int): Total number of episodes for training.
        - save_per_episodes (int): Frequency of saving the Q-table during training.
        - verbose (bool): Flag to control the verbosity of the training output.
        - model_name (str, optional): Base name for saved models and logs.
        - tensorboard_monitoring (bool): Flag to enable TensorBoard logging.
        - **kwargs: Additional keyword arguments such as 'opponent'.
        """
        st = datetime.now()
        if not model_name:
            model_name = self.algorithm + '_' + st.strftime("%b%d_%H%M%S")
        else:
            model_name = model_name + '_' + st.strftime("%b%d_%H%M%S")

        if tensorboard_monitoring:
            log_dir = LOG_PATH + model_name
            summary_writer = tf.summary.create_file_writer(log_dir)
        print(f'{st} | Training for {model_name} agent started')

        opponent = kwargs.get('opponent', RandomAgent())

        # rolling_window = kwargs.get('rolling_window', 50)
        # plot_title = kwargs.get('plot_title', model_name + 'training')

        self.num_steps = 0
        starting_player = 1
        metrics, all_rewards = [], []
        p1_wins, p2_wins, draws = 0, 0, 0
        
        for episode in tqdm(range(num_episodes), desc='Episodes'):
            if episode % save_per_episodes == 0 and episode > 0:
                self.save_q_table(f'{model_name}_qtable_{episode}.pkl')
                if verbose:
                    print(f'{datetime.now()} | Episode: {episode} \t\tq-table size: {len(self.q_table)}')

            state = env.reset()
            env.current_player = starting_player
            done = False
            episode_reward = 0

            player = opponent if env.current_player == -1 else self
            action = player.get_action(state)
            step = 0

            while not done:
                step += 1
                if action is None:
                    action = player.get_action(state)
                    continue
                next_state, reward, done = env.step(action)
                player = opponent if env.current_player == -1 else self
                next_action = player.get_action(next_state)

                if env.current_player == 1:
                    episode_reward += reward
                self.update_q_table(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

            # Decay epsilon after every episode
            self.decay_epsilon()
            # Players get alternate chances to start the game
            starting_player *= -1

            if env.winner == 1:
                p1_wins += 1
            elif env.winner == -1:
                p2_wins += 1
            else:
                draws += 1

            all_rewards.append(episode_reward)
            avg_last_5_rewards = np.mean(all_rewards if episode <= 5 else all_rewards[-5:])
            p1_win_perc = p1_wins / (episode + 1)
            p2_win_perc = p2_wins / (episode + 1)
            draw_perc = draws / (episode + 1)
            metrics.append((p1_wins, p2_wins, draws, p1_win_perc, p2_win_perc, draw_perc, episode_reward, avg_last_5_rewards, step))

            if tensorboard_monitoring:
                with summary_writer.as_default():
                    tf.summary.scalar('Total Reward', episode_reward, step=episode)
                    tf.summary.scalar('Average Rewards (Last 5 Episodes)', avg_last_5_rewards, step=episode)
                    tf.summary.scalar('Total Steps', step, step=episode)
                    # tf.summary.scalar('Win Count', p1_wins, step=episode)
                    # tf.summary.scalar('Draw Count', draws, step=episode)
                    tf.summary.scalar('States Explored', len(self.q_table), step=episode)
                    tf.summary.scalar('Win Percentage', p1_win_perc, step=episode)
                    tf.summary.scalar('Loss Percentage', p2_win_perc, step=episode)
                    tf.summary.scalar('Draw Percentage', draw_perc, step=episode)

        et = datetime.now()
        pickle_name = f'{model_name}_qtable.pkl'
        self.save_q_table(pickle_name)
        print(f'{et} | Training for {self.algorithm} agent Completed. Time taken: {et - st}\nQ-Table saved to {pickle_name}')

        df = pd.DataFrame(metrics, columns=['p1_wins', 'p2_wins', 'draws', 'p1_win_perc', 'p2_win_perc', 'draw_perc', 'tot_reward', 'avg_last_5_rewards', 'steps'])
        df.to_csv(f'{RESULTS_PATH}{model_name}_training.csv', index=False)

    def test(self, env, opponent=RandomAgent(), new_rewards=None, num_episodes=2500, model_name=None, plot_results=True, rolling_window=50, plot_title=None):
        """
        Test the agent against an opponent in the environment.

        Parameters:
        - env: The environment in which the agent is tested.
        - opponent: The opponent against whom the agent is tested.
        - new_rewards (dict): The new reward scheme to update env.rewards.
        - num_episodes (int): Number of episodes to test the agent.
        - model_name (str, optional): Base name for saved results.
        - plot_results (bool): Whether to plot the results of the test.
        - rolling_window (int): The rolling window size for averaging the performance metrics.
        - plot_title (str, optional): Title for results plot.
        """
        st = datetime.now()
        if not model_name:
            model_name = self.algorithm + '_' + st.strftime("%b%d_%H%M%S")

        if new_rewards and isinstance(new_rewards, dict):
            env.update_rewards(**new_rewards)
        if not plot_title:
            plot_title = f'{model_name} (p1) vs. Random (p2)'

        print('Hyperparameters')
        self.display_hyperparameters()
        print('Reward Function')
        env.display_reward_function()

        df = calculate_metrics(env, self, opponent, num_episodes=num_episodes)

        if plot_results:
            df[['p1_win_perc', 'p2_win_perc', 'draw_perc']].rolling(window=rolling_window).mean().plot(
                title=plot_title,
                xlabel='No. of Games',
                ylabel='Percentage of Wins and Draws'
            )
            plt.show()

        df.to_csv(f'{RESULTS_PATH}{model_name}_testing.csv', index=False)
        return df


class QLearningAgent(TDAgent):
    """
    A Q-Learning agent that implements the Q-learning algorithm for updating Q-values.
    """
    def __init__(self, alpha=0.1, gamma=0.9, *args, **kwargs) -> None:
        """
        Initialize the QLearningAgent with specific learning parameters.

        Parameters:
        - alpha (float): Learning rate.
        - gamma (float): Discount factor for future rewards.
        - *args, **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(alpha, gamma, *args, **kwargs)
        self.algorithm = 'q_learn'

    def update_q_table(self, state, action, reward, next_state, *args) -> None:
        """
        Update the Q-table based on the state, action, reward, and next state.

        Parameters:
        - state (array): The current state.
        - action (int): The action taken at the current state.
        - reward (float): The reward received after taking the action.
        - next_state (array): The state transitioned to after taking the action.
        - *args: Allows for flexibility in parameters passed to the method.
        """
        state_str = str(state.ravel().tolist())
        next_state_str = str(next_state.ravel().tolist())
        
        # Ensure states exist in the Q-table
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.num_actions)
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(self.num_actions)

        # Q-Learning formula to update Q-value
        current_q_value = self.q_table[state_str][action]
        max_next_q_value = np.max(self.q_table[next_state_str])
        target = reward + self.gamma * max_next_q_value
        self.q_table[state_str][action] += self.alpha * (target - current_q_value)


class SarsaAgent(TDAgent):
    """
    A SARSA agent that implements the SARSA algorithm for updating Q-values.
    """
    def __init__(self, alpha=0.1, gamma=0.9, *args, **kwargs) -> None:
        """
        Initialize the SarsaAgent with specific learning parameters.

        Parameters:
        - alpha (float): Learning rate.
        - gamma (float): Discount factor for future rewards.
        - *args, **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(alpha, gamma, *args, **kwargs)
        self.algorithm = 'sarsa'

    def update_q_table(self, state, action, reward, next_state, next_action=None, *args) -> None:
        """
        Update the Q-table based on the state, action, reward, and the next state and action.

        Parameters:
        - state (array): The current state.
        - action (int): The action taken at the current state.
        - reward (float): The reward received after taking the action.
        - next_state (array): The state transitioned to after taking the action.
        - next_action (int, optional): The action taken at the next state.
        - *args: Allows for flexibility in parameters passed to the method.
        """
        state_str = str(state.ravel().tolist())
        next_state_str = str(next_state.ravel().tolist())
        
        # Ensure states exist in the Q-table
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(9) 
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(9)

        # SARSA formula to update Q-value
        current_q_value = self.q_table[state_str][action]
        next_q_value = self.q_table[next_state_str][next_action] if next_action is not None else 0
        target = reward + self.gamma * next_q_value
        self.q_table[state_str][action] += self.alpha * (target - current_q_value)


def calculate_metrics(env, agent, opponent=None, num_episodes=1000):
    """
    Calculate game outcome metrics over a series of episodes.

    Parameters:
    - env: The game environment.
    - agent: The primary agent playing as player 1.
    - opponent (optional): The opponent agent playing as player 2, defaults to RandomAgent if not specified.
    - num_episodes (int): The number of episodes to run.

    Returns:
    - df (DataFrame): DataFrame containing game outcome statistics.
    """
    if opponent is None:
        opponent = RandomAgent()

    # Initialize counters for wins, losses, draws, total rewards, and steps.
    metrics = []
    p1_wins, p2_wins, draws = 0, 0, 0
    
    # Simulate the games
    for i in range(num_episodes):
        state = env.reset()
        done = False
        tot_reward = 0
        steps = 0

        while not done:
            if env.current_player == 1:
                action = agent.get_action(state)
                state, reward, done = env.step(action)
                tot_reward += reward
            else:
                action = opponent.get_action(state)
                state, reward, done = env.step(action)

            steps += 1

        # Update results based on the winner.
        if env.winner == 1:
            p1_wins += 1
        elif env.winner == -1:
            p2_wins += 1
        else:
            draws += 1

        metrics.append((p1_wins, p2_wins, draws, tot_reward, steps))

    # Create a DataFrame from the metrics
    df = pd.DataFrame(metrics, columns=['p1_wins', 'p2_wins', 'draws', 'tot_reward', 'steps'])
    total_games = df['p1_wins'] + df['p2_wins'] + df['draws']

    # Calculate percentage metrics
    df['p1_win_perc'] = df['p1_wins'] / total_games
    df['p2_win_perc'] = df['p2_wins'] / total_games
    df['draw_perc'] = df['draws'] / total_games

    return df
