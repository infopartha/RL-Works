# Pong Game Implementation using Deep Q-Networks (DQN)

This repository contains an implementation of the classic Pong game using Deep Q-Networks (DQN), a popular reinforcement learning algorithm. The project is part of a CSCN8020 - Reinforcement Learning course.

## Project Structure

- `pong_utils.py`: Contains utility functions for preprocessing the Pong game frames.
- `pong.ipynb`: Jupyter notebook that implements the Pong game using DQN.


## Project Overview

The Tic-Tac-Toe AI is designed to learn optimal strategies through self-play and interaction with a human-like opponent modeled by another AI. This setup allows the AI to learn both basic and advanced strategies, providing insights into the capabilities and limitations of reinforcement learning in board games.

## Technologies Used

- Python 3.10+
- Gym for standardizing the game environment
- Keras for building DQN models
- NumPy and TensorFlow for handling game state and operations
- Matplotlib for plotting game states
- TensorBoard for monitoring model training

## Setup and Installation

To get the project running locally, follow these steps:

### Prerequisites

Ensure you have Python 3.10 or higher installed on your system. You can download Python from [here](https://www.python.org/downloads/).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/infopartha/RL-Works.git
   cd RL-Works/tic_tac_toe
   ```
2. Setup a python environment
   ```bash
   python -m venv venv
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On MacOS/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```

## pong_utils.py
The `pong_utils.py` script contains the following functions:
- `img_crop(img)`: Crops the input image.
- `downsample(img)`: Downsamples the input image by taking only half of the image resolution.
- `transform_reward(reward)`: Transforms the reward to its sign (-1, 0, or 1).
- `to_grayscale(img)`: Converts the input image to grayscale by averaging the RGB values.
- `normalize_grayscale(img)`: Normalizes the grayscale image from -1 to 1.
- `process_frame(img, image_shape)`: Processes the input frame by cropping, downsampling, converting to grayscale, and normalizing it.

## pong.ipynb
The `pong.ipynb` notebook contains the implementation of the Pong game using DQN. It includes:
- Initialization of the environment and hyperparameters.
- Definition of the neural network architecture for the DQN agent.Preprocessing of the game frames using functions from pong_utils.py.
- Implementation of the DQN algorithm, including the training loop and experience replay.
- Visualization of the training progress and evaluation of the trained agent.

## Contributing

Contributions to the project are welcome! Please feel free to fork the repository, make changes, and submit pull requests. You can also open issues if you find bugs or have feature suggestions.  
Bug Fixes: Submit a pull request.  
Features: Propose new features or improvements in the issues section.  
Documentation: Enhancements to the README or docstrings are appreciated.  


## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](..\LICENSE) file for details.

## Acknowledgments

- Thanks to Prof. Mahmoud Nasr for his guidance and support
- Thanks to [OpenAI Gym](https://gym.openai.com/) for providing the tools to develop and test reinforcement learning algorithms.

![Trained agent playing)(Agent2_learning.gif)
