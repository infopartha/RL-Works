# Tic-Tac-Toe AI

This repository contains the implementation of a Tic-Tac-Toe AI using reinforcement learning algorithms, specifically Q-Learning and SARSA. The project aims to explore the effectiveness of these algorithms in a simple yet strategic game environment.

## Project Overview

The Tic-Tac-Toe AI is designed to learn optimal strategies through self-play and interaction with a human-like opponent modeled by another AI. This setup allows the AI to learn both basic and advanced strategies, providing insights into the capabilities and limitations of reinforcement learning in board games.

## Technologies Used

- Python 3.10+
- Pygame for rendering the game interface
- NumPy for handling game state and operations
- Gym for standardizing the game environment
- Matplotlib for plotting training progress and results
- Pandas for data analysis and manipulation

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

## Contributing

Contributions to the project are welcome! Please feel free to fork the repository, make changes, and submit pull requests. You can also open issues if you find bugs or have feature suggestions.  
Bug Fixes: Submit a pull request.  
Features: Propose new features or improvements in the issues section.  
Documentation: Enhancements to the README or docstrings are appreciated.  


## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](..\LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source community for continuous support and inspiration.
- Special thanks to [OpenAI Gym](https://gym.openai.com/) for providing the tools to develop and test reinforcement learning algorithms.


