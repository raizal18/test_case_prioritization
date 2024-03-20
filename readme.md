
python 3.8.0
packages "requirements.txt"

to start experiment run "rl_tcp.py"


results saved in result/

Test Case Prioritization with Reinforcement Learning
This repository contains code for a test case prioritization project using reinforcement learning techniques. The project aims to automate and optimize the process of test case execution order prioritization within continuous integration (CI) cycles, with the goal of maximizing fault detection while minimizing testing time and resources.

Overview
Test case prioritization is crucial for efficient software testing, especially in CI/CD pipelines where rapid feedback is essential. Traditional prioritization methods often rely on heuristics or static analysis, which may not adapt well to changing project dynamics. Reinforcement learning offers a promising approach to learn optimal test case execution order policies through interaction with a dynamic software system.

In this project, we employ a Deep Q-Network (DQN) algorithm to learn the optimal sequencing of test cases within CI cycles. The DQN agent interacts with a custom Gym environment representing the software system under test, utilizing logs of past CI cycle executions to learn effective prioritization strategies.

Repository Structure
graphql

├── create_environment.py        # Script for creating custom Gym environment
├── CIListWiseEnv.py             # Custom Gym environment implementation
├── TestcaseExecutionDataLoader.py # DataLoader for loading test case execution data
├── Config.py                    # Configuration settings
├── cal_met.py                   # Metrics calculation functions
├── data/                        # Directory containing input data
│   ├── iofrol-additional-features.csv
│   ├── gsdtsr-additional-features.csv
│   ├── paintcontrol-additional-features.csv
├── results/                     # Directory to store experiment results
│   ├── exp_results.csv          # Experiment results CSV file
├── README.md                    # This README file
├── requirements.txt             # Python dependencies
├── train_model.py               # Script for training DQN model
├── evaluate_model.py            # Script for evaluating trained model

Getting Started
Clone this repository to your local machine:
bash
git clone https://github.com/yourusername/test-case-prioritization.git

Install the required dependencies:
bash
cd test-case-prioritization
pip install -r requirements.txt

Run the training script to train the DQN model:
bash

python train_model.py
After training, you can evaluate the trained model on different scenarios using the evaluation script:
bash

python evaluate_model.py


Results
Experiment results are stored in the results/ directory. The exp_results.csv file contains metrics such as APFD, APFD_TA, APFDC, NAPFD, and RMSE for each scenario.

Contributing
Contributions to this project are welcome! If you have suggestions for improvements or encounter any issues, feel free to open an issue or submit a pull request.
