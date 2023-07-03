import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import gymnasium as gym

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3 import PPO
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

env = gym.envs.registry.keys()

model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=20000)

model.predict()




class TestCaseEnv(gym.Env):
    def __init__(self, scenario_provider):
        super(TestCaseEnv, self).__init__()
        self.scenario_provider = scenario_provider
        self.current_scenario = None

        # Define the observation space and action space according to your problem
        self.observation_space = gym.spaces.Discrete(2)  # Example observation space with two choices: 0 or 1
        self.action_space = gym.spaces.Discrete(2)  # Example action space with two choices: 0 or 1

    def step(self, action):
        # Implement the logic to process the action and update the environment state
        # Calculate the reward based on the action and the verdict
        verdict = self.current_scenario.solutions[self.current_scenario.testcases[self.current_test_case]['Id']]
        reward = 1 if action == verdict else 0
        self.current_test_case += 1
        done = self.current_test_case >= len(self.current_scenario.testcases)
        return self.current_test_case, reward, done, {}

    def reset(self):
        # Reset the environment to the initial state
        self.current_scenario = self.scenario_provider.get()
        self.current_test_case = 0
        return self.current_test_case

    def render(self, mode='human'):
        pass
        # visualization or rendering logic if needed

    def close(self):
        pass
        # cleanup or shutdown logic if needed




from create_environment import IndustrialDatasetScenarioProvider

scenario_provider = IndustrialDatasetScenarioProvider('data/iofrol.csv')
env = TestCaseEnv(scenario_provider)


model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=20000)
