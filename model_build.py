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



# import gym
# from gym import spaces
# import numpy as np


class IndustrialScenarioEnv(gym.Env):
    def __init__(self, scenario_provider):
        super(IndustrialScenarioEnv, self).__init__()

        self.scenario_provider = scenario_provider
        self.scenario = None
        self.action_space = spaces.Discrete(2)  # Two possible actions: 0 (don't schedule) and 1 (schedule)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)  # Observation space shape: (6,)

    def reset(self, **kwargs):
        self.scenario.clean()
        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action)

        if action == 0:
            self.scenario.clean()
        else:
            self.scenario.submit()

        done = self.scenario is None
        reward = self._calculate_reward()
        observation = self._get_observation()

        return observation, reward, done, {}

    def _calculate_reward(self):
        # Implement your reward calculation logic based on the scenario's outcome
        result = self.scenario.submit()
        return result[0]  # Example: returning the number of detected failures as the reward

    def _get_observation(self):
        # Implement your observation extraction logic based on the current scenario
        metadata = self.scenario.get_ta_metadata()
        return np.array([
            metadata['availAgents'],
            metadata['totalTime'],
            metadata['minExecTime'],
            metadata['maxExecTime'],
            metadata['scheduleDate'],
            metadata['maxDuration']
        ], dtype=np.float32)
        
from create_environment import IndustrialDatasetScenarioProvider
from TestcaseExecutionDataLoader import TestCaseExecutionDataLoader


scenario_provider = IndustrialDatasetScenarioProvider('data/iofrol.csv')
env = IndustrialScenarioEnv(scenario_provider)


model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=20000) 