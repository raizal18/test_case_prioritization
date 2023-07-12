import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import pandas as pd
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

from CIListWiseEnv import CIListWiseEnv


test_data_loader = TestCaseExecutionDataLoader("data/iofrol-additional-features.csv", "simple")
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()

from Config import Config

conf = Config()

conf.win_size = 3

env = CIListWiseEnv(ci_cycle_logs[1], conf)

# scenario_provider = IndustrialDatasetScenarioProvider('data/iofrol.csv')
# env = IndustrialScenarioEnv(scenario_provider)


model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=40000) 


rmse = env.get_rmse()

from cal_met import calculate_apfd, calculate_apfd_ta, calculate_apfdc, calculate_napfd


## 
test_data_loader = TestCaseExecutionDataLoader("data\gsdtsr-additional-feature.csv", "simple")
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()

from Config import Config

conf = Config()

conf.win_size = 3

env1 = CIListWiseEnv(ci_cycle_logs[1], conf)

##
test_data_loader = TestCaseExecutionDataLoader("data/paintcontrol-additional-features.csv", "simple")
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()

from Config import Config

conf = Config()

conf.win_size = 3

env2 = CIListWiseEnv(ci_cycle_logs[1], conf)


rmse1 = env1.get_rmse()
rmse2 = env2.get_rmse()



def calc(model, env):
    model_predictions = []
    actual_ = []
    n_episodes = 5
    obs = env.reset()
    done = False
    model_prediction_episode = []
    i = 0
    for i in range(0,500):
        try:
            action, _ = model.predict(env.step(i)[0])  # Get the action from the model
            obs, reward, done, _ = env.step(action)
            actual_.append(action)
            model_prediction_episode.append(action)
        except TypeError:
            break

    actualorder = [int(x) for x in actual_]
    predictions = [int(x) for x in model_prediction_episode]
    apfd = calculate_apfd(actualorder, predictions)

    apfdc = calculate_apfdc(apfd,env.cost)

    napfd = calculate_napfd(apfd, len(actual_))

    apfd_ta = calculate_apfd_ta(actualorder, predictions, [1 for i in actual_])
    return apfd, apfdc, napfd, apfd_ta ,apfd/env.cost

apfd, apfdc, napfd, apfd_ta, apfda = calc(model, env)
apfd1, apfdc1, napfd1, apfd_ta1, apfda1 = calc(model, env1)
apfd2, apfdc2, napfd2, apfd_ta2, apfda2 = calc(model, env2)

form = lambda x : format(x,'0.04f')

print(f"APFD iofrol    : {form(apfd)} gsdtsr : {form(apfd1)} paintcontrol :{form(apfd2)} ")
print(f"APFD_TA iofrol : {form(apfd_ta)} gsdtsr : {form(apfd_ta1)} paintcontrol :{form(apfd_ta2)} ")
print(f"APFDC iofrol   : {form(apfdc)} gsdtsr : {form(apfdc1)} paintcontrol :{form(apfdc2)} ")
print(f"APFDa iofrol   : {form(apfda)} gsdtsr : {form(apfda1)} paintcontrol :{form(apfda2)} ")
print(f"NAPFD iofrol   : {form(napfd)} gsdtsr : {form(napfd1)} paintcontrol :{form(napfd2)} ")
print(f"rmse iofrol    : {form(rmse)} gsdtsr : {form(rmse1)} paintcontrol :{form(rmse2)} ")

results = pd.DataFrame([[apfd, apfdc, napfd, apfd_ta, apfda,rmse],
[apfd1, apfdc1, napfd1, apfd_ta1, apfda1,rmse1],
[apfd2, apfdc2, napfd2, apfd_ta2, apfda2,rmse2]], columns = ["APFD", "APFD_C", "NAPFD", "APFD_TA", "APFDA", "RMSE"],index = ['iofrol','gsdtsr','paintcontrl'])

results.to_csv('results/exp_results.csv')