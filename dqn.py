import pandas as pd
import numpy as np
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv
from sklearn.preprocessing import StandardScaler

from data_load import iofrol_raw

# Load the test case prioritization dataset
dataset = iofrol_raw

# Preprocessing
dataset = dataset.drop(['Id', 'Name', 'LastRun', 'Cycle'], axis=1)
dataset = pd.get_dummies(dataset, columns=['LastResults', 'Verdict', 'DurationGroup', 'TimeGroup'])

# Split the dataset into features (X) and target variable (y)
X = dataset.drop('CalcPrio', axis=1)
y = dataset['CalcPrio']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a custom gym environment for test case prioritization
class TestCasePrioritizationEnv(gym.Env):
    def __init__(self):
        super(TestCasePrioritizationEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(X_scaled.shape[1],))
        self.action_space = gym.spaces.Discrete(len(y))
        self.state = None

    def reset(self):
        self.state = np.random.randint(0, len(y))
        return X_scaled[self.state]

    def step(self, action):
        reward = -y[self.state]  # Negative of prioritization score as a reward
        done = True  # Terminate episode after a single step
        return X_scaled[self.state], reward, done, {}

# Create the gym environment
env = DummyVecEnv([lambda: TestCasePrioritizationEnv()])

# Initialize and train the DQN model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate the trained model
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break