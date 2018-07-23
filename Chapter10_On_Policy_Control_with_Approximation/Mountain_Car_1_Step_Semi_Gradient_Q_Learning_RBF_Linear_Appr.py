# Example 10.1 page 198 in Reinforcement Learning: An Introduction Book
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# 1-step Semi Gradient Q-learning Implementation for MountainCar-v0 environment
# RBF Feature Vector Generated for Linear Approximation
# Reference from http://scikit-learn.org/stable/modules/kernel_approximation.html#rbf-kernel-approx

import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler


class QLearningAgent:
    def __init__(self, environment=gym.make('MountainCar-v0')):
        self.env = environment
        self.state = self.env.reset()
        self.state_low_bound = self.env.observation_space.low
        self.state_high_bound = self.env.observation_space.high
        self.n_action = env.action_space.n

        self.action_space = gym.spaces.Discrete(self.n_action)

        self.d = 100
        self.w = np.random.rand(self.d)

        self.feature = RBFSampler(gamma=1, random_state=1)
        X = []
        for _ in range(100000):
            s = env.observation_space.sample()
            sa = np.append(s, np.random.randint(self.n_action))
            X.append(sa)
        self.feature.fit(X)

    def feature_x(self, s, a):
        # print('state = ', s, ' & action = ', a)
        feature_sa = self.feature.transform([[s[0], s[1], a]])
        # print(feature_sa)
        return feature_sa

    def is_state_valid(self, s):
        valid = True
        for i in range(s.shape[0]):
            if (s[i] < self.state_low_bound[i]) and (s[i] > self.state_high_bound[i]):
                valid = False
        return valid

    def Q_hat(self, s, a):
        if self.is_state_valid(s):
            return np.dot(self.feature_x(s, a), np.transpose(self.w))

    def reset(self):
        self.state = self.env.reset()

    def A_max(self, state, epsilon):
        if np.random.rand() < epsilon:
            # Exploration
            return np.random.randint(self.n_action)
        else:
            # Exploitation
            max_a = []
            maxQ = -np.inf
            for a in range(0, self.n_action):
                if self.Q_hat(state, a) > maxQ:
                    max_a = [a]
                    maxQ = self.Q_hat(state, a)
                elif self.Q_hat(state, a) == maxQ:
                    max_a.append(a)
            if max_a != []:
                return max_a[np.random.randint(0, len(max_a))]
            else:
                return np.random.randint(self.n_action)

    def train(self, n_episode=5000, learning_rate=0.01, gamma=0.99, epsilon=0.01):
        num_steps_of_episode = []
        for i_episode in range(n_episode):
            self.reset()
            n_trajectory = 0
            while True:
                while True:
                    try:
                        a = self.A_max(state=self.state, epsilon=epsilon)
                        s_, r_, done, _ = self.env.step(a)
                        # env.render()
                        break
                    except (RuntimeError, TypeError, NameError):
                        print("Action {} at state {} is invalid!".format(a, self.state))
                self.w = self.w + learning_rate * (r_ + gamma * self.Q_hat(s_, self.A_max(state=s_, epsilon=0)) - self.Q_hat(self.state, a)) * self.feature_x(self.state, a)
                self.state = s_
                n_trajectory += 1
                if done:
                    num_steps_of_episode.append(n_trajectory)
                    print("Episode = {}, took {} to go to the goal.".format(i_episode, n_trajectory))
                    break
        return num_steps_of_episode

    def get_w(self):
        return self.w


class SarsaAgent:
    def __init__(self, environment=gym.make('MountainCar-v0')):
        self.env = environment
        self.state = self.env.reset()
        self.state_low_bound = self.env.observation_space.low
        self.state_high_bound = self.env.observation_space.high
        self.n_action = env.action_space.n

        self.action_space = gym.spaces.Discrete(self.n_action)

        self.d = 100
        self.w = np.random.rand(self.d)

        self.feature = RBFSampler(gamma=1, random_state=1)
        X = []
        for _ in range(100000):
            s = env.observation_space.sample()
            sa = np.append(s, np.random.randint(self.n_action))
            X.append(sa)
        self.feature.fit(X)

    def feature_x(self, s, a):
        # print('state = ', s, ' & action = ', a)
        feature_sa = self.feature.transform([[s[0], s[1], a]])
        # print(feature_sa)
        return feature_sa

    def is_state_valid(self, s):
        valid = True
        for i in range(s.shape[0]):
            if (s[i] < self.state_low_bound[i]) and (s[i] > self.state_high_bound[i]):
                valid = False
        return valid

    def Q_hat(self, s, a):
        if self.is_state_valid(s):
            return np.dot(self.feature_x(s, a), np.transpose(self.w))

    def reset(self):
        self.state = self.env.reset()

    def A_max(self, state, epsilon):
        if np.random.rand() < epsilon:
            # Exploration
            return np.random.randint(self.n_action)
        else:
            # Exploitation
            max_a = []
            maxQ = -np.inf
            for a in range(0, self.n_action):
                if self.Q_hat(state, a) > maxQ:
                    max_a = [a]
                    maxQ = self.Q_hat(state, a)
                elif self.Q_hat(state, a) == maxQ:
                    max_a.append(a)
            if max_a != []:
                return max_a[np.random.randint(0, len(max_a))]
            else:
                return np.random.randint(self.n_action)

    def train(self, n_episode=5000, learning_rate=0.01, gamma=0.99, epsilon=0.01):
        num_steps_of_episode = []
        for i_episode in range(n_episode):
            self.reset()
            n_trajectory = 0
            a = self.A_max(state=self.state, epsilon=epsilon)
            while True:
                while True:
                    try:
                        s_, r_, done, _ = self.env.step(a)
                        a_ = self.A_max(state=s_, epsilon=epsilon)
                        # env.render()
                        break
                    except (RuntimeError, TypeError, NameError):
                        print("Action {} at state {} is invalid!".format(a, self.state))
                self.w = self.w + learning_rate * (r_ + gamma * self.Q_hat(s_, a_) - self.Q_hat(self.state, a)) * self.feature_x(self.state, a)
                self.state = s_
                a = a_
                n_trajectory += 1
                if done:
                    num_steps_of_episode.append(n_trajectory)
                    print("Episode = {}, took {} to go to the goal.".format(i_episode, n_trajectory))
                    break
        return num_steps_of_episode

    def get_w(self):
        return self.w


env = gym.make('MountainCar-v0')
agent1 = QLearningAgent(env)
steps_of_episode_QLearning = agent1.train(n_episode=10000, learning_rate=0.0001, gamma=0.99, epsilon=0.001)
env.close()

env = gym.make('MountainCar-v0')
agent2 = SarsaAgent(env)
steps_of_episode_SARSA = agent2.train(n_episode=10000, learning_rate=0.0001, gamma=0.99, epsilon=0.001)
env.close()

plt.plot(steps_of_episode_SARSA, label="SARSA Linear Approximation")
plt.plot(steps_of_episode_QLearning, label="Q-Learning Linear Approximation")
plt.xlabel('episode')
plt.ylabel('number of needed steps')
plt.legend(loc='best')
plt.show()
