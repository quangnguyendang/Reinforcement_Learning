# Example 10.2 page 204 in Reinforcement Learning: An Introduction Book
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# An Access-Control Queuing Task Implemented with Differential Semi-Gradient SARSA

import numpy as np
import gym
from tiles3 import tiles, IHT

# Action
# 0: Reject
# 1:Accept


# State
# [#free server, priority]

class ServerEnv:
    def __init__(self, n=10, priority=[1, 2, 4, 8], p=0.06):
        self.n = n
        self.n_action = 2
        self.priority = priority
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.observation_space = gym.spaces.Discrete([self.n, len(self.priority)])
        self.observation_space.low = [0, np.min(priority)]
        self.observation_space.high = [self.n, np.max(priority)]
        self.state_env = [np.random.randint(self.n), self.priority[np.random.randint(len(self.priority))]]
        self.p = p
        self.i = 0
        self.i_max = 100

    def get_state(self):
        return self.state_env

    def reset(self):
        self.state_env = [np.random.randint(self.n), self.priority[np.random.randint(len(self.priority))]]
        self.i = 0
        # print('Reset Env to state ', self.state_env)
        return self.get_state()

    def step(self, action):
        self.i += 1
        done = (self.i >= self.i_max)
        if (not self.action_space.contains(action)) or (
         not (((self.state_env[0] - action) <= self.observation_space.high[0]) and (self.state_env[0] - action >= self.observation_space.low[0]))):
            # print("Action {} is invalid at state {}".format(action, self.state_env))
            reward = -100
        else:
            # print("Action {} is being performed at state {}".format(action, self.state_env))
            self.state_env[0] = np.min([np.max([self.state_env[0] - action + (np.random.rand() <= self.p), 0]), self.n])
            reward = action * self.state_env[1]
            self.state_env[1] = self.priority[np.random.randint(len(self.priority))]
            # print("Next State is ", self.state_env)
        return self.state_env, reward, done, None


# --------------------------------------------------------
class ApproximationSarsaAgent:
    def __init__(self, environment=ServerEnv(), num_of_tiles=8):
        self.env = environment
        self.state = self.env.get_state()
        self.state_low_bound = self.env.observation_space.low
        self.state_high_bound = self.env.observation_space.high
        self.n_action = env.action_space.n
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.num_of_tiles = num_of_tiles
        self.d = 2048
        self.w = np.zeros(self.d)

        self.hash_table = IHT(self.d)
        self.s0_scale = 1.0 * self.d / self.env.n
        self.s1_scale = 1.0 * self.d / (len(self.env.priority) - 1)

    def is_state_valid(self, s):
        valid = True
        for i in range(np.array(s).shape[0]):
            if (s[i] < self.state_low_bound[i]) or (s[i] > self.state_high_bound[i]):
                valid = False
                break
        return valid

    def reset(self):
        self.state = self.env.reset()

    def convert_tiling_vector(self, tiles):
        tiling_vector = []
        for i in range(self.d):
            if i in tiles:
                tiling_vector.append(0)
            else:
                tiling_vector.append(1)
        return np.array(tiling_vector)

    def feature_x(self, s0, s1, a):
        return self.convert_tiling_vector(
            tiles(self.hash_table, self.num_of_tiles, [s0 * self.s0_scale, s1 * self.s1_scale], [a]))

    def Q_hat(self, s0, s1, a):
        if self.is_state_valid([s0, s1]):
            return np.dot(np.transpose(self.w), self.feature_x(s0, s1, a))

    def A_max(self, state, epsilon):
        if np.random.rand() < epsilon:
            # Exploration
            a = np.random.randint(self.n_action)
            while not self.is_state_valid([state[0] - a, state[1]]):
                a = np.random.randint(self.n_action)
            return a
        else:
            # Exploitation
            max_a = []
            maxQ = -np.inf
            for a in range(0, self.n_action):
                if self.is_state_valid([state[0] - a, state[1]]):
                    if self.Q_hat(state[0], state[1], a) > maxQ:
                        max_a = [a]
                        maxQ = self.Q_hat(state[0], state[1], a)
                    elif self.Q_hat(state[0], state[1], a) == maxQ:
                        max_a.append(a)
            if len(max_a) > 0:
                return max_a[np.random.randint(0, len(max_a))]
            else:
                return np.random.randint(self.n_action)

    def train(self, n_episode=5000, learning_rate=0.001, beta=0.01, gamma=0.99, epsilon=0.01):
        for i_episode in range(n_episode):
            if i_episode % 100000 == 0:
                print("Approximation Sarsa - Episode {} is running.".format(i_episode))
            self.reset()
            r_avg = 0
            a = self.A_max(state=self.state, epsilon=epsilon)
            while not self.is_state_valid([self.state[0] - a, self.state[1]]):
                a = self.A_max(state=self.state, epsilon=epsilon)

            done = False
            while not done:
                s = np.copy(self.state)
                s_, r, done, _ = self.env.step(a)
                a_ = self.A_max(state=s_, epsilon=epsilon)
                while not self.is_state_valid([s_[0] - a_, s_[1]]):
                    a_ = self.A_max(state=s_, epsilon=epsilon)
                TD_error = r - r_avg + gamma * self.Q_hat(s_[0], s_[1], a_) - self.Q_hat(s[0], s[1], a)

                r_avg = (1 - beta) * r_avg + beta * TD_error
                self.w += learning_rate * TD_error * self.feature_x(s[0], s[1], a)

                self.state = s_
                a = a_

    def print_policy(self):
        print(" Tiling Approximation -  SARSA Policy")
        print("Number of free server 0 1 2 3 4 5 6 7 8 9 10")
        for s_1 in self.env.priority:
            row = "           Priority " + str(s_1) + " "
            for s_0 in range(self.state_low_bound[0], self.state_high_bound[0] + 1):
                if self.Q_hat(s_0, s_1, 0) > self.Q_hat(s_0, s_1, 1):
                    row += "R "
                else:
                    row += "A "
            print(row)
        print("\n")

    def print_Q(self):
        for s_0 in range(self.state_low_bound[0], self.state_high_bound[0] + 1):
            row = ""
            for s_1 in self.env.priority:
                row += "[{},{}] ".format(np.round(self.Q_hat(s_0, s_1, 0), 3), np.round(self.Q_hat(s_0, s_1, 1), 3))
            print(row)
        print("\n")


# --------------------------------------------------------
class TBSarsaAgent:
    def __init__(self, environment=ServerEnv()):
        self.env = environment
        self.state = self.env.get_state()
        self.state_low_bound = self.env.observation_space.low
        self.state_high_bound = self.env.observation_space.high
        self.n_action = env.action_space.n
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.Q = np.zeros([self.state_high_bound[0] + 1, self.state_high_bound[0] + 1, self.n_action])

    def is_state_valid(self, s):
        valid = True
        for i in range(np.array(s).shape[0]):
            if (s[i] < self.state_low_bound[i]) or (s[i] > self.state_high_bound[i]):
                valid = False
                break
        return valid

    def reset(self):
        self.state = self.env.reset()

    def A_max(self, state, epsilon):
        if np.random.rand() < epsilon:
            # Exploration
            a = np.random.randint(self.n_action)
            while not self.is_state_valid([state[0] - a, state[1]]):
                a = np.random.randint(self.n_action)
            return a
        else:
            # Exploitation
            max_a = []
            maxQ = -np.inf
            for a in range(0, self.n_action):
                if self.is_state_valid([state[0] - a, state[1]]):
                    if self.Q[state[0], state[1], a] > maxQ:
                        max_a = [a]
                        maxQ = self.Q[state[0], state[1], a]
                    elif self.Q[state[0], state[1], a] == maxQ:
                        max_a.append(a)
            if len(max_a) > 0:
                return max_a[np.random.randint(0, len(max_a))]
            else:
                return np.random.randint(self.n_action)

    def train(self, n_episode=5000, learning_rate=0.001, beta=0.01, gamma=0.99, epsilon=0.01):
        for i_episode in range(n_episode):
            if i_episode % 100000 == 0:
                print("TB Sarsa - Episode {} is running.".format(i_episode))
            self.reset()
            r_avg = 0
            a = self.A_max(state=self.state, epsilon=epsilon)
            while not self.is_state_valid([self.state[0] - a, self.state[1]]):
                a = self.A_max(state=self.state, epsilon=epsilon)

            while True:
                s = np.copy(self.state)
                s_, r, done, _ = self.env.step(a)
                a_ = self.A_max(state=s_, epsilon=epsilon)
                while not self.is_state_valid([s_[0] - a_, s_[1]]):
                    a_ = self.A_max(state=s_, epsilon=epsilon)
                TD_error = r - r_avg + gamma * self.Q[s_[0], s_[1], a_] - self.Q[s[0], s[1], a]
                r_avg = (1 - beta) * r_avg + beta * TD_error
                self.Q[s[0], s[1], a] += learning_rate * TD_error
                self.state = s_
                a = a_
                if done:
                    break

    def get_Q(self):
        return self.Q

    def print_policy(self):
        print(" Tabular SARSA Policy")
        print("Number of free server 0 1 2 3 4 5 6 7 8 9 10")
        for s_1 in self.env.priority:
            row = "           Priority " + str(s_1) + " "
            for s_0 in range(self.state_low_bound[0], self.state_high_bound[0] + 1):
                if self.Q[s_0, s_1, 0] > self.Q[s_0, s_1, 1]:
                    row += "R "
                else:
                    row += "A "
            print(row)
        print("\n")

    def print_Q(self):
        for s_0 in range(self.state_low_bound[0], self.state_high_bound[0] + 1):
            row = ""
            for s_1 in self.env.priority:
                row += "[{},{}] ".format(np.round(self.Q[s_0, s_1, 0], 3), np.round(self.Q[s_0, s_1, 1], 3))
            print(row)
        print("\n")


# --------------------------------------------------------
env = ServerEnv(n=10, priority=[1, 2, 4, 8], p=0.06)

agent1 = ApproximationSarsaAgent(environment=env, num_of_tiles=8)
agent1.train(n_episode=1000000, learning_rate=0.01, beta=0.01, gamma=1, epsilon=0.1)
agent1.print_policy()
agent1.print_Q()

agent2 = TBSarsaAgent(environment=env)
agent2.train(n_episode=1000000, learning_rate=0.01, beta=0.01, gamma=1, epsilon=0.1)
agent2.print_policy()
agent2.print_Q()
