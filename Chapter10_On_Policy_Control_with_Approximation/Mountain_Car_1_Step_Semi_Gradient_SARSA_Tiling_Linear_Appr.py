# Example 10.1 page 198 in Reinforcement Learning: An Introduction Book
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# 1-step Semi Gradient SARSA Implementation for MountainCar-v0 environment
# RBF Feature Vector Generated for Linear Approximation
# Reference from http://scikit-learn.org/stable/modules/kernel_approximation.html#rbf-kernel-approx


import gym
import numpy as np
import matplotlib.pyplot as plt
from tiles3 import tiles, IHT


class SarsaAgent:
    def __init__(self, environment=gym.make('MountainCar-v0'), num_of_tiles=8):
        self.env = environment
        self.state = self.env.reset()
        self.state_low_bound = self.env.observation_space.low
        self.state_high_bound = self.env.observation_space.high
        self.n_action = env.action_space.n

        self.action_space = gym.spaces.Discrete(self.n_action)

        self.d = 100
        self.w = np.random.rand(self.d)

        self.num_of_tiles = num_of_tiles
        self.d = 4096
        self.w = np.zeros(self.d)

        self.hash_table = IHT(self.d)
        self.s0_scale = 1.0 * self.d / (self.state_high_bound[0] - self.state_low_bound[0])
        self.s1_scale = 1.0 * self.d / (self.state_high_bound[1] - self.state_low_bound[1])

    def convert_tiling_vector(self, tiles):
        tiling_vector = []
        for i in range(self.d):
            if i in tiles:
                tiling_vector.append(0)
            else:
                tiling_vector.append(1)
        return np.array(tiling_vector)

    def feature_x(self, s, a):
        return self.convert_tiling_vector(
            tiles(self.hash_table, self.num_of_tiles, [s[0] * self.s0_scale, s[1] * self.s1_scale], [a]))

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
            # Note that now, we cannot know the feature of the action, therefore, may need to
            # catch exception and try another action
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
                self.w = self.w + learning_rate * (r_ + gamma * self.Q_hat(s_, self.A_max(state=s_, epsilon=epsilon)) - self.Q_hat(self.state, a)) * self.feature_x(self.state, a)
                self.state = s_
                n_trajectory += 1
                if self.state[0] >= 0.6:
                    num_steps_of_episode.append(n_trajectory)
                    print("Episode = {}, took {} to go to the goal.".format(i_episode, n_trajectory))
                    break
        return num_steps_of_episode

    def get_w(self):
        return self.w


env = gym.make('MountainCar-v0')
agent1 = SarsaAgent(env)
steps_of_episode = agent1.train(n_episode=10, learning_rate=0.0001, gamma=0.99, epsilon=0.001)
env.close()

plt.plot(steps_of_episode, label="Linear Approximation")
plt.xlabel('episode')
plt.ylabel('number of needed steps')
plt.legend(loc='best')
plt.show()

print(agent1.get_w())

# env = gym.make('MountainCar-v0')
# env = gym.wrappers.Monitor(env, "/tmp_videos/", video_callable=lambda episode_id: (episode_id + 1) % 2000 == 0, force=True)
# agent1 = SarsaAgent(env)
# steps_of_episode = agent1.train(n_episode=10000, learning_rate=0.0001, gamma=0.99, epsilon=0.001)
# env.close()
#
# plt.plot(steps_of_episode, label="Linear Approximation")
# plt.xlabel('episode')
# plt.ylabel('number of needed steps')
# plt.legend(loc='best')
# plt.show()
#
# print(agent1.get_w())
