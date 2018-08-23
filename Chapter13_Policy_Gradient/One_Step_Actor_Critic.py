# Python Code for Chapter 13 Illustration: Policy Gradient
# Mountain Car Example
# RBF Feature Vector Generated for Linear Approximation
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.

import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler


class ActorCriticAgent:
    def __init__(self, environment=gym.make('MountainCar-v0')):
        self.env = environment
        self.state = self.env.reset()
        self.state_low_bound = self.env.observation_space.low
        self.state_high_bound = self.env.observation_space.high
        self.n_action = env.action_space.n

        self.action_space = gym.spaces.Discrete(self.n_action)

        self.d = 100
        self.w = np.random.rand(self.d)
        self.theta = np.random.rand(self.d)

        self.feature = RBFSampler(gamma=1, random_state=1)
        X = []
        for _ in range(100000):
            s = env.observation_space.sample()
            sa = np.append(s, np.random.randint(self.n_action))
            X.append(sa)
        self.feature.fit(X)

    def feature_x(self, s, a):
        feature_sa = 1.0 * self.feature.transform([[s[0], s[1], a]])
        return feature_sa

    def h(self, s, a):
        return np.dot(self.feature_x(s, a), np.transpose(self.theta))

    def pi(self, s, a):
        demoninator = 0.0
        for b in range(self.n_action):
            demoninator = demoninator + np.exp(self.h(s, b))
        numerator = np.exp(self.h(s, a))
        return numerator/demoninator

    def grad_ln_theta(self, s, a):
        sum_b = 0
        for b in range(self.n_action):
            sum_b = sum_b + self.pi(s, b) * self.feature_x(s, b)
        return self.feature_x(s, a) - sum_b

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

    def action_selection(self, state):
        action_prob = np.array([])
        for b in range(self.n_action):
            action_prob = np.append(action_prob, self.pi(state, b))
        return np.random.choice(np.arange(len(action_prob)), p=action_prob)

    def train(self, n_episode=5000, alpha_w=0.001, alpha_theta=0.001, gamma=0.99):
        num_steps_of_episode = []
        for i_episode in range(n_episode):
            self.reset()  # Initialize S
            n_trajectory = 0
            I = 1.0

            done = False
            while not done:
                s = self.state
                a = self.action_selection(state=self.state)
                s_, r, done, _ = self.env.step(a)
                TD_error = r + gamma * self.Q_hat(s_, self.action_selection(state=s_)) - self.Q_hat(s, a)
                self.w = self.w + alpha_w * I * TD_error * self.feature_x(s, a)
                self.theta = self.theta + alpha_theta * I * TD_error * self.grad_ln_theta(s, a)
                I = gamma * I
                self.state = s_

                n_trajectory += 1
                if done:
                    num_steps_of_episode.append(n_trajectory)
                    if i_episode % 10 == 0:
                        print("Episode = {}, took {} to go to the goal.".format(i_episode, n_trajectory))
                        # print(np.round(self.theta, 5))
                    break
        return num_steps_of_episode

    def get_w(self):
        return self.w

    def get_theta(self):
        return self.theta


env = gym.make('MountainCar-v0')
agent1 = ActorCriticAgent(env)
steps_of_episode_Monte = agent1.train(n_episode=2000, alpha_w=0.0005, alpha_theta=0.0005, gamma=0.99)
env.close()

plt.plot(steps_of_episode_Monte, label="1-Step Actor-Critic")
plt.xlabel('episode')
plt.ylabel('number of needed steps')
plt.legend(loc='best')
plt.show()
