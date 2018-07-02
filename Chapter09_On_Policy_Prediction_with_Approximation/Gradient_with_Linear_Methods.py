# Example 6.6 page 108 in Reinforcement Learning: An Introduction Book
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# State aggregation on 1000-state Random Walk

import numpy as np
import gym
import matplotlib.pyplot as plt


class RandomWalkEnv:
    def __init__(self, n=1000, action_range=100):
        self.n = n
        self.action_range = action_range
        self.action_space = gym.spaces.Discrete(2*self.action_range + 2)
        self.state_space = gym.spaces.Discrete(self.n)
        self.state = int(self.n/2)

    def reset(self):
        self.state = int(self.n/2)

    def step(self, action):
        done = False
        if (not self.action_space.contains(action)) or (not self.state_space.contains(self.state + action - self.action_range - 1)):
            print("Action {} is invalid at state {}".format(action, self.state))
            reward = -1000
        else:
            # print("Action {} is being performed at state {}".format(action, self.state))
            self.state = self.state + action - self.action_range - 1
            # print("Next State is ", self.state)
            reward = (-1) * (self.state == 0) + (1) *(self.state == self.n - 1)
            done = (self.state == self.n - 1) or (self.state == 0)
        return self.state, reward, done


class StateAggregateAgent:
    def __init__(self, n=1000, n_group=100, environment=RandomWalkEnv()):
        self.n = n
        self.n_group = n_group
        self.action_space = gym.spaces.Discrete(2*self.n_group + 1)
        self.state_space = gym.spaces.Discrete(self.n)
        self.state = int(self.n / 2)
        self.env = environment
        self.x = np.zeros((int(self.n / self.n_group), int(self.n / self.n_group)))
        for i in range(int(self.n / self.n_group)):
            self.x[i, i] = 1
        self.w = np.random.rand(int(self.n / self.n_group))

    def reset(self):
        self.state = int(self.n / 2)
        self.env.reset()

    def v_hat(self, state):
        if self.state_space.contains(state):
            return np.dot(np.transpose(self.w), self.x[int(state/self.n_group)])
        else:
            return -100000000

    def next_state(self, s, a):
        if self.state_space.contains(s + a - self.n_group - 1):
            return s + a - self.n_group - 1
        else:
            return -1

    def A_max(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            # Exploration
            a = np.random.randint(0, 2*self.n_group + 2)
            while self.next_state(state, a) == -1:
                a = np.random.randint(0,  2*self.n_group + 2)
            return a
        else:
            # Exploitation
            max_a = []
            maxV = -100000001
            for a in range(0, 2*self.n_group + 2):
                if (self.v_hat(self.next_state(state, a)) > maxV) and (self.next_state(state, a) != -1):
                    max_a = [a]
                    maxQ = self.v_hat(self.next_state(state, a))
                elif (self.v_hat(self.next_state(state, a)) == maxV) and (self.next_state(state, a) != -1):
                    max_a.append(a)
            return max_a[np.random.randint(0, len(max_a))]

    def train_w(self, n_episode=500, learning_rate=0.1, gamma=0.99, epsilon=0.001):
        for i_episode in range(n_episode):
            print("Episode #{} is run.".format(i_episode))
            self.reset()
            while True:
                a = self.A_max(state=self.state, epsilon=epsilon)
                s_, reward, done = self.env.step(a)
                self.w = self.w + learning_rate * (reward + gamma*self.v_hat(s_) - self.v_hat(self.state)) * self.x[int(self.state/self.n_group)]
                self.state = s_
                if done:
                    break
            # self.plot_v_hat()

    def get_w(self):
        return self.w

    def plot_v_hat(self):
        V_est = []
        for s in range(self.n):
            V_est.append(self.v_hat(s))
        plt.plot(V_est, label="Stage Aggregation")
        plt.xlabel('state')
        plt.ylabel('value')
        plt.legend(loc='best')
        plt.show()


env = RandomWalkEnv()
agent = StateAggregateAgent(1000, 100, env)
agent.plot_v_hat()
agent.train_w(n_episode=100000, learning_rate=0.00002, gamma=0.99, epsilon=1)
agent.plot_v_hat()
print(agent.get_w())



