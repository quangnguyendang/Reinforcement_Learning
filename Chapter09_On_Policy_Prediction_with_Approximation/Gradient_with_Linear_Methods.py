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
        self.env.reset()
        self.x = np.zeros((int(self.n / self.n_group), int(self.n / self.n_group)))
        for i in range(int(self.n / self.n_group)):
            self.x[i, i] = 1
        # self.x = np.random.rand(int(self.n / self.n_group), int(self.n / self.n_group))
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
                if (self.v_hat(state) > maxV) and (self.next_state(state, a) != -1):
                    max_a = [a]
                    maxQ = self.v_hat(state)
                elif (self.v_hat(state) == maxV) and (self.next_state(state, a) != -1):
                    max_a.append(a)
            return max_a[np.random.randint(0, len(max_a))]

    def train_w(self, n_episode=500, learning_rate=0.1, gamma=0.99, epsilon=0.001):
        for i_episode in range(n_episode):
            if i_episode % 1000 == 0:
                print("Linear Approximation - Episode #{} is run.".format(i_episode))
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

    def get_v_hat(self):
        v_est = []
        for s in range(self.n):
            v_est.append(self.v_hat(s))
        return np.array(v_est)

    def plot_v_hat(self):
        plt.plot(self.get_v_hat(), label="Stage Aggregation")
        plt.xlabel('state')
        plt.ylabel('value')
        plt.legend(loc='best')
        plt.show()


class SarsaAgent:
    def __init__(self, n=1000, n_action=202, environment=RandomWalkEnv()):
        self.n = n
        self.n_action = n_action
        self.n_group = int((self.n_action - 1)/2)
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.state_space = gym.spaces.Discrete(self.n)
        self.state = int(self.n / 2)
        self.Q = np.zeros((self.n, self.n_action))
        self.env = environment
        self.env.reset()

    def reset(self):
        self.state = int(self.n / 2)
        self.env.reset()

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
            maxQ = -100000001
            for a in range(0, 2*self.n_group + 2):
                if (self.Q[state, a] > maxQ) and (self.next_state(state, a) != -1):
                    max_a = [a]
                    maxQ = self.Q[state, a]
                elif (self.Q[state, a] == maxQ) and (self.next_state(state, a) != -1):
                    max_a.append(a)
            return max_a[np.random.randint(0, len(max_a))]

    def walk(self, n_episode=5000, learning_rate=0.01, gamma=0.99, epsilon=0.01):
        i_episode = 0
        while i_episode < n_episode:
            if i_episode % 1000 == 0:
                print("Sarsa Evaluation - Episode #{} is run.".format(i_episode))
            self.env.reset()
            S = self.state
            A = self.A_max(S, epsilon)
            # i_trajectory = ""
            while True:
                # i_trajectory += str(S) + " --> "
                S_, R, done = self.env.step(A)
                A_ = self.A_max(S_, epsilon)
                self.Q[S, A] += learning_rate * (R + gamma * self.Q[S_, A_] - self.Q[S, A])
                A = A_
                S = S_
                if done:
                    break
            # print(i_trajectory)
            i_episode += 1

    def get_v(self):
        v = []
        for s in range(self.n):
            v_current = 0
            for a in range(self.n_action):
                v_current += self.Q[s, a]/self.n_action
            v.append(v_current)
        return v

    def plot_v(self):
        plt.plot(self.get_v(), label="Sarsa Learning")
        plt.xlabel('state')
        plt.ylabel('value')
        plt.legend(loc='best')
        plt.show()


env = RandomWalkEnv()
agent1 = StateAggregateAgent(1000, 100, env)
agent1.train_w(n_episode=100000, learning_rate=0.00002, gamma=0.99, epsilon=1)
V_1 = agent1.get_v_hat()
# agent1.plot_v_hat()

agent2 = SarsaAgent(1000, 202, env)
agent2.walk(n_episode=100000000, learning_rate=0.0001, gamma=0.99, epsilon=1)
V_2 = agent2.get_v()
# agent2.plot_v()

plt.plot(V_1, label="Linear Approximation")
plt.plot(V_2, label="SARSA")
plt.xlabel('state')
plt.ylabel('value')
plt.legend(loc='best')
plt.show()
