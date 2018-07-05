# Example 7.1 page 118 in Reinforcement Learning: An Introduction Book
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# n-step TD Methods on 100-State Random Walk

import numpy as np
import gym
import matplotlib.pyplot as plt


class RandomWalkEnv:
    def __init__(self, n=19, action_range=1):
        self.n = n
        self.action_range = action_range
        self.action_space = gym.spaces.Discrete(2*self.action_range + 1)
        self.state_space = gym.spaces.Discrete(self.n)
        self.state = np.rint(self.n/2)

    def reset(self):
        self.state = int(self.n/2)

    def step(self, action):
        done = False
        if (not self.action_space.contains(action)) or (not self.state_space.contains(self.state + action - self.action_range)):
            print("Action {} is invalid at state {}".format(action, self.state))
            reward = -1000
        else:
            self.state = self.state + action - self.action_range
            reward = (-1) * (self.state == 0) + (1) *(self.state == self.n - 1)
            done = (self.state == self.n - 1) or (self.state == 0)
        return self.state, reward, done


class NStepTDAgent:
    def __init__(self, n=19, action_range=1, environment=RandomWalkEnv()):
        self.n = n
        self.n_action = action_range * 2 + 1
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.state_space = gym.spaces.Discrete(self.n)
        self.state = np.rint(self.n / 2)
        self.V = np.zeros(self.n)
        self.terminals = [0, self.n - 1]
        self.env = environment
        self.env.reset()

    def reset(self):
        self.state = int(np.rint(self.n / 2))
        self.env.reset()
        self.V = np.zeros(self.n)

    def next_state(self, s, a):
        real_action = a - int((self.n_action - 1)/2)
        if self.state_space.contains(int(s + real_action)):
            return s + real_action
        else:
            return -1

    def policy_pi(self, state, epsilon=0):
        if np.random.rand() <= epsilon:
            # Exploration
            a = np.random.randint(0, self.n_action)
            while self.next_state(state, a) == -1:
                a = np.random.randint(0, self.n_action)
            return a
        else:
            # Exploitation
            max_a = []
            maxV = -np.inf
            for a in range(0, self.n_action):
                if self.next_state(state, a) != -1:
                    if self.V[self.next_state(state, a)] > maxV:
                        max_a = [a]
                        maxV = self.V[self.next_state(state, a)]
                    elif self.V[self.next_state(state, a)] == maxV:
                        max_a.append(a)
            return max_a[np.random.randint(0, len(max_a))]

    def random_walk(self, n_episode=5000, learning_rate=0.001, gamma=0.99, epsilon=1, n=1):
        self.reset()
        i_episode = 0
        while i_episode < n_episode:
            if i_episode % 1000 == 0:
                print("Episode #{} is run.".format(i_episode))
            self.env.reset()
            S = [self.state]
            R = [0]
            T = self.n**10

            t = 0
            t_update = 0
            while t_update < T:
                if t < T:
                    A = self.policy_pi(S[t], epsilon)
                    S_, R_t, _ = self.env.step(A)
                    R.append(R_t)
                    S.append(int(S_))
                    if S_ in self.terminals:
                        T = t + 1
                t_update = t - n + 1
                if t_update >= 0:
                    G = 0
                    for i in range(t_update + 1, np.minimum(t_update + n, T) + 1):
                        G = G + (gamma**(i-t_update-1))*R[i]
                    if t_update + n < T:
                        G = G + (gamma**n)*self.V[int(S[t_update + n])]
                    self.V[int(S[t_update])] += learning_rate*(G - self.V[int(S[t_update])])
                t += 1

            i_episode += 1

    def get_v(self):
        return self.V

    def plot_v(self):
        plt.plot(self.get_v(), label="N_Step TD Learning")
        plt.xlabel('state')
        plt.ylabel('value')
        plt.legend(loc='best')
        plt.show()


env = RandomWalkEnv(100, 1)
agent1 = NStepTDAgent(100, 1, env)
agent1.random_walk(n_episode=10000, learning_rate=0.001, gamma=0.99, epsilon=1, n=1)
v_1 = agent1.get_v()
agent1.random_walk(n_episode=10000, learning_rate=0.001, gamma=0.99, epsilon=1, n=10)
v_2 = agent1.get_v()
agent1.random_walk(n_episode=10000, learning_rate=0.001, gamma=0.99, epsilon=1, n=50)
v_3 = agent1.get_v()

plt.title("n-step TD Prediction for Value Function")
plt.plot(v_1, label="TD(0)")
plt.plot(v_2, label="TD(9)")
plt.plot(v_3, label="TD(49)")
plt.xlabel('state')
plt.ylabel('value')
plt.legend(loc='best')
plt.show()