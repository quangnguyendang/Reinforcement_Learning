# Example 7.1 page 118 in Reinforcement Learning: An Introduction Book
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# n-step Tree Backup Algorithm (Off Policy Learning without Importance Sampling) on 19-State Random Walk

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
            reward = (-1) * (self.state == 0) + (self.state == self.n - 1)
            done = (self.state == self.n - 1) or (self.state == 0)
        return self.state, reward, done


class NStepBackupTreeAgent:
    def __init__(self, n=19, action_range=1, environment=RandomWalkEnv()):
        self.n = n
        self.n_action = action_range * 2 + 1
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.state_space = gym.spaces.Discrete(self.n)
        self.state = np.rint(self.n / 2)
        self.Q = np.zeros((self.n, self.n_action))
        self.terminals = [0, self.n - 1]
        self.env = environment
        self.env.reset()

    def reset(self):
        self.state = int(np.rint(self.n / 2))
        self.env.reset()
        self.Q = np.zeros((self.n, self.n_action))

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
            maxQ = -np.inf
            for a in range(0, self.n_action):
                if self.next_state(state, a) != -1:
                    if self.Q[state, a] > maxQ:
                        max_a = [a]
                        maxQ = self.Q[state, a]
                    elif self.Q[state, a] == maxQ:
                        max_a.append(a)
            return max_a[np.random.randint(0, len(max_a))]

    def number_action(self, state):
        action_count = 0
        for a in range(0, self.n_action):
            if self.next_state(state, a) != -1:
                action_count += 1
        return action_count

    def policy(self, s, a, epsilon):
        Q_max = -np.inf
        a_max = []
        for a_i in range(self.n_action):
            if Q_max < self.Q[s, a_i]:
                Q_max = self.Q[s, a_i]
                a_max = [a_i]
            elif Q_max == self.Q[s, a_i]:
                a_max.append(a_i)

        if self.Q[s, a] == np.max(self.Q[s]):
            return (1 - epsilon) / len(a_max) + epsilon / self.number_action(s)
        else:
            return epsilon / self.number_action(s)

    def random_walk(self, n_episode=5000, learning_rate=0.001, gamma=0.99, epsilon=1, n=1):
        self.reset()
        i_episode = 0
        while i_episode < n_episode:
            if i_episode % 5000 == 0:
                print("Episode #{} is run.".format(i_episode))
            self.env.reset()
            S = [self.state]  # Initialize and store S_0 # Terminal
            R = [0]
            A = [self.policy_pi(S[0], epsilon)]  # Select and store action A according to policy pi(.|S_0)
            Q = [self.Q[int(S[0]), int(A[0])]]  # Store Q(S_0, A_0) as Q_0
            T = self.n ** 10  # Set T = infinity
            delta = []  # Temporal Error
            pi = [0]

            t = 0
            t_update = 0
            while t_update < T:
                print("At {}, self.Q = {}".format(t, self.Q))
                if t < T:
                    S_, R_, _ = self.env.step(A[t])  # Take action A_t
                    R.append(R_)  # Observe and store next reward as R_(t+1)
                    S.append(int(S_))  # Observe and store next state as S_(t+1)
                    if S[t+1] in self.terminals:
                        T = t + 1  # If S_(t+1) is the terminal then set T = t + 1 (the end time of the current episode)
                        delta.append(R[t+1] - Q[t])  # Store delta_t
                    else:
                        delta_t = R[t+1] - Q[t]
                        for a_i in range(self.n_action):
                            if self.next_state(S[t+1], a_i) != -1:
                                delta_t += gamma*self.policy(S[t+1], a_i, epsilon)
                        delta.append(delta_t)

                        # Select arbitrarily and store an action as A_t+1 <-- Random policy applied
                        A.append(self.policy_pi(S[t+1], epsilon))

                        # Store Q(S_t+1, A_t+1) as Q_t+1
                        Q.append(self.Q[int(S[t+1]), int(A[t+1])])

                        # Store pi_t+1
                        pi.append(self.policy(int(S[t+1]), int(A[t+1]), epsilon))

                t_update = t - n + 1
                if (t_update >= 0) and (S[t_update] not in self.terminals):
                    G = Q[t_update]
                    Z = 1
                    for k in range(t_update, np.minimum(t_update + n - 1, T - 1) + 1):
                        G = G + Z*delta[k]
                        # print("t_update = ", t_update)
                        # print("t_update + n - 1 = ", t_update + n - 1)
                        # print("T - 1 = ", T - 1)
                        # print("k+1 = ", k+1)
                        # print("pi = ", pi)
                        if k+1 < (T-1):
                            Z = gamma*Z*pi[k+1]
                    self.Q[int(S[t_update]), int(A[t_update])] += learning_rate*(G - self.Q[int(S[t_update]), int(A[t_update])])
                t += 1
            i_episode += 1

    def get_Q(self):
        return self.Q

    def get_v(self):
        V = []
        for s in range(self.n):
            max_Q = -np.inf
            for a in range(self.n_action):
                if (max_Q < self.Q[s, a]) and (self.Q[s,a] != 0):
                    max_Q = self.Q[s, a]
            V.append(max_Q)
        return V

    def get_target_policy(self):
        A = []
        for s in range(self.n):
            max_Q = -np.inf
            max_A = []
            for a in range(self.n_action):
                if (max_Q < self.Q[s, a]) and (self.Q[s, a] != 0):
                    max_Q = self.Q[s, a]
                    max_A = [a]
                elif (max_Q == self.Q[s, a]) and (self.Q[s, a] != 0):
                    max_A.append(a)
            A.append(max_A)
        return A

    def plot_v(self):
        plt.plot(self.get_v(), label="N_Step SARSA Estimation")
        plt.xlabel('state')
        plt.ylabel('value')
        plt.legend(loc='best')
        plt.show()


env = RandomWalkEnv(10, 1)
agent1 = NStepBackupTreeAgent(10, 1, env)
agent1.random_walk(n_episode=1000, learning_rate=0.01, gamma=0.9, epsilon=0.5, n=1)

print(agent1.get_Q())
print(agent1.get_target_policy())
agent1.plot_v()
