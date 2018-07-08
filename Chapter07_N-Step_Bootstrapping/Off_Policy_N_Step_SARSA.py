# Example 7.1 page 118 in Reinforcement Learning: An Introduction Book
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# n-step SARSA for estimating Q with importance sampling factor on 100-State Random Walk

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


class NStepSARSAAgent:
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

    def random_walk(self, n_episode=5000, learning_rate=0.001, gamma=0.99, epsilon=1, n=1):
        self.reset()
        i_episode = 0
        while i_episode < n_episode:
            if i_episode % 5000 == 0:
                print("Episode #{} is run.".format(i_episode))
            self.env.reset()
            S = [self.state]  # Initialize and store S_0 # Terminal
            R = [0]
            A = [self.policy_pi(S[0], epsilon)]  # Select and store action A according to behavior policy b(.|S_0)
            T = self.n ** 10  # Set T = infinity

            t = 0
            t_update = 0
            while t_update < T:
                if t < T:
                    S_, R_t, _ = self.env.step(A[t])  # Take action A_t
                    R.append(R_t)  # Observe and store next reward as R_(t+1)
                    S.append(int(S_))  # Observe and store next state as S_(t+1)
                    if S[t+1] in self.terminals:
                        T = t + 1  # If S_(t+1) is the terminal then set T = t + 1 (the end time of the current episode)
                    else:
                        # Select and store next action A_(t+1) according to behavior policy b(.|S_t+1)
                        A.append(self.policy_pi(S[t+1], epsilon))
                t_update = t - n + 1
                if (t_update >= 0) and (S[t_update] not in self.terminals):
                    G = 0
                    p = 1.0
                    # Calculate importance sampling factor
                    for i in range(t_update + 1, np.minimum(t_update + n - 1, T - 1) + 1):
                        p *= (self.Q[S[i], A[i]] == np.max(self.Q[S[i]])) / ((1 - epsilon) + (epsilon/self.number_action(S[i])))
                        # p *= 1.0 / ((1 - epsilon) + (epsilon / self.number_action(S[i])))
                    for i in range(t_update + 1, np.minimum(t_update + n, T) + 1):
                        G = G + (gamma**(i-t_update-1))*R[i]
                    if t_update + n < T:
                        G = G + (gamma**n)*self.Q[int(S[t_update + n]), int(A[t_update + n])]
                    self.Q[int(S[t_update]), int(A[t_update])] += learning_rate*p*(G - self.Q[int(S[t_update]), int(A[t_update])])
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
                    max_A = [a - int(np.rint(self.n_action / 2)) + 1]
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


env = RandomWalkEnv(20, 2)
agent1 = NStepSARSAAgent(20, 2, env)
agent1.random_walk(n_episode=1000, learning_rate=0.01, gamma=0.99, epsilon=1, n=3)

print(agent1.get_Q())
print(agent1.get_target_policy())
agent1.plot_v()

# v_1 = agent1.get_v()
# agent1.random_walk(n_episode=10000, learning_rate=0.001, gamma=0.99, epsilon=1, n=10)
# v_2 = agent1.get_v()
# agent1.random_walk(n_episode=10000, learning_rate=0.001, gamma=0.99, epsilon=1, n=50)
# v_3 = agent1.get_v()

# plt.title("n-step TD Prediction for Value Function")
# plt.plot(v_1, label="TD(0)")
# plt.plot(v_2, label="TD(9)")
# plt.plot(v_3, label="TD(49)")
# plt.xlabel('state')
# plt.ylabel('value')
# plt.legend(loc='best')
# plt.show()