class SarsaAgent:
    def __init__(self, environment=ServerEnv()):
        self.env = environment
        self.state = self.env.get_state()
        self.state_low_bound = self.env.observation_space.low
        self.state_high_bound = self.env.observation_space.high
        self.n_action = env.action_space.n
        self.action_space = gym.spaces.Discrete(self.n_action)

        # self.maxSize = (self.env.n + 1) * (np.max(self.env.priority)) * self.n_action
        self.maxSize = 1048567
        self.iht = IHT(self.maxSize)
        self.d = 40

        self.w = np.random.rand(self.d) * 0.01

    def feature_x(self, s, a):
        return np.array(tiles(self.iht, self.d, s, [a])) * 10**-3

    def Q_hat(self, s, a):
        if self.is_state_valid(s):
            return np.dot(np.transpose(self.w), self.feature_x(s, a))

    def is_state_valid(self, s):
        valid = True
        for i in range(np.array(s).shape[0]):
            if (s[i] < self.state_low_bound[i]) and (s[i] > self.state_high_bound[i]):
                valid = False
                break
        return valid

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

    def train(self, n_episode=5000, learning_rate=0.001, beta=0.01, gamma=0.99, epsilon=0.01):
        for i_episode in range(n_episode):
            if i_episode % 100 == 0:
                print("Episode {} is running.".format(i_episode))
            self.reset()
            r_avg = 0
            while True:
                r_ = -1
                while r_ not in self.env.priority:
                    a = self.A_max(state=self.state, epsilon=epsilon)
                    s_, r_, done, _ = self.env.step(a)
                TD_error = r_ - r_avg + self.Q_hat(s_, self.A_max(s_, epsilon)) - self.Q_hat(self.state, a)
                r_avg += beta * TD_error
                self.w = self.w + learning_rate * TD_error * self.feature_x(self.state, a)
                self.state = s_
                if done:
                    break

    def get_w(self):
        return self.w

    def print_policy(self):
        print(self.state_low_bound)
        print(self.state_high_bound)
        for s_0 in range(self.state_low_bound[0], self.state_high_bound[0] + 1):
            row = ""
            for s_1 in self.env.priority:
                s = [s_0, s_1]
                row += "[{},{}] ".format(np.round(self.Q_hat(s, 0), 3), np.round(self.Q_hat(s, 1), 3))
            print(row)
