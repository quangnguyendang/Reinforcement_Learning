# Exploration and Exploitation with Multi-armed Bandit Problem
# Epsilon-greedy algorithm with Gym Rendering Demonstration with N-bandit and history of 5 latest rewards
# Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# https://www.youtube.com/watch?v=V8tu8omHs1U&t=12s

import gym
import gym_n_bandit
import numpy as np
from time import sleep

epsilon = 0.1
bandit_N = 10
N_steps = 200
N_experiments = 1

# Initialize an environment for the bandit problem.
env = gym.make('n-bandit-v0')

# Speed for illustration rendering.
speed = 500


# Incremental Implementation of action-value method
class Agent:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.k = np.zeros(bandit_N, dtype=np.int)  # number of time action were chosen
        self.Q = np.ones(bandit_N, dtype=np.float) * 5  # estimated value

    def update_Q(self, action, reward):
        self.k[action] += 1    # Update the count of selections for each action/bandit
        self.Q[action] += (1./self.k[action]) * (reward - self.Q[action])  # Update Q according to the formula
        print("Q table: ", self.Q)

    def choose_action(self):
        rand = np.random.random()
        if rand < self.epsilon:  # Exploring - Random and non-greedy action
            action_explore = np.random.randint(bandit_N)
            return action_explore
        else:  # Exploiting - Greedy action - Choose action that has max value
            action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
            return action_greedy


def human_bandit_select():
    return int(input("Which bandit do you bet on? (0-9)"))


mode = input("Choose game mode to start: Random (r) / Human (h) / Agent (a): ")
if mode == 'r':
    print("---- Random Agent Mode -----")
elif mode == 'h':
    print("-----   Human Mode     -----")
else:
    print("-----   Agent Mode     -----")

for i_experiment in range(N_experiments):
    observation = env.reset()
    agent = Agent(epsilon)

    # Store the number of selection for each action/bandit.
    action_history = np.zeros([N_experiments, bandit_N])
    t_stop = 0
    for t in range(N_steps):

        if mode == 'r':  # random mode - actions are selected randomly
            action_function = env.action_space.sample
            # randint(N_bandits)
            # Wait for few second to make the illustration easy to follow
            sleep(100 / speed if (speed > 0) else 0)
        elif mode == 'h':  # human mode - actions are selected at each step by human
            action_function = human_bandit_select
        else:  # agent mode - actions are selected according to epsilon-greedy algorithm
            action_function = agent.choose_action
            # Wait for few second to make the illustration easy to follow
            sleep(100 / speed if (speed > 0) else 0)

        action = action_function()

        print("Bandit #{} is bet! Spend {} steps".format(action, t+1))
        observation, reward, done, info = env.step(action)  # Respond to the environment.
        # print(observation, done, reward, info)
        agent.update_Q(action, reward)  # Update Q value after an action is selected.
        action_history[i_experiment][action] += 1
        print("Selection of {} bandits: {}".format(bandit_N, action_history[i_experiment]))
        if done:
            t_stop = t + 1
            break
    print("Experiment {} finished after {} timesteps".format(i_experiment+1, t_stop))
    print("Selection of {} bandits: {}".format(bandit_N, action_history[i_experiment]))

