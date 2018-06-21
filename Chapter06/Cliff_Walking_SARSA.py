# Example 6.6 page 108 in Reinforcement Learning: An Introduction Book
# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.

import numpy as np
import matplotlib.pyplot as plt
# -------------- Set Constants ----------------
n_col = 12
n_row = 4

terminal = 47
goal = 47
start = 36

cliff_start = 37
cliff_end = 46

action_space = ['U', 'D', 'L', 'R']

action_index_effect = {'U': (-1 * n_col),
                       'D': n_col,
                       'L': -1,
                       'R': 1}

action_index = {'U': 0,
                'D': 1,
                'L': 2,
                'R': 3}

state_space = range(0, n_col*n_row)

Q = np.zeros((len(state_space), len(action_space)))

# -------------- Utility Functions ----------------

def state_initialize():
    s = np.random.randint(0, n_row*n_col-1)
    if s in range(cliff_start, cliff_end + 1):
        s = start
    # s = start
    return s


def next_state(s, a):
    s_ = s + action_index_effect[a]
    row_s = int(s / n_col)
    col_s = s % n_col
    if (s_ not in range(0, n_col*n_row)) or \
            ((row_s == 0) and (a == 'U')) or \
            ((row_s == (n_row - 1)) and (a == 'D')) or \
            ((col_s == 0) and (a == 'L')) or \
            ((col_s == (n_col - 1)) and (a == 'R')):
        return -1
    else:
        return s_


def take_action(s, a):
    row_s = int(s/n_col)
    col_s = s % n_col
    s_ = s + action_index_effect[a]
    r = -1000
    if s_ in range(cliff_start, cliff_end + 1):
        s_ = start
        r = -100
    elif (row_s == 0 and a == 'U') or \
        ((row_s == n_row - 1) and a == 'D') or \
        (col_s == 0 and a == 'L') or \
        (col_s == (n_col - 1) and a == 'R'):
        s_ = -1
    else:
        r = -1
    return s_, r


def epsilon_greedy(s, epsilon):
    if np.random.rand() < epsilon:
        # Exploration
        a = action_space[np.random.randint(0, len(action_space))]
        while next_state(s, a) == -1:
            a = action_space[np.random.randint(0, len(action_space))]
        return a
    else:
        # Exploitation
        max_index = []
        maxQ = -99999999
        for i in range(0, len(action_space)):
            if (Q[s, i] > maxQ) and (next_state(s, action_space[i]) != -1):
                max_index = [i]
                maxQ = Q[s, i]
            elif (Q[s, i] == maxQ) and (next_state(s, action_space[i]) != -1):
                max_index.append(i)
        return action_space[max_index[np.random.randint(0, len(max_index))]]


def print_policy():
    print('\nOptimal Deterministic Policy:')
    for i in range(0, n_row):
        row_string = ""
        for j in range(0, n_col):
            s = i*n_col + j
            if s in range(cliff_start, cliff_end+1):
                row_string += '%8s' % "<-- "
            else:
                max_index = []
                maxQ = -99999999
                for a in range(0, len(action_space)):
                    if Q[s, a] != 0:
                        if (Q[s, a] > maxQ) and (next_state(s, action_space[a]) != -1):
                            max_index = [action_space[a]]
                            maxQ = Q[s, a]
                        elif (Q[s, a] == maxQ) and (next_state(s, action_space[a]) != -1):
                            max_index.append(action_space[a])
                row_string += '%8s' % str(max_index)
        print(row_string)


def print_Q():
    print('\nQ value table:')
    for i in range(0, n_row):
        row_string = ""
        for j in range(0, n_col):
            s = i*n_col + j
            row_string += str(np.round(Q[s, :], 1)) + "\t"
        print(row_string)


# -----------------------------------------------

def Sarsa_Walking(alpha=0.1, gamma=0.99, epsilon=0.05, max_episode=500):
    i_episode = 0
    return_eps = []
    while i_episode < max_episode:
        print("Iteration {} ...".format(i_episode))
        S = state_initialize()
        A = epsilon_greedy(S, epsilon)
        i_return = 0
        i_trajectory = str(S)
        while S != terminal:
            S_, R = take_action(S, A)
            A_ = epsilon_greedy(S_, epsilon)
            Q[S, action_index[A]] += alpha*(R + gamma*Q[S_, action_index[A_]] - Q[S, action_index[A]])
            i_trajectory += ' > ' + str(S_)
            A = A_
            S = S_
            i_return += R
        i_episode += 1
        print('  trajectory = {}'.format(i_trajectory))
        print('  return = {}'.format(i_return))
        return_eps.append(i_return)

    return return_eps

# -----------------MAIN PROGRAM---------------------

return_data = Sarsa_Walking(alpha=0.1, gamma=0.99, epsilon=0.001, max_episode=5000)

print_policy()
print_Q()

plt.plot(return_data)
plt.xlabel('iteration')
plt.ylabel('return')
plt.show()