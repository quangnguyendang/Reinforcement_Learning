import csv
import numpy as np
from matplotlib import pyplot as plt

# -------------------- UTILITY --------------------------
def plot_reward(x, total_reward_all_eps):
    plt.figure()
    plt.plot(x, total_reward_all_eps, label="DQN for ATARI - Acrobot")
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend(loc='best')
    plt.show()

    window_width = 100
    cumsum_vec = np.cumsum(np.insert(total_reward_all_eps, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    plt.figure()
    plt.plot(x[0:(-window_width+1)], ma_vec, label="DQN for ATARI - Acrobot")
    plt.xlabel('Episode')
    plt.ylabel('Moving-Average Rewards - window = 100')
    plt.legend(loc='best')
    plt.show()


file_name = "17_09_1.csv"
total_return = []
timestep = []
eps = []
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        eps.append(row[0])
        timestep.append(int(row[1]))
        total_return.append(float(row[2]))
        print("Episode #{}, Timestep {}: Return {}".format(eps[-1], timestep[-1], total_return[-1]))

plot_reward(eps, total_return)