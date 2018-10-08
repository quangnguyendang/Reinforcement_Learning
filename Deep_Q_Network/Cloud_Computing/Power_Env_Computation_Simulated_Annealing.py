import numpy as np
from matplotlib import pyplot as plt
import random
from random import random
# ----------------- CONSTANTS SETTINGS -----------------------
BETA = 1000000000



# ----------------- CLOUDING SYSTEM ENVIRONMENT --------------
class cloud_Env:
    def __init__(self, U, M, D, W, Tmax, P, R_sbs, f_local, f_sbs, f_mbs, sigma_l, sigma_sbs, sigma_mbs):
        # Number of Users
        self.U = U
        # Number of Server (not counting Top Remote Server)
        self.M = M

        # Size of Task D_i(1 --> U)
        self.D = D

        # CPU Cycles W_i(1 --> U)
        self.W = W

        # Power to upload data from User i to Server j
        # P has dimension [U,M]
        self.P = P

        # Data rate to upload data from User i to Server j
        # R1 has dimension [U,M]
        self.R_sbs = R_sbs

        # Data rate to upload data from Server j to Top Remote Server is constant == BETA

        # Tmax is the constraint about time of the task of user i
        # Tmax has dimention [1, U]
        self.Tmax = Tmax

        # Power consumption for 1 CPU cycle
        # for local -- Dimension [1, U]
        self.sigma_l = sigma_l
        # for server -- Dimension [1, M]
        self.sigma_sbs = sigma_sbs
        # for top server -- Dimension [1]
        self.sigma_mbs = sigma_mbs

        # Initialize f for local, sbs, and mbs
        self.reset(f_local, f_sbs, f_mbs)

    def reset(self, f_local, f_sbs, f_mbs):
        # CPU freq of local user - has dimension [1, U]
        self.f_local = f_local

        # CPU freq of servers - has dimension [1, M]
        self.f_sbs = f_sbs

        # CPU freq of top remote server - single value
        self.f_mbs = f_mbs

    def step(self, A):
        # Input of Step is Action a which is a matrix containing all of users' actions
        # Action has dimension [U, M]
        # A_i,j = 0 (Task of User i is not perform at server j)
        # A_i,j = 1 (Task of User i is performed at server j)
        # A_i,j = -1 (Task  of User i is uploaded to server j,
        #   then continue uploaded from server j to top server and processed there)

        state = []


        # ------ TIME CALCULATION
        # Time to transmit data from local to servers 1-->M
        T_r_l2s = np.zeros(self.U)
        T_r_s2t = np.zeros(self.U)
        for i in range(self.U):
            # Calculate T_t_r_l2s for User i
            T_r_l2s_i = 0
            T_r_s2t_i = 0
            for j in range(self.M):
                if A[i,j] != 0:
                    T_r_l2s_i += self.D[i] / self.R_sbs[i,j]
                if A[i,j] == -1:
                    T_r_s2t_i += self.D[i] / BETA

            T_r_l2s[i] = T_r_l2s_i
            T_r_s2t[i] = T_r_s2t_i


            # Total time for uploading data from local to server/top server for each user i
            # Dimension [1, U]
            T_r = T_r_l2s + T_r_s2t


            # Processing time
            # Local Processing time
            T_l = np.zeros(self.U)
            for i in range(self.U):
                if np.sum(A[i]) == 0:
                    T_l[i] = self.W[i] / self.f_local[i]

            # Server Processing time for each Task i
            T_sbs = np.zeros(self.U)
            T_mbs = np.zeros(self.U)
            for i in range(self.U):
                T_sbs_i = 0
                T_mbs_i = 0
                for j in range(self.M):
                    if A[i,j] == 1:
                        T_sbs_i += self.W[i] / self.f_sbs[j]
                    if A[i,j] == -1:
                        T_mbs_i += self.W[i] / self.f_mbs
                T_sbs[i] = T_sbs_i
                T_mbs[i] = T_mbs_i

        # Total time for uploading and processing data
        T = T_r + T_l + T_sbs + T_mbs


        # ------ Power Calculation

        # Power for processing
        E_l = np.zeros(self.U)
        E_sbs = np.zeros(self.U)
        E_mbs = np.zeros(self.U)
        for i in range(self.U):
            if np.sum(A[i]) == 0:
                E_l[i] += self.W[i] * self.sigma_l[i]
            elif np.sum(A[i]) == 1:
                selected_server = np.argmax(np.abs(A[i]))
                E_sbs[i] += self.W[i] * self.sigma_sbs[selected_server]
            elif np.sum(A[i]) == -1:
                E_mbs[i] += self.W[i] * self.sigma_mbs

        # Power for uploading data
        E_r_sbs = np.zeros(self.U)
        E_r_mbs = np.zeros(self.U)
        for i in range(self.U):
            selected_server = np.argmax(np.abs(A[i]))
            if np.abs(np.sum(A[i])) == 1:
                E_r_sbs[i] += self.P[i, selected_server] * T_r_l2s[i]

        # ENERGY ENTIRE SYSTEM
        E = E_l + E_sbs + E_mbs + E_r_sbs + E_r_mbs

        # # ENERGY OF USERS
        # E = E_l + E_r_sbs

        return state, T, E

# Recursion Agent
class Annelling_Agent:
    def __init__(self, env, U, M, VALID_ACTIONS):
        self.env = env
        self.U = U
        self.M = M
        self.VALID_ACTIONS = VALID_ACTIONS

    def check_action_constraint(self, A, T):
        # result is % of performed task
        result = 1

        if result == 1:
            count_fail = 0
            check_T_max = (T <= self.env.Tmax)
            for cond_i in check_T_max:
                if not cond_i:
                    count_fail += 1.0
            result = 1.0 - count_fail/self.U

        if result == 1:
            # print("Action: \n", A)
            check_upload_to_only_1_server = np.sum(np.abs(A), axis=1)
            # print("np.sum(np.abs(A), axis=1): \n", np.sum(np.abs(A), axis=1))
            check_upload_to_only_1_server = (check_upload_to_only_1_server <= 1)
            # print("check_upload_to_only_1_server: \n", check_upload_to_only_1_server)

            for cond_i in check_upload_to_only_1_server:
                if not cond_i:
                    result = 0
                    break

        if result == 1:
            f_sbs = np.copy(self.env.f_sbs)
            for i in range(self.env.U):
                for j in range(self.env.M):
                    if A[i,j] == 1:
                        f_sbs[j]= f_sbs[j] - self.env.W[i]

            for j in range(self.env.M):
                if f_sbs[j] < 0:
                    result = 0
                    break

        if result == 1:
            f_mbs = np.copy(self.env.f_mbs)
            for i in range(self.env.U):
                for j in range(self.env.M):
                    if A[i, j] == -1:
                        f_mbs = f_mbs - self.env.W[i]

            if f_mbs < 0:
                result = 0

        return result


    def anneal(sol):
        old_cost = cost(sol)
        T = 1.0
        T_min = 0.00001
        alpha = 0.9
        while T > T_min:
            i = 1
            while i <= 100:
                new_sol = neighbor(sol)
                new_cost = cost(new_sol)
                ap = acceptance_probability(old_cost, new_cost, T)
                if ap > random():
                    sol = new_sol
                    old_cost = new_cost
                i += 1
            T = T * alpha
        return sol, cost

# ------------------------ Data Generation ---------------------------------------
def Data_Gen(U):
    M = 5
    D = (np.random.randint(50, size=(U)) + 380) * 1e3

    W = (np.random.randint(200, size=(U)) + 800) * 1e6

    a = 1.1
    b = 2
    Tmax = (b - a) * np.random.random_sample(size=(U)) + a

    P = np.ones([U, M]) * 0.2

    R_sbs = (np.random.randint(9, size=(U,M)) + 1) * 1000000

    f_local = np.ones(U) * 800000000.0
    f_sbs = np.ones(M) * 4000000000.0
    f_mbs = 8000000000.0

    sigma_l = np.ones(U) * 0.5e-9
    sigma_sbs = np.ones(M) * 1e-9
    sigma_mbs = 1e-9

    return M, D, W, Tmax, P, R_sbs, f_local, f_sbs, f_mbs, sigma_l, sigma_sbs, sigma_mbs

# ------------------------------------ MAIN --------------------------------------
X = []
Y_rand = []
Y_local = []
Y_off = []
Y_off = []

for U in np.arange(15, 18):

    M, D, W, Tmax, P, R_sbs, f_local, f_sbs, f_mbs, sigma_l, sigma_sbs, sigma_mbs = Data_Gen(U)

    print("DATA GENERATION")
    print("U = {}, M = {}".format(U, M))
    print("D = ", D)
    print("W = ", W)
    print("Tmax = ", Tmax)
    print("P = ", P)
    print("R_sbs = ", R_sbs)
    print("f_local = ", f_local)
    print("f_sbs = ", f_sbs)
    print("f_mbs = ", f_mbs)
    print("sigma_l = ", sigma_l)
    print("sigma_sbs = ", sigma_sbs)
    print("sigma_mbs = ", sigma_mbs)


    env = cloud_Env(U, M, D, W, Tmax, P, R_sbs, f_local, f_sbs, f_mbs, sigma_l, sigma_sbs, sigma_mbs)
    agent = Recursion_Agent(env, U, M, [1, -1, 0])

    X.append(U)

    optimal_action, T, maxT = agent.find_optimal_by_random_search(N_iter=100)
    Y_rand.append(maxT)

    print("---------------------------------")
    print("Random Optimal: ")
    print("Action: \n", optimal_action)
    # print("E = ", E)
    print("T = ", T)
    print("Task can performed: ", maxT)

    optimal_action, T, maxT = agent.all_local_process()
    Y_local.append(maxT)

    print("---------------------------------")
    print("All local processing: ")
    print("Action: \n", optimal_action)
    # print("E = ", E)
    print("T = ", T)
    print("Task can performed: ", maxT)

    sbs_action, T, maxT_sbs = agent.find_optimal_in_all_sbs_pushing(N_iter=100)
    Y_off.append(maxT)

    print("---------------------------------")
    print("All offline processing: ")
    print("Action: \n", optimal_action)
    # print("E = ", E)
    print("T = ", T)
    print("Task can performed: ", maxT)

    print("Finished working on U = ", U)
    print("=================================\n")

# Configure the font size for the figure
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

Y_rand = np.array(Y_rand) * 100
Y_local = np.array(Y_local) * 100
Y_off = np.array(Y_off) * 100

plt.figure()
plt.plot(X, Y_rand, 'rv-', label='Random Search Optimal Computing')
plt.plot(X, Y_local, 'bs--', label='Entire Local Computing')
plt.plot(X, Y_off, 'g^--', label='Entire MBS/SBS Computing')

plt.xticks(X)
plt.grid(color='k', linestyle='--', linewidth=0.2)
plt.xlabel('Number of Devices')
plt.ylabel('Percentage of Successful Tasks')
plt.legend(loc='best')
plt.show()
