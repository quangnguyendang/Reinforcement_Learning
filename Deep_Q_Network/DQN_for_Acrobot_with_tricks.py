import gym
import numpy as np
import tensorflow as tf
import random
from collections import namedtuple
from matplotlib import pyplot as plt

# ENVIRONMENT INFORMATION
ENV_NAME = "Acrobot-v1"
env = gym.envs.make(ENV_NAME)

# N_ACTION =  3
# Sample Action:  0
# Observation sample:  [ 0.99999216  0.00396017  0.99584838 -0.09102746  0.02481395  0.00106494]
# Observation higher limit:  [ 1.        1.        1.        1.       12.566371 28.274334]
# Observation lower limit:  [ -1.        -1.        -1.        -1.       -12.566371 -28.274334]

# ---------- HYPER PARAMETERS -------------
BATCH_SIZE = 128
BATCH_ACTION = 1
UPDATE_AFTER_NUMBER_OF_INPUT = 100

N_ACTION = env.action_space.n

REPEATING_FRAME = 1
STACKED_FRAME_NUMBER = 1

N_EPS = 12000

REPLAY_BUFFER_SIZE = 500000
REPLAY_BUFFER_INIT_SIZE = 1000

DISCOUNT_FACTOR = 0.99
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 500000
EPSILON_END_STEP = 500000

LEARNING_RATE = 0.001

TO = 0.01


DISPLAY_AFTER_EPS = [1000, 1300, 1800, 2000, 3000, 4000, 5000, 6000, 8000, 10000]

# --------------------DEEP Q-NETWORK----------------------
class Q_Network():
    def __init__(self, scope):
        self.scope = scope
        self._model_build()

    def _model_build(self):
        with tf.variable_scope(self.scope):
            self.X = tf.placeholder(shape=[None, 6, STACKED_FRAME_NUMBER], dtype=tf.float32, name="X")
            self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            X = tf.to_float(self.X)
            batch_size = BATCH_SIZE

            # Fully connected layers
            flattened = tf.contrib.layers.flatten(X)
            fc1 = tf.contrib.layers.fully_connected(flattened, 128)
            fc2 = tf.contrib.layers.fully_connected(fc1, 64)
            self.predictions = tf.contrib.layers.fully_connected(fc2, N_ACTION, activation_fn=None)

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions
            self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

            # Calculate the loss
            self.losses = tf.squared_difference(self.y, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X: s})

    def update(self, sess, s, a, y):
        _, _, loss = sess.run([tf.train.get_global_step(), self.train_op, self.loss],
                              {self.X: s, self.y: y, self.actions: a})
        return loss


# --------------------DQN AGENT----------------------
class DQN_Agent:
    def __init__(self, env, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps):
        # Tuple to update Replay Buffer
        self.Transition = namedtuple("Transition", ['s', 'a', 'r', 's_', 'done'])

        self.n_actions = N_ACTION
        self.n_update = 0
        self.replay_buffer = []

        self.env = env
        self.discount_factor = discount_factor

        self.epsilon = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_end = EPSILON_END_STEP

        self.q_estimator = Q_Network(scope="estimator")
        self.target_estimator = Q_Network(scope="target")

        self.to_coef = tf.constant(TO, dtype=tf.float32)
        self.to_comp = tf.constant(1 - TO, dtype=tf.float32)

    def update(self, sess, s, a, r, s_, done, mode="DQN_LEARNER"):
        # print('UPDATING REPLAY BUFFER!')
        if len(self.replay_buffer) > REPLAY_BUFFER_SIZE:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(self.Transition(s, a, r, s_, done))

        loss = 0

        if mode == "DQN_LEARNER":
            # Update Estimator Network
            # print('UPDATING ESTIMATOR NETWORK!')
            self.n_update = self.n_update + 1
            if self.n_update % BATCH_ACTION == 0:
                minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*minibatch))

                # DQN
                q_values_next = self.q_estimator.predict(sess, next_state_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = self.target_estimator.predict(sess, next_state_batch)
                target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * \
                               q_values_next_target[np.arange(BATCH_SIZE), best_actions]

                state_batch = np.array(state_batch)
                loss = self.q_estimator.update(sess, state_batch, action_batch, target_batch)

            if self.n_update % UPDATE_AFTER_NUMBER_OF_INPUT == 0:
                self.great_update(sess, self.q_estimator, self.target_estimator)

        return loss

    def epsilon_greedy_policy(self, sess, q_network, observation, epsilon):
        action_probs = np.ones(N_ACTION, dtype=float) * 1.0 * epsilon / N_ACTION
        q_values = q_network.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        action_probs[best_action] = action_probs[best_action] + 1 - epsilon
        return action_probs

    def action_selection(self, sess, state, t):
        # DQN
        epsilon_t = self.epsilon[min(t, self.epsilon_end - 1)]
        action_probs = self.epsilon_greedy_policy(sess=sess, q_network=self.q_estimator, observation=state,
                                                  epsilon=epsilon_t)
        action = np.random.choice(np.arange(N_ACTION), p=action_probs)
        return action

    def great_update(self, sess, estimator1, estimator2):
        # Update Target Network with parameters from q estimator network
        # print('UPDATING TARGET NETWORK!')

        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            p1 = tf.multiply(e1_v, self.to_coef)
            p2 = tf.multiply(e2_v, self.to_comp)
            p3 = tf.add(p1, p2)
            # op = e2_v.assign(e1_v)
            op = e2_v.assign(p3)
            update_ops.append(op)

        sess.run(update_ops)

# -------------------- UTILITY --------------------------
def plot_reward(total_reward_all_eps):
    for window_width in [1, 100, 500, 1000]:
        cumsum_vec = np.cumsum(np.insert(total_reward_all_eps, 0, 0))
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

        plt.figure()
        plt.plot(ma_vec, label="DQN for ATARI Games - Breakout")
        plt.xlabel('Episode')
        plt.ylabel('Moving-Average Rewards - window = ' + str(window_width))
        plt.legend(loc='best')
        plt.show()

# --------------------MAIN FUNCTION----------------------
total_reward_all_eps = []
total_steps_all_eps = []
DISPLAY_CHECK = [True, True, True, True, True, True, True, True, True, True]

tf.reset_default_graph()
# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
agent = DQN_Agent(env=env,
                  discount_factor=DISCOUNT_FACTOR,
                  epsilon_start=EPSILON_START,
                  epsilon_end=EPSILON_END,
                  epsilon_decay_steps=EPSILON_DECAY_STEPS)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # BUILD REPLAY MEMORY
    count = 0
    done = True
    s = None
    while count < REPLAY_BUFFER_INIT_SIZE:
        if done:
            s = env.reset()
            s = np.stack([s] * STACKED_FRAME_NUMBER, axis=1)
            done = False

        count = count + 1
        a = env.action_space.sample()
        s_, r, done, _ = env.step(a)

        s_ = np.append(s[:, 1:], np.expand_dims(s_, 1), axis=1)

        agent.update(sess, s, a, r, s_, done, mode="REPLAY_BUILDER")
        s = s_
        if count % 10000 == 0:
            print("- Built {}% of Replay Buffer".format(int(100.0 * count / REPLAY_BUFFER_INIT_SIZE)))
    print("Finished building REPLAY BUFFER!")
    print("-----------------------------------")

    # LEARN WITH DQN
    s = None
    i = 0
    ep = 0
    while ep < N_EPS:
        ep = ep + 1

        s = env.reset()
        s = np.stack([s] * STACKED_FRAME_NUMBER, axis=1)

        done = False
        t = 0
        total_reward = 0
        while not done:
            a = agent.action_selection(sess, s, i)
            # print("Take Action A = ", a)
            for j in range(REPEATING_FRAME):
                i = i + 1
                t = t + 1
                s_, r, done, _ = env.step(a)
                # env.render()

                s_ = np.append(s[:, 1:], np.expand_dims(s_, 1), axis=1)

                loss = agent.update(sess, s, a, r, s_, done, mode="DQN_LEARNER")

                s = s_
                total_reward += r

                if done:
                    break

        total_reward_all_eps.append(total_reward)
        total_steps_all_eps.append(t)
        if ep % 5 == 0:
            print("- Episode #{}, Frame #{}, total_step = {}, total_reward = {}".format(ep, i, t + 1, total_reward))

        # ---------- Display Intermediate Results ----------
        for k in range(len(DISPLAY_AFTER_EPS)):
            if ep > DISPLAY_AFTER_EPS[k] and DISPLAY_CHECK[k]:
                plot_reward(total_reward_all_eps)
                DISPLAY_CHECK[k] = False

env.close()

plot_reward(total_reward_all_eps)
