# Follow tut on https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

import gym
import numpy as np
import tensorflow as tf
import random
from collections import namedtuple
from matplotlib import pyplot as plt
import os

# ---------- HYPER PARAMETERS -------------
ENV_NAME = "Pendulum-v0"
env = gym.envs.make(ENV_NAME)

BATCH_SIZE = 64

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
ACTION_BOUND = env.action_space.high[0]

N_EPS = 2000

REPLAY_BUFFER_SIZE = 1000000
REPLAY_BUFFER_INIT_SIZE = 50000

DISCOUNT_FACTOR = 0.99

LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001

TO = 0.001

DISPLAY_AFTER_EPS = [200, 250, 300, 400, 500, 600, 800, 1000]

# ------------------------ Actor Network --------------------------------
class Actor_Network():
    def __init__(self, scope, start_param_index=0):
        self.scope = scope
        self._model_build()
        # Determine the network coeff in trainable variable of tensorflow to perform fast soft copy
        self.start_param_index = start_param_index
        self.end_param_index = start_param_index + len(tf.trainable_variables()[start_param_index:]) - 1

    def _model_build(self):
        with tf.variable_scope(self.scope):
            self.X = tf.placeholder(shape=[None, STATE_DIM], dtype=tf.float32, name="X")
            self.isTraining = tf.placeholder(dtype=tf.bool, name='isTraining')

            flattened = tf.contrib.layers.flatten(self.X)

            #   Layer with --- BATCHNORM

            fc1 = tf.contrib.layers.fully_connected(flattened, 400, activation_fn=None)
            fc1 = tf.contrib.layers.batch_norm(fc1, is_training=self.isTraining)
            fc1 = tf.nn.relu(fc1)

            #   Layer with --- BATCHNORM
            fc2 = tf.contrib.layers.fully_connected(fc1, 300, activation_fn=None)
            fc2 = tf.contrib.layers.batch_norm(fc2, is_training=self.isTraining)
            fc2 = tf.nn.relu(fc2)

            out = tf.contrib.layers.fully_connected(fc2, ACTION_DIM, weights_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3), activation_fn=tf.nn.tanh)

            # Scale output to -action_bound to action_bound
            self.predictions = tf.multiply(out, ACTION_BOUND)

            self.network_params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]

            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, ACTION_DIM])

            # Combine the gradients here
            self.unnormalized_actor_gradients = tf.gradients(self.predictions, self.network_params, -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x, BATCH_SIZE), self.unnormalized_actor_gradients))

            # Optimization Op
            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE_ACTOR).apply_gradients(zip(self.actor_gradients, self.network_params))


    def predict(self, sess, state):
        return sess.run(self.predictions, {self.X: state, self.isTraining: False})

    def update(self, sess, state, a_grad):
        sess.run([self.train_op],
                 {self.X: state, self.action_gradient: a_grad, self.isTraining: True})

    def get_param_index(self):
        return [self.start_param_index, self.end_param_index]

# ------------------------ Critic Network -------------------------------
class Critic_Network():
    def __init__(self, scope, start_param_index=0):
        self.scope = scope
        self._model_build()
        # Determine the network coeff in trainable variable of tensorflow to perform fast soft copy
        self.start_param_index = start_param_index
        self.end_param_index = start_param_index + len(tf.trainable_variables()[start_param_index:]) - 1

    def _model_build(self):
        with tf.variable_scope(self.scope):
            self.X = tf.placeholder(shape=[None, STATE_DIM], dtype=tf.float32, name="X")
            self.actions = tf.placeholder(shape=[None, ACTION_DIM], dtype=tf.float32, name="actions")
            self.isTraining = tf.placeholder(dtype=tf.bool, name='isTraining')

            # Layers with --- BATCHNORM
            flattened = tf.contrib.layers.flatten(self.X)
            fc1 = tf.contrib.layers.fully_connected(flattened, 400, activation_fn=None)
            fc1 = tf.contrib.layers.batch_norm(fc1, is_training=self.isTraining)
            fc1 = tf.nn.relu(fc1)

            t1 = tf.contrib.layers.fully_connected(fc1, 300, activation_fn=None)
            t2 = tf.contrib.layers.fully_connected(self.actions, 300, activation_fn=None)

            t1_W = tf.get_default_graph().get_tensor_by_name(os.path.split(t1.name)[0] + '/weights:0')
            t2_W = tf.get_default_graph().get_tensor_by_name(os.path.split(t2.name)[0] + '/weights:0')
            t2_b = tf.get_default_graph().get_tensor_by_name(os.path.split(t1.name)[0] + '/biases:0')

            # Layers with --- BATCHNORM
            fc2 = tf.matmul(fc1, t1_W) + tf.matmul(self.actions, t2_W) + t2_b
            # fc2 = tf.contrib.layers.batch_norm(fc2, is_training=self.isTraining)
            fc2 = tf.nn.relu(fc2)

            self.predictions = tf.contrib.layers.fully_connected(fc2, 1, weights_initializer=tf.initializers.random_uniform(minval=-3e-4, maxval=3e-4), activation_fn=None)

            # Network target (y_i)
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

            # Define loss and optimization Op
            self.losses = tf.squared_difference(self.predicted_q_value, self.predictions)
            self.loss = tf.reduce_mean(self.losses)
            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE_CRITIC).minimize(self.loss)

            # Get the gradient of the net w.r.t. the action
            self.action_grads = tf.gradients(self.predictions, self.actions)

    def predict(self, sess, state, action):
        return sess.run(self.predictions, {self.X: state, self.actions: action, self.isTraining: False})

    def update(self, sess, state, action, predicted_q_value):
        return sess.run([self.predictions, self.train_op], {self.X: state, self.actions: action, self.predicted_q_value: predicted_q_value, self.isTraining: True})

    def get_action_gradients(self, sess, state, action):
        return np.array(sess.run([self.action_grads], {self.X: state, self.actions: action, self.isTraining: False})).reshape([BATCH_SIZE, 1])

    def get_param_index(self):
        return [self.start_param_index, self.end_param_index]


# ------------------------ Ornstein Uhlenbeck Action Noise --------------
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# --------------------DDPG Agent-------------------------
class DDPG_Agent():
    def __init__(self):
        self.Transition = namedtuple("Transition", ['s', 'a', 'r', 's_', 'done'])
        self.replay_buffer = []

        self.actor_noise = self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ACTION_DIM))

        self.critic_estimator = Critic_Network(scope="estimator_critic", start_param_index=0)
        self.critic_target = Critic_Network(scope="target_critic", start_param_index=self.critic_estimator.get_param_index()[1] + 1)

        self.actor_estimator = Actor_Network(scope="estimator_actor", start_param_index=self.critic_target.get_param_index()[1] + 1)
        self.actor_target = Actor_Network(scope="target_actor", start_param_index=self.actor_estimator.get_param_index()[1] + 1)

        print("CRITIC ESTIMATOR PARAMS  : ", self.critic_estimator.get_param_index())
        print("CRITIC TARGET PARAMS     : ", self.critic_target.get_param_index())

        print("ACTOR ESTIMATOR PARAMS   : ", self.actor_estimator.get_param_index())
        print("ACTOR TARGET PARAMS      : ", self.actor_target.get_param_index())

        self.update_network_params = []
        for i in np.arange(self.critic_target.get_param_index()[0], self.critic_target.get_param_index()[1] + 1):
            self.update_network_params.append(tf.trainable_variables()[i].assign(tf.multiply(tf.trainable_variables()[i], 1 - TO) + tf.multiply(tf.trainable_variables()[i - self.critic_target.get_param_index()[0] + self.critic_estimator.get_param_index()[0]], TO)))

        for i in np.arange(self.actor_target.get_param_index()[0], (self.actor_target.get_param_index()[1]) + 1):
            self.update_network_params.append(tf.trainable_variables()[i].assign(tf.multiply(tf.trainable_variables()[i], 1 - TO) + tf.multiply(tf.trainable_variables()[i - self.actor_target.get_param_index()[0] + self.actor_estimator.get_param_index()[0]], TO)))

    def action_select(self, sess, state):
        acts = self.actor_estimator.predict(sess, np.array(state).reshape([1, STATE_DIM])) + self.actor_noise()[0]
        return acts[0]

    def update(self, sess, s, a, r, s_, done, learning = True, step_count=0):
        # UPDATING REPLAY BUFFER
        if len(self.replay_buffer) > REPLAY_BUFFER_SIZE:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(self.Transition(s, a, r, s_, done))

        if learning:
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*minibatch))

            target_q = self.critic_target.predict(sess, next_state_batch, self.actor_target.predict(sess, next_state_batch))
            y_i = []
            for k in range(BATCH_SIZE):
                if done_batch[k]:
                    y_i.append(reward_batch[k])
                else:
                    y_i.append(reward_batch[k] + DISCOUNT_FACTOR * target_q[k])

            # ----- Update Critic Network
            y_i = np.array(y_i).reshape((BATCH_SIZE, 1))
            self.critic_estimator.update(sess, state_batch, action_batch, y_i)

            # ----- Update Actor Network
            action_outs = self.actor_estimator.predict(sess, state_batch)
            grads = self.critic_estimator.get_action_gradients(sess, state_batch, action_outs)
            self.actor_estimator.update(sess, state_batch, grads)

            # ----- Soft Copy for Target Network Update
            sess.run(self.update_network_params)


# -------------------- UTILITY --------------------------
def plot_reward(total_reward_all_eps):
    for window_width in [1, 50, 100, 200]:
        cumsum_vec = np.cumsum(np.insert(total_reward_all_eps, 0, 0))
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

        plt.figure()
        plt.plot(ma_vec, label="DQN for " + ENV_NAME)
        plt.xlabel('Episode')
        plt.ylabel('Moving-Average Rewards - window = ' + str(window_width))
        plt.legend(loc='best')
        plt.show()


# ------------------------- MAIN ------------------------
total_reward_all_eps = []
display_check = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]

tf.reset_default_graph()
global_step = tf.Variable(0, name='global_step', trainable=False)

agent = DDPG_Agent()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # ----- BUILD REPLAY MEMORY
    count = 0
    done = True
    while count < REPLAY_BUFFER_INIT_SIZE:
        if done:
            s = env.reset()
            done = False

        count = count + 1
        a = env.action_space.sample()
        s_, r, done, _ = env.step(a)

        agent.update(sess, s, a, r, s_, done, learning=False)
        s = s_

        if count % 10000 == 0:
            print("- Built {}% of Replay Buffer".format(int(100.0 * count / REPLAY_BUFFER_INIT_SIZE)))

    print("Finished building initial REPLAY BUFFER!")
    print("-----------------------------------")

    # ----- DDPG LEARNING
    frame_count = 0
    for ep_i in range(N_EPS):
        total_reward = 0
        s = env.reset()
        done = False
        step_count = 1
        while not done:
            a = agent.action_select(sess, state=s)
            s_, r, done, _ = env.step(a)
            agent.update(sess, s, a, r, s_, done, learning=True, step_count=step_count)
            total_reward += r
            frame_count += 1
            # print("Ep = {}, step = {}, s = {}, a = {}, r = {}, s_ = {}, done = {}".format(ep_i, step_count, s, a, r, s_, done))
            s = s_
            step_count += 1

        total_reward_all_eps.append(total_reward)
        if ep_i % 1 == 0:
            print("- Episode #{}, Frame #{}, total_step = {}, total_reward = {}".format(ep_i, frame_count, step_count, total_reward))

        # ---------- Display Intermediate Results ----------
        for k in range(len(DISPLAY_AFTER_EPS)):
            if ep_i > DISPLAY_AFTER_EPS[k] and display_check[k]:
                plot_reward(total_reward_all_eps)
                display_check[k] = False

env.close()
plot_reward(total_reward_all_eps)