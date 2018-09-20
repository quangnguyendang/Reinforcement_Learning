import gym
import numpy as np
import tensorflow as tf
import random
from collections import namedtuple
from matplotlib import pyplot as plt

# ---------- HYPER PARAMETERS -------------
ENV_NAME = "Breakout-v0"
env = gym.envs.make(ENV_NAME)

BATCH_SIZE = 16
BATCH_ACTION = 1
UPDATE_AFTER_NUMBER_OF_INPUT = 32

N_ACTION = env.action_space.n

REPEATING_ACTION_FRAME = 4

N_EPS = 500

REPLAY_BUFFER_SIZE = 500000
REPLAY_BUFFER_INIT_SIZE = 1000

DISCOUNT_FACTOR = 0.99
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 500000

LEARNING_RATE = 0.00025

TO = 0.01
GRADIENT_CLIPPING_NORM = 5
REWARD_CLIPPING_VALUE = 1

DISPLAY_AFTER_EPS = [1000, 1500, 2000, 2500, 3000, 4000]

# --------------------DEEP Q-NETWORK----------------------
class Q_Network():
    def __init__(self, scope):
        # Q network
        # tf.reset_default_graph()
        self.scope = scope
        # Build Model
        self._model_build()

    def _model_build(self):
        with tf.variable_scope(self.scope):
            # Inputs are 4 image frames with shape 84x84
            self.X = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
            self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
            self.isTraining = tf.placeholder(dtype=tf.bool, name="isTraining")
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            X = tf.to_float(self.X) / 255.0
            batch_size = tf.shape(self.X)[0]

            # CNN with Batchnorm
            conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=None)
            conv1_bn = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=self.isTraining)
            h1 = tf.nn.relu(conv1_bn, 'relu')
            conv2 = tf.contrib.layers.conv2d(h1, 64, 4, 2, activation_fn=None)
            conv2_bn = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=self.isTraining)
            h2 = tf.nn.relu(conv2_bn, 'relu')
            conv3 = tf.contrib.layers.conv2d(h2, 64, 3, 1, activation_fn=None)
            conv3_bn = tf.contrib.layers.batch_norm(conv3, center=True, scale=True, is_training=self.isTraining)
            h3 = tf.nn.relu(conv3_bn, 'relu')

            # Fully Connected Layers
            flattened = tf.contrib.layers.flatten(h3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=None)
            fc1_bn = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=self.isTraining)
            fc_act = tf.nn.relu(fc1_bn, 'relu')
            self.predictions = tf.contrib.layers.fully_connected(fc_act, N_ACTION)



            # Original Convolution layers
            # conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
            # conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            # conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

            # Original Fully connected layers
            # flattened = tf.contrib.layers.flatten(conv3)
            # fc1 = tf.contrib.layers.fully_connected(flattened, 512)
            # self.predictions = tf.contrib.layers.fully_connected(fc1, N_ACTION)


            # Q value for action-state pairs
            # [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, **2**, 3]
            # 4 batches --> tf.range = [0,1,2,3]
            # After reshape to 1 dimension array --> if want to find Q_value for a action 1
            # --> index of selection action is 4 * (n_batch of action - 1) + actions
            # Example: action 2 @ batch 3 ---> index = (3-1)*4 + 2 = 10 (remember index counted from 0)

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions
            self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

            # Calculate the loss
            self.losses = tf.squared_difference(self.y, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, GRADIENT_CLIPPING_NORM) for gradient in gradients]
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))

            # gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            # gradients, _ = tf.clip_by_value(gradients, -5, 5)
            # self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))

            # # Original Optimizer
            # self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.99, 0.0, 1e-6)
            # self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X: s, self.isTraining: 0})

    def update(self, sess, s, a, y):
        _, _, loss = sess.run([tf.train.get_global_step(), self.train_op, self.loss], {self.X: s, self.y: y, self.actions: a, self.isTraining: 1})
        return loss

# --------------------UTILITY FUNCTION----------------------
class downsample_and_convertGrayscale:
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output_state = tf.image.rgb_to_grayscale(self.input_state)
            self.output_state = tf.image.crop_to_bounding_box(self.output_state, 34, 0, 160, 160)
            self.output_state = tf.image.resize_images(
                self.output_state, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output_state = tf.squeeze(self.output_state)

    def process(self, sess, s):
        return sess.run(self.output_state, {self.input_state: s})

# For Testing....

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

e = Q_Network(scope="test")
sp = downsample_and_convertGrayscale()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Example observation batch
    observation = env.reset()

    observation_p = sp.process(sess, observation)
    observation = np.stack([observation_p] * 4, axis=2)
    observations = np.array([observation] * 2)

    # Test Prediction
    print(e.predict(sess, observations))

    # Test training step
    y = np.array([10.0, 10.0])
    a = np.array([1, 3])
    print(e.update(sess, observations, a, y))

print("Testing ok !")

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
        self.epsilon_end = epsilon_decay_steps

        self.q_estimator = Q_Network(scope="estimator")
        self.target_estimator = Q_Network(scope="target")

        self.to_coef = tf.constant(TO, dtype=tf.float32)
        self.to_comp = tf.constant(1 - TO, dtype=tf.float32)

    def update(self, sess,  s, a, r, s_, done, mode="DQN_LEARNER"):
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
                target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * q_values_next_target[np.arange(BATCH_SIZE), best_actions]

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
        action_probs = self.epsilon_greedy_policy(sess=sess, q_network=self.q_estimator, observation=state, epsilon=epsilon_t)
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
losses = []
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
    process_frame = downsample_and_convertGrayscale()

    # BUILD REPLAY MEMORY
    count = 0
    done = True
    s = None
    while count < REPLAY_BUFFER_INIT_SIZE:
        if done:
            s = env.reset()
            s = process_frame.process(sess, s)
            s = np.stack([s] * 4, axis=2)
            done = False

        count = count + 1
        a = env.action_space.sample()
        s_, r, done, _ = env.step(a)
        s_ = process_frame.process(sess, s_)
        s_ = np.append(s[:, :, 1:], np.expand_dims(s_, 2), axis=2)
        r = np.clip(r, 0, REWARD_CLIPPING_VALUE)

        agent.update(sess, s, a, r, s_, done, mode="REPLAY_BUILDER")
        s = s_
        if count % 10000 == 0:
            print("- Built {}% of Replay Buffer".format(int(100.0*count/REPLAY_BUFFER_INIT_SIZE)))
    print("Finished building REPLAY BUFFER!")
    print("-----------------------------------")

    # LEARN WITH DQN
    s = None
    i = 0
    ep = 0

    while ep < N_EPS:
        ep = ep + 1

        s = env.reset()
        s = process_frame.process(sess, s)
        s = np.stack([s] * 4, axis=2)

        done = False
        t = 0
        total_reward = 0
        while not done:
            a = agent.action_selection(sess, s, i)

            for j in range(REPEATING_ACTION_FRAME):
                i = i + 1
                t = t + 1
                s_, r, done, _ = env.step(a)

                r = np.clip(r, 0, REWARD_CLIPPING_VALUE)

                s_ = process_frame.process(sess, s_)
                s_ = np.append(s[:, :, 1:], np.expand_dims(s_, 2), axis=2)

                loss = agent.update(sess, s, a, r, s_, done, mode="DQN_LEARNER")

                s = s_
                total_reward += r
                if done:
                    break

        total_reward_all_eps.append(total_reward)
        total_steps_all_eps.append(t)

        if ep % 1 == 0:
            print("- Episode #{}, Frame #{}, total_step = {}, total_reward = {}".format(ep, i, t + 1, total_reward))

        # ---------- Display Intermediate Results ----------
        for k in range(len(DISPLAY_AFTER_EPS)):
            if ep > DISPLAY_AFTER_EPS[k] and DISPLAY_CHECK[k]:
                plot_reward(total_reward_all_eps)
                DISPLAY_CHECK[k] = False

env.close()

plot_reward(total_reward_all_eps)
