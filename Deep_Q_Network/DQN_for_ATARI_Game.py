import gym
import numpy as np
import tensorflow as tf
import random
from collections import namedtuple
from matplotlib import pyplot as plt


BATCH_SIZE = 32
BATCH_ACTION = 4
UPDATE_AFTER_NUMBER_OF_INPUT = 10000
N_ACTION = 4

N_EPISODE = 4000

REPLAY_BUFFER_SIZE = 500000
REPLAY_BUFFER_INIT_SIZE = 50000

# --------------------DEEP Q-NETWORK----------------------
class Q_Network:
    def __init__(self, scope):
        # Q network
        # tf.reset_default_graph()
        self.scope = scope
        # Build Model
        with tf.variable_scope(scope):
            self._model_build()

    def _model_build(self):
        # Inputs are 4 image frames with shape 84x84
        self.X = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        X_float = tf.to_float(self.X) / 255

        # CNN
        conv1 = tf.contrib.layers.conv2d(X_float, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully Connected Layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, N_ACTION)

        # Q value for action-state pairs
        # [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, **2**, 3]
        # 4 batches --> tf.range = [0,1,2,3]
        # After reshape to 1 dimension array --> if want to find Q_value for a action 1
        # --> index of selection action is 4 * (n_batch of action - 1) + actions
        # Example: action 2 @ batch 3 ---> index = (3-1)*4 + 2 = 10 (remember index counted from 0)
        gather_indices = tf.range(BATCH_SIZE) * tf.shape(self.predictions)[1] + self.actions
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X: s})

    def update(self, sess, s, a, y):
        _, loss = sess.run([self.train_op, self.loss],
                           {self.X: s, self.y: y, self.actions: a})
        return loss

# --------------------DQN AGENT----------------------
class DQN_Agent:
    def __init__(self, n_actions, env, sess, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps, process_frame):
        self.n_actions = n_actions
        self.n_update = 0
        self.replay_buffer = []
        self.env = env
        self.sess = sess
        self.process_frame = process_frame
        self.discount_factor = discount_factor
        self.epsilon = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_end = epsilon_end
        self.state = None
        self.reset()

        # Tuple to update Replay Buffer
        self.Updated_Tuple = namedtuple("Updated_Tuple", ['s', 'a', 'r', 's_', 'done'])

        self.q_estimator = Q_Network(scope="estimator")
        self.target_estimator = Q_Network(scope="target")

    def reset(self):
        self.state = self.process_frame.process(self.sess, self.env.reset())
        self.state = np.stack([self.state] * 4, axis=2)

    def update(self, s, a, r, s_, done, mode="DQN_LEARNER"):
        next_state = np.append(self.state[:, :, 1:], np.expand_dims(s_, 2), axis=2)
        if len(self.replay_buffer) > REPLAY_BUFFER_SIZE:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(self.Updated_Tuple(self.state, a, r, next_state, done))
        # self.replay_buffer = np.append(self.replay_buffer, self.Updated_Tuple(self.state, a, r, next_state, done))
        self.state = next_state

        loss = 0

        if mode == "DQN_LEARNER":
            # Update Estimator Network
            self.n_update = self.n_update + 1
            if self.n_update % BATCH_ACTION == 0:
                minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*minibatch))
                q_value_next = self.target_estimator.predict(self.sess, next_state_batch)
                target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * np.max(q_value_next, axis=1)
                state_batch = np.array(state_batch)
                loss = self.q_estimator.update(self.sess, state_batch, action_batch, target_batch)

            if self.n_update % UPDATE_AFTER_NUMBER_OF_INPUT == 0:
                self.great_update()

        return loss

    def epsilon_greedy_policy(self, q_network, state, epsilon):
        action_probs = np.ones(N_ACTION, dtype=float) * epsilon/N_ACTION
        q_values = q_network.predict(self.sess, np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)
        action_probs[best_action] += 1 - epsilon
        return action_probs

    def action_selection(self, t):
        action_probs = self.epsilon_greedy_policy(self.q_estimator, self.state, epsilon=self.epsilon[min(t, int(self.epsilon_end - 1))])
        action = np.random.choice(N_ACTION, p=action_probs)
        return action

    def great_update(self):
        # Update Target Network with parameters from q estimator network
        param_est = [t for t in tf.trainable_variables() if t.name.startswith(self.q_estimator.scope)]
        param_est = sorted(param_est, key=lambda v: v.name)
        param_target = [t for t in tf.trainable_variables() if t.name.startswith(self.target_estimator.scope)]
        param_target = sorted(param_target, key=lambda v: v.name)

        update_ops = []
        for p1, p2 in zip(param_est, param_target):
            op = p2.assign(p1)
            update_ops.append(op)

        self.sess.run(update_ops)


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


# --------------------MAIN FUNCTION----------------------
env = gym.envs.make("BreakoutDeterministic-v4")

total_reward_all_eps = []
total_steps_all_eps = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    process_frame = downsample_and_convertGrayscale()

    agent = DQN_Agent(n_actions=env.action_space.n, sess=sess, env=env, discount_factor=0.99, epsilon_start=1,
                      epsilon_end=0.01, epsilon_decay_steps=10000, process_frame=process_frame)
    sess.run(tf.global_variables_initializer())

    # BUILD REPLAY MEMORY
    count = 0
    done = True
    s = None
    while count < REPLAY_BUFFER_INIT_SIZE:
        if done:
            s = process_frame.process(sess, env.reset())
            done = False
        count = count + 1
        a = env.action_space.sample()
        s_, r, done, _ = env.step(a)
        s_ = process_frame.process(sess, s_)
        agent.update(s, a, r, s_, done, mode="REPLAY_BUILDER")
        s = s_
    print("Finished building REPLAY BUFFER!")

    # LEARN WITH DQN
    s = process_frame.process(sess, env.reset())
    total_t = 0
    for i in range(N_EPISODE):
        agent.reset()
        done = False
        t = 0
        total_reward = 0
        while not done:
            t = t + 1
            a = agent.action_selection(total_t)
            s_, r, done, _ = env.step(a)
            s_ = process_frame.process(sess, s_)
            loss = agent.update(s, a, r, s_, done, mode="DQN_LEARNER")
            s = s_
            total_reward += r
            total_t = total_t + 1
        total_reward_all_eps.append(total_reward)
        total_steps_all_eps.append(t)
        print("i = {}, t = {}, total_reward = {}".format(i, t + 1, total_reward))


env.close()

plt.figure()
plt.plot(total_reward_all_eps, label="DQN for ATARI - Breakage")
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(total_steps_all_eps, label="DQN for ATARI - Breakage")
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
plt.legend(loc='best')
plt.show()

