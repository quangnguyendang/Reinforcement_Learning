import gym
import numpy as np
import tensorflow as tf
import random
from matplotlib import pyplot as plt


# ------------------------- Policy Neural Network -------------------------
class PolicyNetwork:
    def __init__(self, scope, n_action, learning_rate=0.01, sess=tf.get_default_session()):
        self.scope = scope
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.sess = sess

        init = tf.initialize_all_variables()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        self.sess.run([init, init_g, init_l])

        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[2], name="state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # state_one_hot = tf.contrib.layers.flatten(tf.one_hot(self.state, self.n_action))
            self.output_layer = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.state, 0),
                                                                  num_outputs=self.n_action,
                                                                  activation_fn=None,
                                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

            # Squeeze: Removes dimensions of size 1 from a tensor
            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.selected_action_prob = tf.gather(self.action_probs, self.action)

            # Loss
            self.loss = -tf.log(self.selected_action_prob) * self.target
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state):
        feed_dict = {self.state: state}
        return self.sess.run(self.action_probs, feed_dict)

    def update(self, state, target, action):
        feed_dict = {self.state: state, self.action: action, self.target: target}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss


# ------------------------- Value Neural Network -------------------------
class ValueNetwork:
    def __init__(self, scope, learning_rate=0.01, sess=tf.get_default_session()):
        self.scope = scope
        self.learning_rate = learning_rate
        self.sess = sess

        init = tf.initialize_all_variables()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        self.sess.run([init, init_g, init_l])

        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[2], name="state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.output_layer = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.state, 0),
                                                                  num_outputs=1,
                                                                  activation_fn=None,
                                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

            self.value_estimate = tf.contrib.layers.flatten(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state):
        feed_dict = {self.state: state}
        return self.sess.run(self.value_estimate, feed_dict)

    def update(self, state, target):
        feed_dict = {self.state: state, self.target:self.target}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss


# --------------------------- Actor-Critic Agent -----------------------------
def actor_critic_agent(env, P_estimator, V_estimator, number_episodes, discount_factor=0.99):
    # Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    total_steps_needed = []
    for i_episode in range(number_episodes):
        state = env.reset()
        done = False
        t = 0
        while not done:
            action_probs = P_estimator.predict(state)
            print("ACTION PROB = ", action_probs)
            a = np.random.choice(np.array(action_probs).shape[0], p=action_probs)
            s_, r, done, _ = env.step(a)

            print("State [{}], Reward [{}], done = {}".format(s_, r, done))

            t = t + 1
            v_ = V_estimator.predict(s_)
            td_target = r + discount_factor * v_
            td_error = td_target - V_estimator.predict(state)

            V_estimator.update(state, td_target)
            P_estimator.update(state, td_error, a)

            state = s_

            if done:
                print('Episode #{}, need {} steps.'.format(i_episode, t))
                total_steps_needed.append(t)

    return total_steps_needed


# -------------------------- MAIN FUNCTION -----------------------------
env = gym.envs.make("MountainCar-v0")
print(np.array(env.reset()).shape)
init = tf.initialize_all_variables()
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init_g)
    sess.run(init_l)
    tf.initialize_all_variables().run()
    sess.run(init)
    P_estimator = PolicyNetwork("Policy", env.action_space.n, learning_rate=0.001, sess=sess)
    V_estimator = ValueNetwork("Value", learning_rate=0.001, sess=sess)
    total_steps_needed = actor_critic_agent(env, P_estimator, V_estimator, number_episodes=100)

plt.plot(total_steps_needed, label="1-Step Actor-Critic")
plt.xlabel('episode')
plt.ylabel('number of needed steps')
plt.legend(loc='best')
plt.show()

