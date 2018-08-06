# Credit to https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

# NEURAL NETWORK IMPLEMENTATION
tf.reset_default_graph()

# Feature vector for current state representation
input1 = tf.placeholder(shape=[1, env.observation_space.n], dtype=tf.float32)

# tf.Variable(<initial-value>, name=<optional-name>)
# tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

# Weighting W vector in range 0 - 0.01 (like the way Andrew Ng did with *0.01
W = tf.Variable(tf.random_uniform([env.observation_space.n, env.action_space.n], 0, 0.01))

# Qout with shape [1, env.action_space.n] - Action state value for Q[s, a] with every a available at a state
Qout = tf.matmul(input1, W)

# Greedy action at a state
predict = tf.argmax(Qout, axis=1)

# Feature vector for next state representation
nextQ = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)

# Entropy loss
loss = tf.reduce_sum(tf.square(Qout - nextQ))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# TRAIN THE NETWORK
init = tf.global_variables_initializer()

# Set learning parameters
y = 0.99
e = 0.1
number_episodes = 2000

# List to store total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(number_episodes):
        print("Episode #{} is running!".format(i))

        # First state
        s = env.reset()

        rAll = 0
        d = False
        j = 0
        # Q network
        while j < 200:  # or While not d:
            j += 1
            # Choose action by epsilon (e) greedy
            # print("s = ", s," --> Identity s:s+1: ", np.identity(env.observation_space.n)[s:s+1])

            # s = 0 --> Identity s: s + 1:  [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
            # s = 1 --> Identity s: s + 1:  [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
            # Identity [s:s+1] is a one-hot vector
            # Therefore W is the actual Q value
            a, allQ = sess.run([predict, Qout], feed_dict={input1: np.identity(env.observation_space.n)[s:s+1]})

            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            s1, r, d, _ = env.step(a[0])
            # Obtain next state Q value by feeding the new state throughout the network
            Q1 = sess.run(Qout, feed_dict={input1: np.identity(env.observation_space.n)[s1:s1+1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1

            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={input1: np.identity(env.observation_space.n)[s:s+1], nextQ: targetQ})
            rAll += r
            s = s1
            if d:
                e = 1./((i/50) + 10)
                break

        jList.append(j)
        rList.append(rAll)

env.close()

plt.figure()
plt.plot(rList, label="Return - Q Learning")
plt.show()

plt.figure()
plt.plot(jList, label="Steps - Q Learning")
plt.show()

# -------------------------------------------------------------------------

# TABULAR IMPLEMENTATION
#
# # Set learning parameters
# lr = 0.8
# y = 0.95
# number_episodes = 20000
#
# # Initial table with all zeros
# Q = np.zeros([env.observation_space.n, env.action_space.n])
#
# # List of reward and steps per episode
# rList = []
# for i in range (number_episodes):
#     print("Episode #{} is running!".format(i))
#     s = env.reset()
#     rAll = 0
#     d = False
#     j = 0
#     while j < 99:
#         j += 1
#         # Choose an action by greedily (with noise) picking from Q table
#         # Because of the noise, it is epsilon-greedy with epsilon decreasing over time
#         a = np.argmax(Q[s, :] + np.random.rand(1, env.action_space.n)*(1./(i + 1)))
#         s1, r, d, _ = env.step(a)
#         # env.render()
#
#         # Update Q table with new knowledge
#         Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
#         rAll += r
#         s = s1
#         if d:
#             break
#     rList.append(rAll)




