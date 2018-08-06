# Follow instruction in https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

import gym

env = gym.envs.make("BreakoutDeterministic-v4")

print("Action space size: {}".format(env.action_space.n))
print("Action space: {}".format(env.unwrapped.get_action_meanings()))

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

done = False
while not done:
    frame, reward, done, _ = env.step(env.action_space.sample())
    env.render('human')

env.close()
