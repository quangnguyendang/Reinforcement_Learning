import gym
env = gym.make('MountainCar-v0')

n_action = env.action_space.n
print("n action = ", n_action)
print(env.observation_space.high)
print(env.observation_space.low)

for i_episode in range(20):
    state = env.reset()
    info = None
    for t in range(1000):
        # env.render()
        print(state.shape[0])
        print(state[0], "-----", state[1])
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break