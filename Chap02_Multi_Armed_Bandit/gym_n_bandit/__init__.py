from gym.envs.registration import register

register(
    id='n-bandit-v0',
    entry_point='gym_n_bandit.envs:NBanditEnv',
)
register(
    id='n-bandit-extrahard-v0',
    entry_point='gym_n_bandit.envs:NBanditExtraHardEnv',
)
