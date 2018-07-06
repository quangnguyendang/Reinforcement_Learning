# Exploration and Exploitation with Multi-armed Bandit Problem
# N-Bandit Environment
# Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# https://www.youtube.com/watch?v=V8tu8omHs1U&t=12s

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym.envs.classic_control import rendering

# Our environment is a 10-arm bandit with stationary probabilities independent for each.
bandit_prob = [0.1, 0.2, 0.3, 0.15, 0.2, 0.05, 0.1, 0.2, 0.6, 0.3]


class NBanditEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'arb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.bandit_prob = np.array(bandit_prob)
        self.N = self.bandit_prob.shape[0]

        self.total_reward = 0
        self.reward = 0
        self.action = 0
        self.goal = float("inf")
        # self.goal = 500

        self.digitRadius = 20
        self.historySave = 5

        self.screen_width = 80 * self.N + 200
        self.screen_height = 80 * self.historySave + 200

        self.historyReward = np.zeros([self.N, self.historySave])

        self.viewer = None

        self.action_space = spaces.Discrete(self.N)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Action is the bandit # selection
        rand = np.random.random()
        # According to the pre-defined probability array, reward will be returned when an action is taken.
        reward = 1 if (rand < self.bandit_prob[action]) else 0

        self.total_reward += reward
        self.reward = reward
        self.action = action

        # Update the history reward array for rendering purpose for the bandit/action that is currently selected.

        self.historyReward[action][:-1] = self.historyReward[action][1:]
        self.historyReward[action][-1] = reward

        done = bool(self.total_reward >= self.goal)
        print("Reward = {}. Total reward = {}".format(self.reward, self.total_reward))

        self.render()

        return np.array(self.total_reward), reward, done, {}

    def reset(self):
        self.total_reward = 0
        return np.array(self.total_reward)

    # def clear_bandit(self, bandit=-1):
    #     # Clear drawing of bandit #bandit
    #     # Clear all if bandit parameter = -1
    #     if bandit == -1:
    #         clear_box = rendering.FilledPolygon(
    #             [(0, 0), (0, self.screen_height), (self.screen_width, self.screen_height), (self.screen_width, 0)])
    #     else:
    #         clear_box = rendering.make_circle(self.digitRadius + 20, 30, True)
    #     clear_box.set_color(1, 1, 1)
    #     clear_box.add_attr(self.numbertrans)
    #     self.viewer.add_geom(clear_box)

    def render(self, mode='human'):
        # Initiate the viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

        clearance = 10

        for i in range(0, self.N):
            for j in range(0, self.historySave):
                # Define the shift in position to transform the coordinate of the current drawing.
                display_pos_x = (self.screen_width / (self.N + 1)) * (i + 1)
                display_pos_y = (self.screen_height / (self.historySave + 2)) * (self.historySave - j + 1)
                circletrans = rendering.Transform()
                circletrans.set_translation(display_pos_x, display_pos_y)

                # Clear the drawing by adding a larger circle with background color.
                clear_box = rendering.make_circle(self.digitRadius + j*7, 30, True)
                clear_box.set_color(1, 1, 1)
                clear_box.add_attr(circletrans)
                self.viewer.add_geom(clear_box)

                # Draw the circle ball for each bandit.
                # Radius of the circle is changed to address the current/past reward illustration.
                number_0 = rendering.make_circle(self.digitRadius + j*4, 30, False)
                number_0.add_attr(rendering.Transform(translation=(0, clearance)))
                number_0.set_color(0, 0, 1)
                number_1 = rendering.make_circle(self.digitRadius + j*4, 30, True)
                number_1.add_attr(rendering.Transform(translation=(0, clearance)))
                number_1.set_color(0, 0.5, 1)

                if self.historyReward[i][j] == 0:
                    number_0.add_attr(circletrans)
                    self.viewer.add_geom(number_0)
                    if j == (self.historySave -1):
                        number_0.set_color(1, 0, 0)
                else:
                    number_1.add_attr(circletrans)
                    self.viewer.add_geom(number_1)
                    if j == (self.historySave -1):
                        number_1.set_color(0, 0, 1)

        # Clear the cursor area for updating the current cursor's position.
        display_pos_x = 0
        display_pos_y = (self.screen_height / (self.historySave + 2)) - self.digitRadius*3
        clear_box = rendering.make_polygon([(0, 0), (0, self.digitRadius * 5), (self.screen_width, self.digitRadius * 5),
                                        (self.screen_width, 0)])
        boxtrans = rendering.Transform()
        boxtrans.set_translation(display_pos_x, display_pos_y)
        clear_box.add_attr(boxtrans)
        clear_box.set_color(1, 1, 1)
        self.viewer.add_geom(clear_box)

        # Drawing the cursor to show which action/bandit is currently selected.
        display_pos_x = (self.screen_width / (self.N + 1)) * (self.action + 1) - self.digitRadius
        display_pos_y = (self.screen_height / (self.historySave + 2)) - self.digitRadius
        arrow = rendering.make_polygon([(0, 0), (0, self.digitRadius*2), (self.digitRadius*2, self.digitRadius*2), (self.digitRadius*2, 0)])
        boxtrans = rendering.Transform()
        boxtrans.set_translation(display_pos_x, display_pos_y)
        arrow.add_attr(boxtrans)
        arrow.set_color(1, 0, 0)
        self.viewer.add_geom(arrow)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()