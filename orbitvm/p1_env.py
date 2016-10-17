import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.registration import register

from orbitvm.p1 import P1

logger = logging.getLogger(__name__)

class OrbitP1Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        high2 = np.array([np.finfo(np.float32).max] * 2)
        high5 = np.array([np.finfo(np.float32).max] * 4)
        self.action_space = spaces.Box(-high2, high2)
        self.observation_space = spaces.Box(-high5, high5)
        self._seed()
        self.reset()
        self.viewer = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
            self.vm_input[0x3E80] = 1001
            self.vm_input[0x2] = action[0]
            self.vm_input[0x3] = action[1]
            self.orbitvm.step(self.vm_input, self.vm_output)
            done = False
            reward = 1.0
            return np.array(self.vm_output), reward, done, {}

    def _reset(self):
        self.orbitvm = P1()
        self.vm_input = [0.0] * 0x3E81
        self.vm_output = [0.0] * 6

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = 2e+08
        scale = screen_width / world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)

            earth = rendering.make_circle(radius=scale * 6.357e+06, filled=True)
            earth_trans = rendering.Transform(translation=(screen_width / 2, screen_height / 2))
            earth.set_color(.1,.1,.8)
            earth.add_attr(earth_trans)

            satellite = rendering.make_circle(radius=scale * 6.357e+05, filled=True)
            satellite.set_color(.8,.6,.4)
            self.satellite_trans = rendering.Transform()
            satellite.add_attr(self.satellite_trans)

            orbit = rendering.make_circle(radius=scale * self.vm_output[0x4], filled=False)
            orbit_trans = rendering.Transform(translation=(screen_width / 2, screen_height / 2))
            orbit.add_attr(orbit_trans)

            self.viewer.add_geom(orbit)
            self.viewer.add_geom(earth)
            self.viewer.add_geom(satellite)


        x, y = self.vm_output[0x2], self.vm_output[0x3]
        satx = x * scale + screen_width / 2.0
        saty = y * scale + screen_height / 2.0

        self.satellite_trans.set_translation(satx, saty)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


register(
    id='OrbitP1-v0',
    entry_point='orbitvm.p1_env:OrbitP1Env')
