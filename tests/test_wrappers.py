import unittest
import numpy.testing as npt
import numpy as np
import gym
from heightgrid.heightgrid import AgentObj, Goal, GridWorld, OBJECT_TO_IDX
from heightgrid.wrappers import FlatObsSimpleWrapper, FlatObsWrapper, RGBImgObsWrapper, ImgObsWrapper


class WrapperTester(unittest.TestCase):
    def setUp(self) -> None:
        target_height = np.zeros((3, 3))
        target_height[2, 2] = 2
        current_height = np.zeros((3, 3))
        current_height[1, 2] = -3
        self.grid = GridWorld(current_height, target_height)
        self.env = FlatObsSimpleWrapper(self.grid)
        self.empty_grid = gym.make("HeightGrid-Empty-5x5-v0")

    def test_init(self):
        pass

    def test_wrapper(self):
        env = FlatObsSimpleWrapper(self.empty_grid)
        obs = env.reset()
        obs, reward, done, info = env.step(0)
        print(obs)

    def test_rgb_wrapper(self):
        env = RGBImgObsWrapper(self.grid)
        env = ImgObsWrapper(env)
        obs = env.reset()
        print(obs)

    def test_wrapper_ordering(self):
      obs = self.env.reset()
      print(obs)

    def test_wrapper_goal_position(self):
      env = FlatObsSimpleWrapper(self.empty_grid)
      obs = env.reset()
      print(obs)