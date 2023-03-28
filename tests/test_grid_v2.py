import unittest
import numpy.testing as npt
import numpy as np
from heightgrid.heightgrid_v2 import AgentObj, Goal, GridWorld, OBJECT_TO_IDX


class GridTester(unittest.TestCase):

    def setUp(self) -> None:
        rewards = {"collision_reward": -0.001,  # against wall 0, ok
                   "longitudinal_step_reward": -0.0005,
                   "base_turn_reward": -0.002,  # ok
                   "dig_reward": .1,  # ok
                   "dig_wrong_reward": -2,  # ok
                   "move_dirt_reward": .1,
                   "existence_reward": -0.0005,  # ok
                   "cabin_turn_reward": -0.0005,  # ok
                   "reward_scaling": 1, # ok
                   "terminal_reward": 1}

        initial_height_grid = np.zeros((5, 5))
        # target area is 3x3 centered and with value 1
        target_map = np.zeros((5, 5))
        target_map[1:4, 1:4] = -1
        # excavation area is 3x3 centered and with value 1, outside -1
        excavation_map = -1 * np.ones((5, 5))
        excavation_map[1:4, 1:4] = 1

        self.basic_grid = GridWorld(initial_height_grid, target_map, excavation_map, rewards=rewards)

    def test_grid_dimensions(self):
        height_map = np.zeros((5, 5))
        npt.assert_allclose(height_map, self.basic_grid.grid_height)


    def test_obs(self):
        obs = self.basic_grid.reset()
        initial_height_grid = np.zeros((5, 5))
        # target area is 3x3 centered and with value 1
        target_map = np.zeros((5, 5))
        target_map[1:4, 1:4] = -1
        # excavation area is 3x3 centered and with value 1, outside -1
        excavation_map = -1 * np.ones((5, 5))
        excavation_map[1:4, 1:4] = 1

        npt.assert_allclose(initial_height_grid, obs["image"][:, :, 0])
        npt.assert_allclose(target_map, obs["image"][:, :, 1])

    def test_vector_obs(self):
        obs = self.basic_grid.reset()

if __name__ == '__main__':
    unittest.main()

