import unittest
import numpy.testing as npt
import numpy as np
from heightgrid.heightgrid_v2 import AgentObj, Goal, GridWorld, OBJECT_TO_IDX


class GridTester(unittest.TestCase):

    def setUp(self) -> None:
        rewards = {"collision_reward": -1,
                   "longitudinal_step_reward": -0.1,
                   "lateral_step_reward": -0.1,
                   "dig_reward": 1,
                   "move_dirt_reward": -0.1,
                   "existence_reward": -0.05,
                   "cabin_turn_reward": -0.1,
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
        npt.assert_allclose(excavation_map, obs["image"][:, :, 2])

    def test_vector_obs(self):
        obs = self.basic_grid.reset()
        x_dim, y_dim = self.basic_grid.grid_height.shape
        pos_x = np.zeros(x_dim)
        pos_y = np.zeros(y_dim)
        pos_x[0] = 1
        pos_y[0] = 1
        npt.assert_allclose(pos_x, obs["pos_x"])
        npt.assert_allclose(pos_y, obs["pos_y"])

        dir = np.zeros(8)
        dir[6] = 1
        npt.assert_allclose(dir, obs["cabin_orientation"])

        bucket = np.zeros(2)
        bucket[0] = 1
        npt.assert_allclose(bucket, obs["bucket"])

    def test_vector_obs_2(self):
        obs = self.basic_grid.reset((2, 3, 1))
        x_dim, y_dim = self.basic_grid.grid_height.shape
        pos_x = np.zeros(x_dim)
        pos_y = np.zeros(y_dim)
        pos_x[2] = 1
        pos_y[3] = 1
        npt.assert_allclose(pos_x, obs["pos_x"])
        npt.assert_allclose(pos_y, obs["pos_y"])

        dir = np.zeros(8)
        dir[1] = 1
        npt.assert_allclose(dir, obs["cabin_orientation"])

        bucket = np.zeros(2)
        bucket[0] = 1
        npt.assert_allclose(bucket, obs["bucket"])

    def test_get_obj(self):
        self.basic_grid.reset(agent_pose=(1, 1, 1))
        self.assertEqual(AgentObj, type(self.basic_grid.get(1, 1)))

    def test_excavation_map(self):
        self.basic_grid.reset(agent_pose=(1, 1, 1))
        expected_excavation_map = np.zeros((3, 3))


if __name__ == '__main__':
    rewards = {"collision_reward": -1,
               "longitudinal_step_reward": -0.1,
               "base_turn_reward": -0.1,
               "dig_reward": 1,
               "move_dirt_reward": -0.1,
               "existence_reward": -0.05,
               "cabin_turn_reward": -0.1,
               "terminal_reward": 1}

    initial_height_grid = np.zeros((5, 5))
    # target area is 3x3 centered and with value 1
    target_map = np.zeros((5, 5))
    target_map[1:4, 1:4] = -1
    # excavation area is 3x3 centered and with value 1, outside -1
    excavation_map = -1 * np.ones((5, 5))
    excavation_map[1:4, 1:4] = 1

    gridworld = GridWorld(initial_height_grid, target_map, excavation_map, rewards=rewards)
    obs = gridworld.reset()
    print(obs['image'])

