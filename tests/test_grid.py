from heightgrid.envs.empty import EmptyEnv5x5
import unittest
import numpy.testing as npt
import numpy as np
from heightgrid.heightgrid import AgentObj, Goal, GridWorld, OBJECT_TO_IDX

class GridTester(unittest.TestCase):    

    def setUp(self) -> None:
        height_map_basic = np.zeros((5,5))
        self.basic_grid = GridWorld(height_map_basic, height_map_basic)


    def test_grid_init(self):
        height_map = np.zeros((5, 5))
        target_map = np.ones((5, 5))
        grid = GridWorld(height_map, target_map)


    def test_grid_dimensions(self):
        height_map = np.zeros((5, 5))
        npt.assert_allclose(height_map, self.basic_grid.grid_height)
    

    def test_pose_grid(self):
        self.basic_grid.reset()
        target_position = np.zeros((5,5,1))
        target_position[0, 0, 0] = 1

        target_orientation = np.zeros((5,5,1))
        target_pose = np.concatenate((target_position, target_orientation), axis=2)
        
        npt.assert_allclose(target_pose, self.basic_grid.grid_object_pose)


    def test_obs(self):
        obs = self.basic_grid.reset()
        target_position = np.zeros((5, 5, 1))
        target_position[0, 0, 0] = 1

        target_orientation = np.zeros((5, 5, 1))

        height_map = np.zeros((5, 5, 1))
        target_map = np.zeros((5, 5, 1))

        target_obs = np.concatenate((height_map, target_map, target_position), axis=2)
        npt.assert_allclose(target_obs, obs['image'])


    def test_grid_obj(self):
        len_grid_obj = 5 * 5 
        self.assertAlmostEqual(5, self.basic_grid.x_dim)
        self.assertAlmostEqual(5, self.basic_grid.y_dim)
        self.assertAlmostEqual(len_grid_obj, len(self.basic_grid.grid_object))


    def test_add_obj_pos(self):
        goal = Goal()
        self.basic_grid.reset(agent_pose=(0, 0, 0))
        self.basic_grid.place_obj_at_pos(goal, np.array([3, 3]))
        object_map = np.zeros((5, 5))
        object_map[0, 0] = 1
        object_map[3, 3] = OBJECT_TO_IDX['goal']
        npt.assert_allclose(object_map, self.basic_grid.grid_object_pose[:, :, 0])


    def test_intitial_reset_pos(self):
        self.basic_grid.reset(agent_pose=(1, 1, 1))
        object_pose_target = np.zeros((5, 5, 2))
        object_pose_target[1, 1, 0] = 1
        object_pose_target[1, 1, 1] = 1
        npt.assert_allclose(object_pose_target[:, :, 0], self.basic_grid.grid_object_pose[:, :, 0])


    def test_get_obj(self):
        self.basic_grid.reset(agent_pose=(1, 1, 1))
        self.assertEqual(AgentObj, type(self.basic_grid.get(1, 1)))

    
    def test_action_mask(self):
        height_map = np.zeros((5, 5))
        target_map = np.ones((5, 5))
        grid = GridWorld(height_map, target_map, mask=True)
        obs = grid.reset()
        target_action_mask = np.array([1, 1, 1, 0, 0], dtype=np.uint8)
        npt.assert_allclose(obs['mask'], target_action_mask) 


    def test_action_mask_dig(self):
        height_map = np.zeros((5, 5))
        target_map = -np.ones((5, 5))
        grid = GridWorld(height_map, target_map, mask=True)
        obs = grid.reset()
        target_action_mask = np.array([1, 1, 1, 1, 0], dtype=np.uint8)
        npt.assert_allclose(obs['mask'], target_action_mask) 


    def test_action_mask_dig_while_carrying(self):
        height_map = np.zeros((5, 5))
        target_map = -np.ones((5, 5))
        grid = GridWorld(height_map, target_map, mask=True)
        obs = grid.reset()
        grid.carrying = 1
        obs, r, d, i = grid.step(0)
        target_action_mask = np.array([1, 1, 1, 0, 0], dtype=np.uint8)
        npt.assert_allclose(obs['mask'], target_action_mask) 