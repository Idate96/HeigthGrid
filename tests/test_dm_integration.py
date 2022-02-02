import dm_env
from numpy import testing
from heightgrid.dm_heightgrid import AgentObj, Goal, GridWorld, OBJECT_TO_IDX
import unittest
import numpy.testing as npt
import numpy as np 

class DmTest(unittest.TestCase):

    def setUp(self) -> None:
        self.height_map = np.zeros((5, 5))
        self.target_map = np.ones((5, 5))
        self.env = GridWorld(self.height_map, self.target_map)


    def test_instance(self):
        height_map = np.zeros((5, 5))
        target_map = np.ones((5, 5))
        grid = GridWorld(height_map, target_map)

    
    def test_action_space(self):
        num_actions = self.env.action_spec().num_values
        self.assertEqual(num_actions, 5)
    

    def test_reset(self):
        timestep = self.env.reset()

    
    def test_observation(self):
        timestep = self.env.reset()
        timestep = self.env.step(0)
        npt.assert_allclose(timestep.observation['grid'][:, :, 0], self.height_map)
    