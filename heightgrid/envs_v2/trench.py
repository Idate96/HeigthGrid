from heightgrid.heightgrid_v2 import *
from heightgrid.register import register
from heightgrid.create_maps import create_rectangle

class TrenchEnv7x7_3x1(GridWorld):
    def __init__(self, **kwargs):

        rewards = {"collision_reward": -0.01,  # against wall 0, ok
                   "longitudinal_step_reward": -0.01,
                   "base_turn_reward": -0.02,  # ok
                   "dig_reward": 1,  # ok
                   "dig_wrong_reward": -2,  # ok
                   "move_dirt_reward": 1,
                   "existence_reward": -0.005,  # ok
                   "cabin_turn_reward": -0.005,  # ok
                   "terminal_reward": 10}

        heightgrid = np.zeros((5, 5))
        # target area is 3x3 centered and with value 1
        target_map = np.ones((5, 5))
        target_map[1:4, 1:4] = -1
        # excavation area is 3x3 centered and with value 1, outside -1
        dirt_grid = np.zeros((5, 5))

        super().__init__(heightgrid=heightgrid,
                         target_grid_height=target_map,
                         dirt_grid=dirt_grid,
                         rewards=rewards,
                        **kwargs)

    def reset(self):
        agent_pose = (0, 0, 0, 0)
        obs = super().reset(agent_pose=agent_pose)
        return obs
        # self.place_obj_at_pos(Goal(), np.array([4, 4]))


class TrenchEnv5x5_3x3(GridWorld):
    def __init__(self, **kwargs):

        rewards = {"collision_reward": -0.01,  # against wall 0, ok
                   "longitudinal_step_reward": -0.01,
                   "base_turn_reward": -0.02,  # ok
                   "dig_reward": 1,  # ok
                   "dig_wrong_reward": -2,  # ok
                   "move_dirt_reward": 1,
                   "existence_reward": -0.005,  # ok
                   "cabin_turn_reward": -0.005,  # ok
                   "terminal_reward": 10}

        heightgrid = np.zeros((5, 5))
        # target area is 3x3 centered and with value 1
        target_map = np.ones((5, 5))
        target_map[1:4, 1:4] = -1
        # excavation area is 3x3 centered and with value 1, outside -1
        dirt_grid = np.zeros((5, 5))

        super().__init__(heightgrid=heightgrid,
                         target_grid_height=target_map,
                         dirt_grid=dirt_grid,
                         rewards=rewards,
                        **kwargs)

    def reset(self):
        agent_pose = (0, 0, 0, 0)
        obs = super().reset(agent_pose=agent_pose)
        return obs
        # self.place_obj_at_pos(Goal(), np.array([4, 4]))


class TrenchEnv5x5_1x1(GridWorld):
    def __init__(self, **kwargs):

        rewards = {"collision_reward": -0.01,  # against wall 0, ok
                   "longitudinal_step_reward": -0.01,
                   "base_turn_reward": -0.02,  # ok
                   "dig_reward": 1,  # ok
                   "dig_wrong_reward": -2,  # ok
                   "move_dirt_reward": 1,
                   "existence_reward": -0.005,  # ok
                   "cabin_turn_reward": -0.005,  # ok
                   "terminal_reward": 10}

        heightgrid = np.zeros((5, 5))
        # target area is 3x3 centered and with value 1
        target_map = np.ones((5, 5))
        target_map[2, 2] = -1
        # excavation area is 3x3 centered and with value 1, outside -1
        dirt_grid = np.zeros((5, 5))

        super().__init__(heightgrid=heightgrid,
                         target_grid_height=target_map,
                         dirt_grid=dirt_grid,
                         rewards=rewards,
                        **kwargs)

    def reset(self):
        agent_pose = (0, 0, 0, 0)
        obs = super().reset(agent_pose=agent_pose)
        return obs
        # self.place_obj_at_pos(Goal(), np.array([4, 4]))


register(
    id="HeightGrid-Hole3-v0",
    entry_point='heightgrid.envs_v2:HoleEnv5x5_3x3'
)

register(
    id='HeightGrid-Hole1-v1',
    entry_point='heightgrid.envs_v2:HoleEnv5x5_1x1'
)

