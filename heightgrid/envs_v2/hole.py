from heightgrid.heightgrid_v2 import *
from heightgrid.register import register
from heightgrid.create_maps import create_rectangle


class Hole(GridWorld):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        heightgrid: np.ndarray,
        target_grid_height: np.ndarray,
        dirt_grid: np.ndarray,
        step_reward: float = -0.05,
        max_steps: int = 256,
        seed=24,
        rewards: Dict[str, float] = None,
    ) -> None:
        self.agent_start_pos = (0, 0)
        self.agent_start_dir = 6

        super().__init__(
            heightgrid=heightgrid,
            target_grid_height=target_grid_height,
            dirt_grid=dirt_grid,
            rewards=rewards
        )

    def reset(self, agent_pose=(0, 0, 6)):
        # print("init pose ", agent_pose)
        obs = super().reset(agent_pose=agent_pose)
        self.agent_start_pos = (agent_pose[0], agent_pose[1])
        self.agent_dir = agent_pose[2]
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        return obs


class HoleEnv5x5_3x3(GridWorld):
    def __init__(self, rewards: Dict[str, float] = None, **kwargs):

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

class HoleEnv5x5_1x1(GridWorld):
    def __init__(self, rewards: Dict[str, float] = None, **kwargs):

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
    id='HeightGrid-Hole-5x5-3x3-v2',
    entry_point='heightgrid.envs:HoleEnv5x5_3x3'
)

register(
    id='HeightGrid-Hole-5x5-1x1-v2',
    entry_point='heightgrid.envs:HoleEnv5x5_1x1'
)

