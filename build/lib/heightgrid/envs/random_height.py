from heightgrid.heightgrid import *
from heightgrid.register import register

class RandomHeights(GridWorld):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        grid_height, 
        target_grid_height,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            target_grid_height=target_grid_height,
            grid_height=grid_height
        )

    def reset(self):
        obs = super().reset(agent_pose=(*self.agent_start_pos, self.agent_start_dir))
        self.agent_position = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        return obs

class RandomHeightEnv5x5(RandomHeights):
    def __init__(self, **kwargs):
        grid_height = np.random.randint(0, 2, (5, 5))
        zero_height_pos = np.where(grid_height == 0)

        super().__init__(grid_height=grid_height, 
                         target_grid_height=np.zeros((5,5)),
                         agent_start_pos=(zero_height_pos[0][0], zero_height_pos[1][0]),
                        **kwargs)
        self.place_obj_at_pos(Goal(), np.array([4, 4]))


register(
    id='HeightGrid-RandomHeight-5x5-v0',
    entry_point='heightgrid.envs:RandomHeightEnv5x5'
)
