from heightgrid.dm_heightgrid import *
# from heightgrid.register import register
from heightgrid.create_maps import create_rectangle


class Hole(GridWorld):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        grid_height, 
        target_grid_height,
        agent_start_pos=(0,0),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            target_grid_height=target_grid_height,
            grid_height=grid_height
        )

    def reset(self, agent_pose=(0, 0, 0)):
        # print("init pose ", agent_pose)
        restart = super().reset(agent_pose=agent_pose)
        self.agent_start_pos = (agent_pose[0], agent_pose[1])
        self.agent_start_dir = agent_pose[2]
        self.agent_position = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        return restart


class HoleEnv5x5(Hole):
    def __init__(self, **kwargs):
        grid_height = np.zeros((5, 5))
        zero_height_pos = np.where(grid_height == 0)
        target_map = np.zeros((5, 5))
        target_map = create_rectangle(target_map, (2, 3), (1, 1), value=-1)
        target_map = create_rectangle(target_map, (2, 3), (3, 1), value=1)

        super().__init__(grid_height=grid_height, 
                         target_grid_height=target_map,
                         agent_start_pos=(zero_height_pos[0][0], zero_height_pos[1][0]),
                        **kwargs)
    def reset(self):
        zero_height_pos = np.where(self.grid_height == 0)
        idx = np.random.randint(len(zero_height_pos[0]))
        agent_position = (zero_height_pos[0][idx], zero_height_pos[1][idx])
        agent_orientation = np.random.randint(0, 4)
        restart = super().reset((*agent_position, agent_orientation))
        return restart
        # self.place_obj_at_pos(Goal(), np.array([4, 4]))


class HoleEnv32x32(Hole):
    def __init__(self, **kwargs):
        size = 32
        grid_height = np.zeros((size, size))
        zero_height_pos = np.where(grid_height == 0)
        target_map = np.zeros((size, size))
        target_map = create_rectangle(target_map, (2, 3), (1, 1), value=-1)
        target_map = create_rectangle(target_map, (2, 3), (3, 1), value=1)

        super().__init__(grid_height=grid_height, 
                         target_grid_height=target_map,
                         agent_start_pos=(zero_height_pos[0][0], zero_height_pos[1][0]),
                        **kwargs)
    def reset(self):
        zero_height_pos = np.where(self.grid_height == 0)
        idx = np.random.randint(len(zero_height_pos[0]))
        agent_position = (zero_height_pos[0][idx], zero_height_pos[1][idx])
        agent_orientation = np.random.randint(0, 4)
        restart = super().reset((*agent_position, agent_orientation))
        return restart
        # self.place_obj_at_pos(Goal(), np.array([4, 4]))



# register(
#     id='dm_HeightGrid-Hole-5x5-v0',
#     entry_point='heightgrid.envs:HoleEnv5x5'
# )
