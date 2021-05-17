from heightgrid.heightgrid import *
from heightgrid.register import register

class FlatEnv(GridWorld):
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
        restart = super().reset(agent_pose=(*self.agent_start_pos, self.agent_start_dir))
        self.agent_position = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        return restart

class EmptyEnv5x5(FlatEnv):
    def __init__(self, **kwargs):
        super().__init__(grid_height=np.zeros((5, 5)), 
                         target_grid_height=np.zeros((5,5)), **kwargs)
        self.place_obj_at_pos(Goal(), np.array([4, 4]))

class EmptyRandomEnv5x5(FlatEnv):
    def __init__(self):
        super().__init__(grid_height=np.zeros((5, 5)), 
                         target_grid_height=np.zeros((5,5)), agent_start_pos=None)
class EmptyEnv6x6(FlatEnv):
    def __init__(self, **kwargs):
        super().__init__(grid_height=np.zeros((6, 6)), 
                         target_grid_height=np.zeros((6, 6)), **kwargs)

class EmptyRandomEnv6x6(FlatEnv):
    def __init__(self):
        super().__init__(grid_height=np.zeros((6, 6)), 
                         target_grid_height=np.zeros((6, 6)), agent_start_pos=None)

class EmptyEnv16x16(FlatEnv):
    def __init__(self, **kwargs):
        super().__init__(grid_height=np.zeros((16, 16)), 
                         target_grid_height=np.zeros((16, 16)), **kwargs)

class EmptyEnv32x32(FlatEnv):
    def __init__(self, **kwargs):
        super().__init__(grid_height=np.zeros((32, 32)), 
                         target_grid_height=np.zeros((32, 32)), **kwargs)

# register(
#     id='HeightGrid-Empty-5x5-v0',
#     entry_point='heightgrid.envs:EmptyEnv5x5'
# )

# register(
#     id='HeightGrid-Empty-Random-5x5-v0',
#     entry_point='heightgrid.envs:EmptyRandomEnv5x5'
# )

# register(
#     id='HeightGrid-Empty-6x6-v0',
#     entry_point='heightgrid.envs:EmptyEnv6x6'
# )

# register(
#     id='HeightGrid-Empty-Random-6x6-v0',
#     entry_point='heightgrid.envs:EmptyRandomEnv6x6'
# )

# register(
#     id='HeightGrid-Empty-8x8-v0',
#     entry_point='heightgrid.envs:EmptyEnv'
# )

# register(
#     id='HeightGrid-Empty-16x16-v0',
#     entry_point='heightgrid.envs:EmptyEnv16x16'
# )

# register(
#     id='HeightGrid-Empty-32x32-v0',
#     entry_point='heightgrid.envs:EmptyEnv32x32'
# )
