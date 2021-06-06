from numpy import random
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
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            target_grid_height=target_grid_height, grid_height=grid_height, **kwargs
        )

    def reset(self):
        obs = super().reset(agent_pose=(*self.agent_start_pos, self.agent_start_dir))
        self.agent_position = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        return obs


class RandomTargetHeightEnv5x5(RandomHeights):
    def __init__(self, num_digging_pts, **kwargs):
        random_idx = np.random.choice(25, num_digging_pts, replace=False)
        grid_height = np.random.randint(0, 2, (5, 5))
        zero_height_pos = np.where(grid_height == 0)

        super().__init__(
            grid_height=grid_height,
            target_grid_height=np.zeros((5, 5)),
            agent_start_pos=(zero_height_pos[0][0], zero_height_pos[1][0]),
            **kwargs
        )
        self.place_obj_at_pos(Goal(), np.array([4, 4]))


class RandomTargetHeightEnv8x8(RandomHeights):
    def __init__(self, num_digging_pts, **kwargs):
        random_idx_dig = np.random.choice(64, num_digging_pts, replace=False)
        random_idx_dump = np.random.choice(64, num_digging_pts, replace=False)

        x_idx = np.zeros((num_digging_pts,), dtype=int)
        y_idx = np.zeros((num_digging_pts,), dtype=int)

        x_idx_dump = np.zeros((num_digging_pts,), dtype=int)
        y_idx_dump = np.zeros((num_digging_pts,), dtype=int)

        for i in range(len(random_idx_dig)):
            x_idx[i] = random_idx_dig[i] // 8
            y_idx[i] = random_idx_dig[i] % 8

            x_idx_dump[i] = random_idx_dump[i] // 8
            y_idx_dump[i] = random_idx_dump[i] % 8

        target_height = np.zeros((8, 8))
        target_height[x_idx, y_idx] = -1
        target_height[x_idx_dump, y_idx_dump] = 1

        # zero_height_pos = np.where(grid_height == 0)

        super().__init__(
            grid_height=np.zeros((8, 8)),
            target_grid_height=target_height,
            agent_start_pos=(4, 4),
            **kwargs
        )
        # self.place_obj_at_pos(Goal(), np.array([4, 4]))


class RandomTargetHeightEnv(RandomHeights):
    def __init__(self, size, num_digging_pts, **kwargs):
        self.size = size
        self.num_digging_pts = num_digging_pts

        target_height = self.generate_grid()
        # zero_height_pos = np.where(grid_height == 0)

        super().__init__(
            grid_height=np.zeros((size, size)),
            target_grid_height=target_height,
            agent_start_pos=(
                np.random.randint(0, size),
                np.random.randint(0, size),
            ),
            **kwargs
        )

        # self.place_obj_at_pos(Goal(), np.array([4, 4]))

    def generate_grid(self):
        random_idx_dig = np.random.choice(
            self.size ** 2, self.num_digging_pts, replace=False
        )
        random_idx_dump = np.random.choice(
            self.size ** 2, self.num_digging_pts, replace=False
        )

        # prevent overlapping of sites
        overlapping = np.intersect1d(random_idx_dig, random_idx_dump)
        print(overlapping)
        while overlapping.size > 0:
            random_idx_dump = np.random.choice(
                self.size ** 2, self.num_digging_pts, replace=False
            )
            overlapping = np.intersect1d(random_idx_dig, random_idx_dump)

        x_idx = np.zeros((self.num_digging_pts,), dtype=int)
        y_idx = np.zeros((self.num_digging_pts,), dtype=int)

        x_idx_dump = np.zeros((self.num_digging_pts,), dtype=int)
        y_idx_dump = np.zeros((self.num_digging_pts,), dtype=int)

        for i in range(len(random_idx_dig)):
            x_idx[i] = random_idx_dig[i] // self.size
            y_idx[i] = random_idx_dig[i] % self.size

            x_idx_dump[i] = random_idx_dump[i] // self.size
            y_idx_dump[i] = random_idx_dump[i] % self.size

        target_height = np.zeros((self.size, self.size))
        target_height[x_idx, y_idx] = -1
        target_height[x_idx_dump, y_idx_dump] = 1
        return target_height

    def reset(self):
        self.grid_target = self.generate_grid()
        agent_pos_idx = np.random.choice(self.size ** 2, 1, replace=False)
        self.agent_start_pos = (
            int(agent_pos_idx // self.size),
            int(agent_pos_idx % self.size),
        )
        self.agent_start_dir = np.random.randint(0, 4)
        return super().reset()


register(
    id="HeightGrid-RandomHeight-5x5-v0",
    entry_point="heightgrid.envs:RandomHeightEnv5x5",
)


register(
    id="HeightGrid-RandomTargetHeight-8x8-v0",
    entry_point="heightgrid.envs:RandomTargetHeightEnv8x8",
)


register(
    id="HeightGrid-RandomTargetHeight-v0",
    entry_point="heightgrid.envs:RandomTargetHeightEnv",
)
