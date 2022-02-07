from heightgrid.heightgrid_v2 import *
from heightgrid.register import register
from heightgrid.create_maps import create_rectangle

class ProceduralTrenchEnv(GridWorld):
    def __init__(self, **kwargs):
        self.global_rewards = {"collision_reward": -0.01,  # against wall 0, ok
                   "longitudinal_step_reward": -0.005,
                   "base_turn_reward": -0.02,  # ok
                   "dig_reward": 1,  # ok
                   "dig_wrong_reward": -2,  # ok
                   "move_dirt_reward": 1,
                   "existence_reward": -0.005,  # ok
                   "cabin_turn_reward": -0.005,  # ok
                   "reward_scaling": 1, # ok
                   "terminal_reward": 10}

        self.name = "ProceduralTrench"
        self.map_size = 7
        grid_height = np.zeros((self.map_size, self.map_size))
        self.trench_lenght = 2
        self.trench_width = 1
        self.num_trenches = 1
        self.min_unoccupied_space = 0.5

        self.max_trench_length = self.map_size - 2
        self.min_trench_length = 2
        self.max_trench_width = 1
        self.min_trench_width = 1
        self.max_num_trenches = self.map_size - 2
        self.min_num_trenches = 1
        self.level = 0
        self.max_level = (self.max_num_trenches - self.min_num_trenches + 1) * (self.max_trench_width - self.min_trench_width + 1) * (self.max_trench_length - self.min_trench_length + 1)
        print("Num Levels: ", self.max_level)
        self.reward_scaling = self.global_rewards["reward_scaling"]
        self.terminal_reward_0 = self.global_rewards["terminal_reward"]
        # rewards:
        # level 0: 0 -> max_reward + num_blocks * 2 - num_blocks * step_cost * 2 * 5
        target_map = self.generate_level()
        self.trench_max_steps = 512
        # update dictionary with new rewards

        super().__init__(heightgrid=grid_height,
                         target_grid_height=target_map,
                         rewards=self.global_rewards,
                         max_steps=self.trench_max_steps,
                         **kwargs)

    def generate_level(self):
        # print("Generating level: {}".format(self.level))
        # progressively increase in this order the number of trenches, the trench length and the trench width
        self.trench_lenght = self.min_trench_length + self.level % (self.max_trench_length - self.min_trench_length + 1)
        self.trench_width = self.min_trench_width + int(self.level / (self.max_trench_length - self.min_trench_length + 1)) % (self.max_trench_width - self.min_trench_width + 1)
        self.num_trenches = self.min_num_trenches + int(self.level / ((self.max_trench_width - self.min_trench_width + 1) *
                                                        (self.max_trench_length - self.min_trench_length + 1))) % (self.max_num_trenches - self.min_num_trenches + 1)
        # print("trench_lenght: {}, trench_width: {}, num_trenches: {}".format(self.trench_lenght, self.trench_width, self.num_trenches))

        occupied_space = 1
        num_blocks = 0
        num_trences = np.random.randint(self.min_num_trenches, self.num_trenches + 1)
        while occupied_space > (1 - self.min_unoccupied_space):
            target_map_0 = np.ones((self.map_size, self.map_size))
            for i in range(num_trences):
                target_map_0 = self.create_trench(target_map_0)
            # compute occupied space
            num_blocks = np.sum(target_map_0 == -1)
            occupied_space = num_blocks / (self.map_size * self.map_size)
            # print("occupied space: ", occupied_space)
        target_map_0 = np.rot90(target_map_0, np.random.randint(0, 4))

        self.reward_scaling = self.min_trench_length / (self.level + 1) + 0.1
        return target_map_0

    def level_up(self):
        self.level = min(self.level + 1, self.max_level - 1)
        print("New level: {}".format(self.level))

    def create_trench(self, map):
        # place randomly the trench but make sure it fits in the grid
        trench_length = self.min_trench_length + np.random.randint(0, self.trench_lenght - self.min_trench_length + 1)
        # pick random trench width between the min and self.trench_width
        trench_width = self.min_trench_width + np.random.randint(0, self.trench_width - self.min_trench_width + 1)

        # place randomly the trench but make sure it fits in the grid
        trench_start_x = np.random.randint(0, self.map_size - trench_length)
        trench_start_y = np.random.randint(0, self.map_size - trench_width)
        trench_end_x = trench_start_x + trench_length
        trench_end_y = trench_start_y + trench_width

        map[trench_start_x:trench_end_x, trench_start_y:trench_end_y] = -1
        # randomly rotate the map 90 degrees
        map = np.rot90(map, np.random.randint(0, 4))
        return map

    def reset(self):
        target_map = self.generate_level()
        # udpate rewards
        rewards = {}
        for (k, v) in self.global_rewards.items():
            # except for the reward scaling and terminal reward
            if k == "dig_reward" and k != "move_dirt_reward":
                rewards[k] = v * self.reward_scaling
            else:
                rewards[k] = v

        # pick a random position inside the grid
        agent_pos = np.array([np.random.randint(0, self.grid_height.shape[0]),
                              np.random.randint(0, self.grid_height.shape[1])])
        # pick a random orientation
        agent_pose = (agent_pos[0], agent_pos[1], np.random.randint(0, 4), np.random.randint(0, 8))
        obs = super().reset(agent_pose=agent_pose, target_map=target_map, rewards=rewards)
        return obs



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

        heightgrid = np.zeros((7, 7))
        # target area is 3x3 centered and with value 1
        target_map = np.ones((7, 7))
        target_map[1:6, 3:4] = -1
        # excavation area is 3x3 centered and with value 1, outside -1

        super().__init__(heightgrid=heightgrid,
                         target_grid_height=target_map,
                         rewards=rewards,
                        **kwargs)

    def reset(self):
        # pick a random position inside the grid
        agent_pos = np.array([np.random.randint(0, self.grid_height.shape[0]),
                              np.random.randint(0, self.grid_height.shape[1])])
        # pick a random orientation
        agent_pose = (agent_pos[0], agent_pos[1], np.random.randint(0, 4), np.random.randint(0, 8))
        obs = super().reset(agent_pose=agent_pose)
        return obs


class TrenchEnv5x5_3x1(GridWorld):
    def __init__(self, **kwargs):

        rewards = {"collision_reward": -0.01,  # against wall 0, ok
                   "longitudinal_step_reward": -0.01,
                   "base_turn_reward": -0.02,  # ok
                   "dig_reward": 1,  # ok
                   "dig_wrong_reward": -2,  # ok
                   "move_dirt_reward": 1,
                   "existence_reward": -0.05,  # ok
                   "cabin_turn_reward": -0.005,  # ok
                   "terminal_reward": 10}

        heightgrid = np.zeros((5, 5))
        # target area is 3x3 centered and with value 1
        target_map = np.ones((5, 5))
        target_map[1:4, 2:3] = -1
        # excavation area is 3x3 centered and with value 1, outside -1

        super().__init__(heightgrid=heightgrid,
                         target_grid_height=target_map,
                         rewards=rewards,
                        **kwargs)

    def reset(self):
        # pick a random position inside the grid
        agent_pos = np.array([np.random.randint(0, self.grid_height.shape[0]),
                              np.random.randint(0, self.grid_height.shape[1])])
        # pick a random orientation
        agent_pose = (agent_pos[0], agent_pos[1], np.random.randint(0, 4), np.random.randint(0, 8))
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
                   "existence_reward": -0.05,  # ok
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
                         rewards=rewards,
                        **kwargs)

    def reset(self):
        agent_pose = (0, 0, 0, 0)
        obs = super().reset(agent_pose=agent_pose)
        return obs
        # self.place_obj_at_pos(Goal(), np.array([4, 4]))


register(
    id="HeightGrid-ProceduralTrench-v0",
    entry_point='heightgrid.envs_v2:ProceduralTrenchEnv'
)

register(
    id="HeightGrid-Trench-7-3-v0",
    entry_point='heightgrid.envs_v2:TrenchEnv7x7_3x3'
)

register(
    id="HeightGrid-Trench-5-3-v0",
    entry_point='heightgrid.envs_v2:TrenchEnv5x5_3x3'
)

register(
    id='HeightGrid-Trench-5-1-v1',
    entry_point='heightgrid.envs_v2:TrenchEnv5x5_1x1'
)

