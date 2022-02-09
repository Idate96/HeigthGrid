from typing import Tuple, Dict

from matplotlib.pyplot import grid
import numpy as np
import gym
from gym import spaces
from abc import abstractmethod, ABCMeta

from numpy.core.numeric import False_
import heightgrid.rendering
from enum import IntEnum
import warnings
import heightgrid.window
import networkx as nx


class Actions(IntEnum):
    # Turn left, turn right, move forward
    rotate_cabin_counter = 0
    rotate_cabin_clock = 1
    forward = 2
    backward = 3
    rotate_base_counter = 4
    rotate_base_clock = 5
    do = 6


eps = 10 ** -7

SIZE_TILE_PIXELS = 8

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

OBJECT_TO_IDX = {
    "empty": 0,
    "goal": -1,
    "wall": 2,
    "ramp": 3,
    "agent": 1,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

DIR_TO_INT = [0, 1, 2, 3]

DIR_TO_VEC_BASE = [
    np.array((1, 0)),
    # right down
    np.array((0, 1)),
    # left down
    # Pointing left (negative X)
    np.array((-1, 0)),
    # left Up (negative Y)
    # up
    np.array((0, -1)),
    # right up
]


DIR_TO_VEC_CABIN = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # right down
    np.array((1, 1)),
    # Down (positive Y)
    np.array((0, 1)),
    # left down
    np.array((-1, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # left Up (negative Y)
    np.array((-1, -1)),
    # up
    np.array((0, -1)),
    # right up
    np.array((1, -1)),
]


class GridObject(metaclass=ABCMeta):
    """Base class for objects are present in the environment"""

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color

        self.type = type
        self.color = color
        self.orientable = False
        self.current_pos = None

    @abstractmethod
    def can_overlap(self):
        raise NotImplementedError

    def encode(self):
        """Encode the object type into and integer"""
        return (
            OBJECT_TO_IDX[self.type],
            IDX_TO_COLOR[OBJECT_TO_IDX[self.type] % 10],
            32,
        )


class Goal(GridObject):
    def __init__(self):
        super().__init__("goal", "green")

    def can_overlap(self):
        return True

    def render(self, img):
        heightgrid.rendering.fill_coords(
            img, heightgrid.rendering.point_in_rect(0, 1, 0, 1), COLORS[self.color]
        )


class Ramp(GridObject):
    def __init__(self):
        super().__init__("ramp", "red")
        self.orientable = True
        self.orientation = None

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        heightgrid.rendering.fill_coords(
            img, heightgrid.rendering.point_in_rect(0, 1, 0, 1), c
        )

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            heightgrid.rendering.fill_coords(
                img,
                heightgrid.rendering.point_in_line(0.1, ylo, 0.3, yhi, r=0.03),
                (0, 0, 0),
            )
            heightgrid.rendering.fill_coords(
                img,
                heightgrid.rendering.point_in_line(0.3, yhi, 0.5, ylo, r=0.03),
                (0, 0, 0),
            )
            heightgrid.rendering.fill_coords(
                img,
                heightgrid.rendering.point_in_line(0.5, ylo, 0.7, yhi, r=0.03),
                (0, 0, 0),
            )
            heightgrid.rendering.fill_coords(
                img,
                heightgrid.rendering.point_in_line(0.7, yhi, 0.9, ylo, r=0.03),
                (0, 0, 0),
            )


class Wall(GridObject):
    def __init__(self, color="grey"):
        super().__init__("wall", color)

    def can_overlap(self):
        return False

    def render(self, img):
        heightgrid.rendering.fill_coords(
            img, heightgrid.rendering.point_in_rect(0, 1, 0, 1), COLORS[self.color]
        )


class AgentObj(GridObject):
    def __init__(self, type="agent", color="blue"):
        super().__init__(type, color)
        self.orientable = True
        self.orientation = None

    def can_overlap(self):
        return True

    def render(self, img):
        # heightgrid.rendering.fill_coords(img, heightgrid.rendering.point_in_rect(0, 1, 0, 1), np.array([255, 255, 255])*(height + 3)/7)
        heightgrid.rendering.fill_coords(
            img, heightgrid.rendering.point_in_rect(0, 1, 0, 1), COLORS[self.color]
        )


class GridWorld(gym.Env):
    tile_cache = {}
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            heightgrid: np.ndarray,
            target_grid_height: np.ndarray,
            step_reward: float = -0.05,
            max_steps: int = 256,
            seed=24,
            rewards: Dict[str, float] = None,
            render=True
    ) -> None:
        super().__init__()
        assert np.shape(heightgrid) == np.shape(target_grid_height)
        # actions are discrete
        self.action_space = spaces.Discrete(len(Actions))
        self.actions = Actions
        self.num_actions = len(Actions)

        self.x_dim, self.y_dim = np.shape(heightgrid)

        # observations
        self.image_obs = np.zeros((*np.shape(heightgrid), 2))
        self.pos_x = np.zeros(self.x_dim)
        self.pos_y = np.zeros(self.y_dim)
        self.cabin_dir_ = np.zeros(8)
        self.base_dir_ = np.zeros(4)
        self.bucket = np.zeros(1)

        # these values are not modified during run time
        self.heigthgrid_0 = heightgrid
        self.grid_height = heightgrid
        self.grid_target = target_grid_height
        self.x_dim, self.y_dim = np.shape(heightgrid)
        # rendering
        self.grid_object = [None] * (self.x_dim * self.y_dim)

        # current grid height, relative target height, current obj pos, current obj orientation
        self.observation_space = spaces.Dict(
            {
                # 3 channels: current height, target height, dirt sinks and sources (-1 sink, 0 no dirt, 1 dirt there)
                "image": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(self.x_dim, self.y_dim, 2),
                    dtype=np.int8,
                ),
                # normalized x, y position of agent
                # orientation of the cabin
                # bucket state
                "vector": spaces.MultiBinary(self.x_dim +
                                             self.y_dim +
                                             4 +
                                             8 +
                                             1),
            }
        )

        # handy
        self.agent_pos = (0, 0)
        self.base_dir = 0
        self.cabin_dir = 0
        self.bucket_full = 0
        self.last_dig_pos = (0, 0)

        self.number_dig_sites = np.sum(np.abs(target_grid_height))
        self.window = None
        self.window_target = None
        self._goal = False
        self.step_count = 0
        self.max_steps = max_steps
        self.seed(seed=seed)

        # rewards
        self.collision_reward = rewards["collision_reward"]
        self.longitudinal_step_reward = rewards["longitudinal_step_reward"]
        self.base_turn_reward = rewards["base_turn_reward"]
        self.dig_reward = rewards["dig_reward"]
        self.move_dirt_reward = rewards["move_dirt_reward"]
        self.existence_reward = rewards["existence_reward"]
        self.cabin_turn_reward = rewards["cabin_turn_reward"]
        self.terminal_reward = rewards["terminal_reward"]
        # self.dig_wrong_reward = rewards["dig_wrong_reward"]

        # rendering flat
        self.render_env = render
        self.action_mask = np.ones(self.num_actions)

        self.excavation_mask = self.image_obs[:, :, 1] == -1
        self.neutral_area = self.image_obs[:, :, 1] == 0

    @property
    def agent_pos(self):
        """
        Transforms 1-hot encoding of pos_x and pos_y into a tuple (idx_x, idx_y)
        """
        return (
            np.argmax(self.pos_x),
            np.argmax(self.pos_y),
        )

    @agent_pos.setter
    def agent_pos(self, pos):
        """
        Transforms a tuple (idx_x, idx_y) into 1-hot encoding of pos_x and pos_y
        """
        # reset pos_x and pos_y
        self.pos_x = np.zeros(self.x_dim)
        self.pos_y = np.zeros(self.y_dim)
        self.pos_x[pos[0]] = 1
        self.pos_y[pos[1]] = 1

    @property
    def base_dir(self):
        """
        Transforms 1-hot encoding of cabin_dir into an int
        """
        return np.argmax(self.base_dir_)

    @base_dir.setter
    def base_dir(self, dir):
        """
        Transforms an int into 1-hot encoding of cabin_dir
        """
        self.base_dir_ = np.zeros(4)
        self.base_dir_[dir] = 1

    def set_base_dir(self, vector_dir: np.array):
        """
        Transforms an int into 1-hot encoding of cabin_dir
        """
        # check if vector_dir is contained in DIR_TO_VEC_BASE list
        if vector_dir not in DIR_TO_VEC_BASE:
            raise ValueError("vector_dir not in DIR_TO_VEC_BASE")

        self.base_dir_ = vector_dir

    @property
    def cabin_dir(self):
        """
        Transforms 1-hot encoding of cabin_dir into an int
        """
        return np.argmax(self.cabin_dir_)

    @cabin_dir.setter
    def cabin_dir(self, dir):
        """
        Transforms an int into 1-hot encoding of cabin_dir
        """
        self.cabin_dir_ = np.zeros(8)
        self.cabin_dir_[dir] = 1

    def set_cabin_dir(self, vector_dir: np.array):
        """
        Transforms an int into 1-hot encoding of cabin_dir
        """
        # check if vector_dir is contained in DIR_TO_VEC_CABIN list
        if vector_dir not in DIR_TO_VEC_CABIN:
            raise ValueError("vector_dir not in DIR_TO_VEC_CABIN")

        self.cabin_dir_ = vector_dir

    @property
    def grid_height(self):
        return self.image_obs[:, :, 0]

    @grid_height.setter
    def grid_height(self, value):
        self.image_obs[:, :, 0] = value

    @property
    def grid_target(self):
        return self.image_obs[:, :, 1]

    @grid_target.setter
    def grid_target(self, value):
        self.image_obs[:, :, 1] = value

    @property
    def front_pos(self):
        front_pos = self.agent_pos + DIR_TO_VEC_BASE[self.base_dir]
        if front_pos[0] > self.x_dim or front_pos[1] > self.y_dim:
            front_pos = None
        return front_pos

    @property
    def cabin_front_pos(self):
        front_pos = self.agent_pos + DIR_TO_VEC_CABIN[self.cabin_dir]
        if front_pos[0] > self.x_dim or front_pos[1] > self.y_dim:
            front_pos = None
        return front_pos

    # @property
    # def dirt_grid(self):
    #     return self.image_obs[:, :, 2:]
    #
    # @dirt_grid.setter
    # def dirt_grid(self, value):
    #     self.image_obs[:, :, 2] = value

    def get_height(self, i, j):
        return self.grid_height[i, j]

    # @property
    # def action_mask(self):
    #     available_actions = np.ones((self.num_actions,), dtype=np.uint8)
    #
    #     fwd_pos = self.front_pos
    #
    #     if self.in_bounds(fwd_pos):
    #         if not self.is_traversable(self.agent_pos, fwd_pos):
    #             available_actions[self.actions.forward] = 0
    #         # Get the contents of the cell in front of the agent
    #         if self.carrying:
    #             available_actions[self.actions.dig] = 0
    #         else:
    #             available_actions[self.actions.drop] = 0
    #
    #         if self.obs[fwd_pos[0], fwd_pos[1], 1] > -0.5:
    #             available_actions[self.actions.dig] = 0
    #
    #         if self.obs[fwd_pos[0], fwd_pos[1], 1] < 0.5:
    #             available_actions[self.actions.drop] = 0
    #     else:
    #         available_actions[self.actions.dig] = 0
    #         available_actions[self.actions.drop] = 0
    #         available_actions[self.actions.forward] = 0
    #
    #     return available_actions

    def fwd_direction_allowed(self, fwd_pos):
        return self.in_bounds(fwd_pos) and self.is_traversable(self.agent_pos, fwd_pos)

    def action_masks(self):
        action_mask = np.ones((self.num_actions,), dtype=np.uint8)

        if not self.bucket_full:
            # mask forward action?
            fwd_pos = self.agent_pos + DIR_TO_VEC_BASE[self.base_dir]
            if not self.fwd_direction_allowed(fwd_pos):
                action_mask[self.actions.forward] = 0

            # mask backward action?
            fwd_pos = self.agent_pos + DIR_TO_VEC_BASE[(self.base_dir + 2) % 4]
            if not self.fwd_direction_allowed(fwd_pos):
                action_mask[self.actions.backward] = 0

            # mask rotating the base action?
            # if after rotating the machine cannot move forward or backward mask the action
            # counter clockwise
            base_dir_counter = (self.base_dir + 1) % 4
            # check if at least one of the two directions is traversable
            fwd_dir_counter = self.agent_pos + DIR_TO_VEC_BASE[base_dir_counter]
            fwd_allowed_counter = self.fwd_direction_allowed(fwd_dir_counter)
            back_dir_counter = self.agent_pos + DIR_TO_VEC_BASE[(base_dir_counter + 2) % 4]
            back_allowed_counter = self.fwd_direction_allowed(back_dir_counter)

            if not fwd_allowed_counter and not back_allowed_counter:
                action_mask[self.actions.rotate_base_clock] = 0
                action_mask[self.actions.rotate_base_counter] = 0

            # mask cabin rotation if there is nothing to dig around the excavator
            action_mask[self.actions.rotate_cabin_counter] = 0
            action_mask[self.actions.rotate_cabin_clock] = 0
            num_dig_spots = 0

            for i in range(8):
                dir = DIR_TO_VEC_CABIN[i]
                pos = self.agent_pos + dir
                # if can't dig in any spot mask the action
                if self.can_dig(pos):
                    num_dig_spots += 1

            # dumping spots are diggable so we need at least two diggable spots
            if num_dig_spots > 0:
                action_mask[self.actions.rotate_cabin_clock] = 1
                action_mask[self.actions.rotate_cabin_counter] = 1

            # mask do digging if nothing is diggable in the cabin direction
            if not self.can_dig(self.agent_pos + DIR_TO_VEC_CABIN[self.cabin_dir]):
                action_mask[self.actions.do] = 0

            # prevents from blocking itself! (this is with bucket empty)
            action_mask[self.actions.do] = 0
            num_excavated_adj_areas = 0
            adj_cell_height = self.adjacent_cells_height(self.agent_pos)
            # if it contains two -1 block the action
            for (i, h) in enumerate(adj_cell_height):
                if h == -1:
                    num_excavated_adj_areas += 1
                if num_excavated_adj_areas > 1:
                    action_mask[self.actions.do] = 0
                    break

            # mask digging if cabin direction is aligned with the base direction
            if np.equal(np.abs(DIR_TO_VEC_CABIN[self.cabin_dir]), np.abs(DIR_TO_VEC_BASE[self.base_dir])).all():
                # check if blocks have been dug in the base direction
                # if one block has already been dug block the digging action
                pos_fwd = self.agent_pos + DIR_TO_VEC_BASE[self.base_dir]
                pos_back = self.agent_pos - DIR_TO_VEC_BASE[(self.base_dir + 2) % 4]
                # if any of the two is not has been dug block the action
                if self.is_dug(pos_fwd) or self.is_dug(pos_back):
                    action_mask[self.actions.do] = 0

        if self.bucket_full:
            # if cabin points to dug area mask the action
            if not self.can_drop(self.agent_pos + DIR_TO_VEC_CABIN[self.cabin_dir]):
                action_mask[self.actions.do] = 0

        return action_mask

    def adjancent_cells(self, pos):
        return [pos + DIR_TO_VEC_BASE[i] for i in range(4)]

    def adjacent_cells_height(self, pos):
        return np.array([self.get_height(pos + DIR_TO_VEC_BASE[i]) for i in range(4)])

    def is_dug(self, pos):
        if not self.in_bounds(pos):
            return False
        return self.image_obs[pos[0], pos[1], 0] < -0.5

    def __str__(self):
        repre = "Height " + 10 * "-" + "\n"
        repre += np.array2string(self.image_obs[:, :, 0])
        repre += "\nRelative Height " + 10 * "-" + "\n"
        repre += np.array2string(self.image_obs[:, :, 1])
        repre += (
                "Base orientation "
                + np.array2string(DIR_TO_VEC_BASE[int(self.base_dir)])
                + "\n" +
                "Cabin orientation "
                + np.array2string(DIR_TO_VEC_CABIN[int(self.cabin_dir)])
                + "\n" +
                "Agent position " +
                str(self.agent_pos)
                + "\n"
        )
        repre += "Full bucket " + str(self.bucket_full)
        return repre

    @property
    def _observation(self):
        vector_obs = np.concatenate([
            self.pos_x,
            self.pos_y,
            self.base_dir_,
            self.cabin_dir_,
            np.array([self.bucket_full])
        ])

        obs = {
            "image": self.image_obs,
            "vector": vector_obs
        }

        return obs

    def reset(self, agent_pose: tuple = (0, 0, 0, 0), target_map: np.array = None, rewards: dict = None):
        # this is handy when dealing with environments with different number of excavation areas
        # rescaling the rewards we can get approximately the same end reward.
        if rewards:
            # rewards
            self.collision_reward = rewards["collision_reward"]
            self.longitudinal_step_reward = rewards["longitudinal_step_reward"]
            self.base_turn_reward = rewards["base_turn_reward"]
            self.dig_reward = rewards["dig_reward"]
            self.move_dirt_reward = rewards["move_dirt_reward"]
            self.existence_reward = rewards["existence_reward"]
            self.cabin_turn_reward = rewards["cabin_turn_reward"]
            self.terminal_reward = rewards["terminal_reward"]
            # self.dig_wrong_reward = rewards["dig_wrong_reward"]

        # reset current height
        self.grid_height = self.heigthgrid_0
        self.agent_pos = agent_pose[:2]
        self.base_dir = agent_pose[2]
        self.cabin_dir = agent_pose[3]

        if target_map is not None:
            self.grid_target = target_map

        if self.render_env:
            self.place_obj_at_pos(AgentObj(), agent_pose[:2])
        # self.grid_object_pose[self.agent_pos[0], self.agent_pos[1], 0] = OBJECT_TO_IDX[
        #     "agent"
        # ]
        # self.agent_dir = agent_pose[2]
        # self.grid_object_pose[self.agent_pos[0], self.agent_pos[1], 1] = self.agent_dir

        # add agent to the obj list for heightgrid.rendering
        # self.grid_object = self.x_dim * self.y_dim * [None]
        # self.place_obj_at_pos(AgentObj(), self.agent_pos)

        self.excavation_mask = self.image_obs[:, :, 1] == -1
        self.neutral_area = self.image_obs[:, :, 1] == 0

        self.step_count = 0
        self.number_dig_sites = np.sum(np.abs(self.grid_target))
        return self._observation

    def place_obj(
            self, obj: GridObject, pos: np.array = np.array([0, 0]), random=False
    ):
        """Place an object in the grid

        Args:
            obj (GridObject):
            position (np.array): array specifing the two indeces of grid
            random (bool): specifies if the location is random or not
        """

        if random:
            pass
        else:
            self.place_obj_at_pos(obj, pos)

    def place_obj_at_pos(self, obj: GridObject, pos: np.array):
        """Place an object at a specific point of the grid

        Args:
            obj (GridObject)
            pos (np.array)
        """
        if self.grid_object[pos[0] + self.x_dim * pos[1]] is None:
            self.grid_object[pos[0] + self.x_dim * pos[1]] = obj
            # self.grid_object_pose[pos[0], pos[1], 0] = OBJECT_TO_IDX[obj.type]

        else:
            warnings.warn(
                "The provided location (%d, %d) is already occupied by another object".format(
                    pos[0], pos[1]
                )
            )

    def remove_obj_at_pos(self, pos: np.array):
        """Remove an object at a specific point of the grid

        Args:
            pos (np.array)
        """
        self.grid_object[pos[0] + self.x_dim * pos[1]] = None
        # self.grid_object_pose[pos[0], pos[1], 0] = 0

    def get(self, i: int, j: int) -> GridObject:
        """Retrieve object at location (i, j)

        Args:
            i (int): index of the x location
            j (int): index of the y location
        """
        assert i <= self.x_dim, "Grid index i out of bound"
        assert j <= self.y_dim, "Grid index j out of boudns"
        return self.grid_object[i + self.x_dim * j]

    def remove_obj_at_pos(self, pos: np.array):
        if self.grid_object[pos[0] + self.x_dim * pos[1]] is None:
            warnings.warn(
                "Cannot remove object at (%d, %d), no object present".format(
                    pos[0], pos[1]
                )
            )
        else:
            self.grid_object[pos[0] + self.x_dim * pos[1]] = None
            # self.grid_object_pose[pos[0], pos[1], 0] = 0

    def is_traversable(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> bool:
        # if the height is not the same or there is dirt in the way then it is not traversable
        heigth_diff = self.image_obs[current_pos[0], current_pos[1], 0] - self.image_obs[
            target_pos[0], target_pos[1], 0]
        # if height diff is not approximately zero then it is not traversable
        if not np.isclose(heigth_diff, 0):
            return False

        return True

    def can_dig(self, target_pos):
        if target_pos[0] < 0 or target_pos[0] >= self.x_dim:
            return False
        if target_pos[1] < 0 or target_pos[1] >= self.y_dim:
            return False
        if target_pos is None:
            return False
        # not allowed if it is already dug
        if self.image_obs[target_pos[0], target_pos[1], 0] == 1:
            return True

        if self.image_obs[target_pos[0], target_pos[1], 0] == -1. or self.image_obs[target_pos[0], target_pos[1], 1] != -1.:
            return False

        return True

        # no_objects = self.get(*target_pos) is None
        # return diggable and no_objects

    def can_drop(self, target_pos):
        # if target pos is inside the bounds it's allowed
        if target_pos[0] < 0 or target_pos[0] >= self.x_dim:
            return False
        if target_pos[1] < 0 or target_pos[1] >= self.y_dim:
            return False
        # if the target is already dug is not allowed
        if self.image_obs[target_pos[0], target_pos[1], 0] == -1:
            return False

        return True

    def in_bounds(self, pos):
        x_bounded = 0 <= pos[0] < self.x_dim
        y_bounded = 0 <= pos[1] < self.y_dim
        return x_bounded and y_bounded

    def get_height(self, pos):
        if self.in_bounds(pos):
            return self.image_obs[pos[0], pos[1], 0]
        else:
            return -1

    def move_agent_pos(self, fwd_pos):
        self.move_obj_pos(self.agent_pos, fwd_pos, self.get(*self.agent_pos))
        self.agent_pos = fwd_pos

    def step(self, action):
        self.step_count += 1
        # living negative reward to encourage shortest trajectory
        reward = self.existence_reward
        done = False

        # if the bucket is full you can't move the agent just rotate it
        if not self.bucket_full:
            # move
            if action == self.actions.forward:
                fwd_pos = self.agent_pos + DIR_TO_VEC_BASE[self.base_dir]
                self.agent_pos = fwd_pos
                reward += self.longitudinal_step_reward

            elif action == self.actions.backward:
                fwd_pos = self.agent_pos + DIR_TO_VEC_BASE[(self.base_dir + 2) % 4]
                self.agent_pos = fwd_pos
                reward += self.longitudinal_step_reward

            elif action == self.actions.rotate_base_clock:
                self.base_dir = (self.base_dir + 1) % 4
                self.cabin_dir = (self.cabin_dir + 2) % 8
                reward += self.base_turn_reward

            elif action == self.actions.rotate_base_counter:
                self.base_dir = (self.base_dir - 1) % 4
                self.cabin_dir = (self.cabin_dir - 2) % 8
                reward += self.base_turn_reward

            elif action == self.actions.do:
                    # check if there is dirt first
                    if self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 0] == 1:
                        # # remove elevation
                        # self.image_obs[self.front_pos[0], self.front_pos[1], 0] = 0
                        # remove from dirt map
                        self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 0] = 0
                        # if it was in a +1 target location penalize else the agent will get stuck in a loop
                        if self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 1] == 1:
                            reward -= self.move_dirt_reward
                        # check if you blocked the access to a cell

                    else:
                        if self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 1] == -1:
                            self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 0] = -1
                            reward += self.dig_reward


                        # check if any cells got blocked by digging a cell
                        for adj_cell in self.adjancent_cells(self.cabin_front_pos):
                            if self.get_height(adj_cell) != -1:
                                adj_cell_height = self.adjacent_cells_height(adj_cell)
                                if np.sum(adj_cell_height) == -4:
                                    reward -= self.terminal_reward
                                    done = True


                        #     done = True
                        # # else:
                        # delta_h_t = self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 1] - self.image_obs[
                        #     self.cabin_front_pos[0], self.cabin_front_pos[1], 0]
                        # self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 0] = -1
                        # delta_h_tp1 = self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 1] - self.image_obs[
                        #     self.cabin_front_pos[0], self.cabin_front_pos[1], 0]
                        # reward_sign = - (np.abs(delta_h_tp1) - np.abs(delta_h_t))
                        # # penalize for digging in the wrong location
                        # reward_value = reward_sign * self.dig_reward
                        # if reward_value < 0:
                        #     # to discourage of pick and drop of the dirt
                        #     reward += reward_value * 1.1
                        # else:
                        #     reward += reward_value
                    self.bucket_full = 1
        else:
            if action == self.actions.do:
                # if drops dirt on target elevation +1 (no problem with dirt)
                if self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 1] == 1:
                    reward += self.move_dirt_reward

                self.image_obs[self.cabin_front_pos[0], self.cabin_front_pos[1], 0] = 1
                self.bucket_full = 0

        # rotate
        if action == self.actions.rotate_cabin_clock:
            self.cabin_dir = (self.cabin_dir + 1) % 8
            reward += self.cabin_turn_reward
        elif action == self.actions.rotate_cabin_counter:
            self.cabin_dir = (self.cabin_dir - 1) % 8
            reward += self.cabin_turn_reward



        height = self.image_obs[:, :, 0]
        target_height = self.image_obs[:, :, 1]

        if np.sum(target_height[self.excavation_mask] - height[self.excavation_mask]) == 0 and not self.bucket_full:
            if np.sum(height[self.neutral_area]) == 0:
                reward += self.terminal_reward
                done = True

        if self.step_count >= self.max_steps:
            done = True

        # print action masks
        action_masks = self.action_masks()
        return self._observation, reward, done, {}

    @classmethod
    def render_tile(
            cls, obj, height, base_dir=None, cabin_dir=None, tile_size=SIZE_TILE_PIXELS, subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache

        key = str(tile_size) + "h" + str(height)
        key = obj.type + key if obj else key
        # key = obj.encode() if obj else key

        if key in cls.tile_cache and obj is None:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)

        # if obj != None:
        #     obj.render(img)
        # else:
        heightgrid.rendering.fill_coords(
            img,
            heightgrid.rendering.point_in_rect(0, 1, 0, 1),
            np.array([255, 255, 255]) * (height + 3) / 7,
        )
        heightgrid.rendering.fill_coords(
            img, heightgrid.rendering.point_in_rect(0, 0.031, 0, 1), (100, 100, 100)
        )
        heightgrid.rendering.fill_coords(
            img, heightgrid.rendering.point_in_rect(0, 1, 0, 0.031), (100, 100, 100)
        )

        # Overlay the agent on top
        if base_dir is not None and cabin_dir is not None:
            # draw the base a yellow rectangle with one side longer than the other
            # to make it easier to see the direction
            back_base_fn = heightgrid.rendering.point_in_rect(
                0.25, 0.75, 0.0, 0.25
            )

            back_base_fn = heightgrid.rendering.rotate_fn(
                back_base_fn, cx=0.5, cy=0.5, theta=-np.pi/2 + np.pi / 2 * base_dir
            )
            # render in black
            heightgrid.rendering.fill_coords(
                img, back_base_fn, (0, 0, 0))

            base_fn = heightgrid.rendering.point_in_rect(
                0.25, 0.75, 0.25, 1
            )

            base_fn = heightgrid.rendering.rotate_fn(
                base_fn, cx=0.5, cy=0.5, theta=-np.pi/2 + np.pi / 2 * base_dir
            )

            heightgrid.rendering.fill_coords(img, base_fn, (255, 255, 0))

            tri_fn = heightgrid.rendering.point_in_triangle(
                (0.12, 0.81),
                (0.12, 0.19),
                (0.87, 0.50),
            )

            # Rotate the agent based on its direction
            tri_fn = heightgrid.rendering.rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta= np.pi / 4 * cabin_dir
            )
            heightgrid.rendering.fill_coords(img, tri_fn, (255, 0, 0))

        # Downsample the image to perform supersampling/anti-aliasing
        img = heightgrid.rendering.downsample(img, subdivs)

        # Cache the rendered tile

        cls.tile_cache[key] = img

        return img

    def render_grid(
            self,
            tile_size,
            height_grid,
            agent_pos=None,
            base_dir=None,
            cabin_dir=None,
            render_objects=True,
            target_height=False
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        # Compute the total grid size
        width_px = self.x_dim * tile_size
        height_px = self.y_dim * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.x_dim):
            for i in range(0, self.y_dim):

                if render_objects:
                    cell = self.get(i, j)
                else:
                    cell = None

                if target_height:
                    agent_here = False
                else:
                    agent_here = np.array_equal(agent_pos, (i, j))

                tile_img = self.render_tile(
                    cell,
                    height_grid[i, j],
                    base_dir=base_dir if agent_here else None,
                    cabin_dir=cabin_dir if agent_here else None,
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img.transpose(0, 1, 2)

    def render(
            self,
            mode="rgb_mode",
            close=False,
            block=False,
            key_handler=None,
            highlight=False,
            tile_size=SIZE_TILE_PIXELS,
    ):
        """
        Render the whole-grid human view
        """
        self.place_obj_at_pos(AgentObj(), self.agent_pos)

        if close:
            if self.window:
                self.window.close()
            if self.window_target:
                self.window_target.close()
            return

        if mode == "rgb_mode" and not self.window:
            self.window = heightgrid.window.Window("heightgrid")

        # Render the whole grid
        img = self.render_grid(
            tile_size, self.image_obs[:, :, 0], self.agent_pos, self.base_dir, self.cabin_dir
        )

        img_target = self.render_grid(
            tile_size,
            self.image_obs[:, :, 1],
            self.agent_pos,
            self.base_dir,
            self.cabin_dir,
            render_objects=True,
            target_height=True
        )

        # white row of pixels
        img_white = np.ones(shape=(tile_size * self.x_dim, tile_size, 3), dtype=np.uint8) * 255

        img = np.concatenate((img_white, img, img_white, img_target), axis=1)
        # add a row of white pixels at the bottom
        white_row = np.ones(shape=(tile_size, tile_size * self.y_dim * 2 + 2 * tile_size, 3), dtype=np.uint8) * 255
        img = np.concatenate((img, white_row), axis=0)

        if key_handler:
            if mode == "rgb_mode":
                # self.window.set_caption(self.mission)
                self.window.show_img(img)
                # self.window_target.show_img(img_target)
                # manually controlled
                self.window.reg_key_handler(key_handler)
                # self.window_target.reg_key_handler(key_handler)
                self.window.show(block=block)
                # self.window_target.show(block=block)

        return img
