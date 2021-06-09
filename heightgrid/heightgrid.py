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


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

    # dig in the cell in from of you
    dig = 3
    # Drop an soil that was dug
    drop = 4
    # make a ramp
    # toggle = 5

    # Done completing task
    # done = 6


eps = 10 ** -7

SIZE_TILE_PIXELS = 32

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

DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
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
        grid_height: np.ndarray,
        target_grid_height: np.ndarray,
        step_cost: float = -0.05,
        max_steps: int = 256,
        seed=24,
        mask: bool = False,
        collition_cost:float = 0
    ) -> None:
        super().__init__()
        assert np.shape(grid_height) == np.shape(target_grid_height)
        # actions are discrete
        self.action_space = spaces.Discrete(len(Actions))
        # print(len(Actions))
        self.actions = Actions
        self.num_actions = len(Actions)
        # these values are not modified during run time
        self.grid_height = grid_height
        self.height, self.width = grid_height.shape[0], grid_height.shape[1]
        self.grid_target = target_grid_height

        self.x_dim, self.y_dim = np.shape(grid_height)
        # ordering lateral and then up
        self.grid_object = [None] * (self.x_dim * self.y_dim)
        # encode position and orientation of objects
        # self.grid_object_pose = np.zeros((self.x_dim, self.y_dim, 2))
        self.carrying = 0

        # current grid height, relative target height, current obj pos, current obj orientation
        if mask:
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=-10,
                        high=10,
                        shape=(*np.shape(grid_height), 3),
                        dtype=np.int8,
                    ),
                    "vector": spaces.Box(low=0, high=1, shape=(3,), dtype=np.uint8),
                    "mask": spaces.MultiBinary(len(Actions)),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=-10,
                        high=10,
                        shape=(*np.shape(grid_height), 3),
                        dtype=np.int8,
                    ),
                    "vector": spaces.Box(low=0, high=1, shape=(3,), dtype=np.uint8),
                }
            )
        # self.observation_space = spaces.Box(low=0, high=255, shape=(*np.shape(grid_height), 4), dtype=np.uint8)
        self.obs = np.zeros((self.x_dim, self.y_dim, 4))

        self.mask = mask
        self.max_steps = max_steps
        self.collision_cost = collision_cost

        self.seed(seed=seed)

        self.obs = np.zeros((*np.shape(grid_height), 4))
        # array with two coordinates
        self.agent_pos = None
        # 0 up, 1 right, 2 down, 3 left (clockwise)
        self.agent_dir = None

        self.bucket_full = None

        self.number_dig_sites = np.sum(np.abs(target_grid_height - grid_height))
        # bucket full
        self.carrying = 0
        self.step_count = 0
        self.step_cost = step_cost
        # agent has to reach a goal
        self._goal = False

        self.window = None
        self.window_target = None

    @property
    def grid_object_pose(self):
        return self.obs[:, :, 2:]

    @grid_object_pose.setter
    def grid_object_pose(self, value):
        self.obs[:, :, 2:] = value

    @property
    def grid_height_curr(self):
        return self.obs[:, :, 0]

    @grid_height_curr.setter
    def grid_height_curr(self, value):
        self.obs[:, :, 0] = value

    @property
    def grid_target_rel(self):
        return self.obs[:, :, 1]

    @grid_target_rel.setter
    def grid_target_rel(self, value):
        self.obs[:, :, 1] = value

    @property
    def front_pos(self):
        if self.agent_dir == 0:
            front_pos = self.agent_pos + np.array([0, 1])
        elif self.agent_dir == 1:
            front_pos = self.agent_pos + np.array([1, 0])
        elif self.agent_dir == 2:
            front_pos = self.agent_pos + np.array([0, -1])
        elif self.agent_dir == 3:
            front_pos = self.agent_pos + np.array([-1, 0])

        if front_pos[0] > self.x_dim or front_pos[1] > self.y_dim:
            front_pos = None
        return front_pos

    @property
    def action_mask(self):
        available_actions = np.ones((self.num_actions,), dtype=np.uint8)

        fwd_pos = self.front_pos

        if self.in_bounds(fwd_pos):
            # Get the contents of the cell in front of the agent
            if self.carrying:
                available_actions[self.actions.dig] = 0
            else:
                available_actions[self.actions.drop] = 0

            if self.obs[fwd_pos[0], fwd_pos[1], 1] > -0.5:
                available_actions[self.actions.dig] = 0

            if self.obs[fwd_pos[0], fwd_pos[1], 1] < 0.5:
                available_actions[self.actions.drop] = 0
        else:
            available_actions[self.actions.dig] = 0
            available_actions[self.actions.drop] = 0

        return available_actions

    def __str__(self):
        repre = "Height " + 10 * "-" + "\n"
        repre += np.array2string(self.obs[:, :, 0])
        repre += "\nRelative Height " + 10 * "-" + "\n"
        repre += np.array2string(self.obs[:, :, 1])
        repre += "\nObjects " + 10 * "-" + "\n"
        repre += np.array2string(self.obs[:, :, 2])
        repre += (
            "Agent orientation "
            + np.array2string(DIR_TO_VEC[int(self.agent_dir)])
            + "\n"
        )
        repre += "Full bucket " + str(self.carrying)
        return repre

    @property
    def _observation(self):
        vector_obs = np.insert(DIR_TO_VEC[self.agent_dir], self.carrying, 2)

        if self.mask:
            obs = {
                "image": self.obs[:, :, :3],
                "vector": vector_obs,
                "mask": self.action_mask,
            }
        else:
            obs = {
                "image": self.obs[:, :, :3],
                "vector": vector_obs,
            }

        return obs

    def reset(self, agent_pose: tuple = (0, 0, 0)):
        # add agent to the observation space
        self.grid_height_curr = self.grid_height
        self.grid_target_rel = self.grid_target - self.grid_height
        self.grid_object_pose = np.zeros((*np.shape(self.grid_height), 1))

        self.agent_pos = np.array(agent_pose[:2])
        self.grid_object_pose[self.agent_pos[0], self.agent_pos[1], 0] = OBJECT_TO_IDX[
            "agent"
        ]
        self.agent_dir = agent_pose[2]
        self.grid_object_pose[self.agent_pos[0], self.agent_pos[1], 1] = self.agent_dir

        # add agent to the obj list for heightgrid.rendering
        self.grid_object = self.x_dim * self.y_dim * [None]
        self.place_obj_at_pos(AgentObj(), self.agent_pos)
        self.step_count = 0

        self.number_dig_sites = np.sum(np.abs(self.grid_target_rel))
        # print(self.grid_object)
        return self._observation

    def update_grid(self, grid_height: np.array, target_grid_height: np.array):
        """Vanilla implementation of the grid. More advanced world shoud consider randomization

        Returns:
            Union[np.nparray, np.array, np.array]: current grid height, relative target grid height and pose of the agent
        """
        self.grid_height = grid_height
        self.target = target_grid_height

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
            self.grid_object_pose[pos[0], pos[1], 0] = OBJECT_TO_IDX[obj.type]
            # orientation of the object same as the one of the agent
            # if obj.orientable:
            #     self.grid_object_pose[:, : , 1] = self.agent_dir
        else:
            warnings.warn(
                "The provided location (%d, %d) is already occupied by another object".format(
                    pos[0], pos[1]
                )
            )

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
            self.grid_object_pose[pos[0], pos[1], 0] = 0
            # self.grid_object_pose[pos[0], pos[1], 1] = 0

    def is_traversable(self, current_pos, target_pos):
        # if different heights not traversable
        heigth_diff = (
            -self.obs[current_pos[0], current_pos[1], 0]
            + self.obs[target_pos[0], target_pos[1], 0]
        )
        # to get higher in evelation ramp is in front of you
        if heigth_diff > (0 - eps):
            ramp_present = type(self.get(*target_pos)) == Ramp
        # get lower in elevation you must be on top of ramp
        elif heigth_diff < 0:
            ramp_present = type(self.get(*current_pos)) == Ramp
        traversable = (np.abs(heigth_diff) - ramp_present) < 10 ** -7
        return traversable

    def can_dig(self, current_pos, target_pos):
        diggable = (
            np.abs(
                (
                    self.obs[current_pos[0], current_pos[1], 0]
                    - self.obs[target_pos[0], target_pos[1], 0]
                )
            )
            < 1 + 10 ** -7
        )
        no_objects = self.get(*target_pos) is None
        return diggable and no_objects

    def can_drop(self, current_pos, target_pos):
        height_diff = (
            self.obs[target_pos[0], target_pos[1], 0]
            - self.obs[current_pos[0], current_pos[1], 0]
        )
        # can drop in into hole or arbitrary hight but can't drop if height of tharget is higher
        return height_diff < 1

    def in_bounds(self, pos):
        x_bounded = 0 <= pos[0] < self.x_dim
        y_bounded = 0 <= pos[1] < self.y_dim
        return x_bounded and y_bounded

    def dig_reward(self):
        current_dig_sites = np.sum(np.abs(self.grid_target_rel))
        reward = self.number_dig_sites - current_dig_sites
        self.number_dig_sites = current_dig_sites
        # print("dig reward ", reward)
        return reward

    def get_height(self, pos):
        return self.obs[pos[0], pos[1], 0]

    def move_obj_pos(self, pos, fwd_pos, obj):
        self.remove_obj_at_pos(pos)
        self.obs[pos[0], pos[1], 2] = 0
        self.obs[fwd_pos[0], fwd_pos[1], 2] = OBJECT_TO_IDX[obj.type]
        self.place_obj_at_pos(obj, fwd_pos)

    def move_agent_pos(self, fwd_pos):
        self.move_obj_pos(self.agent_pos, fwd_pos, self.get(*self.agent_pos))
        self.agent_pos = fwd_pos

    def place_ramp(self, curret_pos, target_pos):
        # if a block has a higher height you can modify it to create a ramp
        # if a block has lower height you cannot modify it to create a ramp
        if -eps < self.get_height(target_pos) - self.get_height(curret_pos) < 1 + eps:
            self.place_obj_at_pos(Ramp(), target_pos)

    def step(self, action):
        self.step_count += 1
        # living negative reward to encourage shortest trajectory
        reward = self.step_cost
        done = False

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        if self.in_bounds(fwd_pos):
            # Get the contents of the cell in front of the agent
            fwd_cell = self.get(*fwd_pos)

            # Move forward
            if action == self.actions.forward:
                # check if height difference allows for passing though
                if self.is_traversable(self.agent_pos, fwd_pos):
                    # check if objects prevents moving forward, walls for example cannot be overlapped
                    if fwd_cell is None:
                        self.move_agent_pos(fwd_pos)
                    else:
                        if fwd_cell.can_overlap:
                            self.move_agent_pos(fwd_pos)

                        if fwd_cell.type == "goal":
                            done = True
                            # print("Goal Reached")
                            reward = 1
                else:
                    # for the moment restart on collision or fall into hole
                    if type(fwd_cell) != Ramp:
                        reward -= self.collision_cost

            # dig soil
            elif action == self.actions.dig:
                if self.can_dig(self.agent_pos, fwd_pos):
                    # can dig if bucket is empty
                    if self.carrying != 1:
                        # remove one unit of soild
                        if self.obs[fwd_pos[0], fwd_pos[1], 1] < -0.5:
                            # print("dig")
                            self.obs[fwd_pos[0], fwd_pos[1], 0] -= 1
                            self.grid_target_rel = self.grid_target - self.obs[:, :, 0]
                            # bucket full
                            self.carrying = 1
                            # +1 if dug were supposed, -1 otherwise
                            reward += self.dig_reward()/10

            # Dump soil an object
            elif action == self.actions.drop:
                if self.carrying == 1:
                    if not fwd_cell and self.can_dig(self.agent_pos, fwd_pos):
                        if self.obs[fwd_pos[0], fwd_pos[1], 1] > 0.5:
                            # increase height map where soil is dumped
                            # print("dropp")
                            self.obs[fwd_pos[0], fwd_pos[1], 0] += 1
                            self.grid_target_rel = self.grid_target - self.obs[:, :, 0]

                            # bucket is empty
                            self.carrying = 0
                            reward += self.dig_reward()/10

        #     # Toggle/activate an object
        #     elif action == self.actions.toggle:
        #         if fwd_cell is None:
        #             self.place_ramp(self.agent_pos, fwd_pos)
        #         else:
        #             self.remove_obj_at_pos(fwd_pos)

        # # Done action (not used by default)
        # elif action == self.actions.done:
        #     pass
        # finished the escavation project
        if not self._goal:
            if np.sum(np.abs(self.obs[:, :, 1])) < eps:
                # print("Done excavation")
                # print("done")
                done = True
                reward = 1

        if self.step_count >= self.max_steps:
            done = True

        return self._observation, reward, done, {}

    @classmethod
    def render_tile(
        cls, obj, height, agent_dir=None, tile_size=SIZE_TILE_PIXELS, subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache

        key = str(tile_size) + "h" + str(height)
        key = obj.type + key if obj else key
        # print(key, obj)
        # key = obj.encode() if obj else key

        if key in cls.tile_cache and obj is None:
            # print("cached")
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)

        if obj != None:
            obj.render(img)
        else:
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
        if agent_dir is not None:
            tri_fn = heightgrid.rendering.point_in_triangle(
                (0.12, 0.81),
                (0.12, 0.19),
                (0.87, 0.50),
            )

            # Rotate the agent based on its direction
            tri_fn = heightgrid.rendering.rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=np.pi / 2 - 0.5 * np.math.pi * agent_dir
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
        agent_dir=None,
        render_objects=True,
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        # print(self)
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

                agent_here = np.array_equal(agent_pos, (i, j))
                # if agent_here:
                #     # print("agent here : ({}, {}) {}" .format(i, j, agent_here))
                #     print(cell)

                tile_img = self.render_tile(
                    cell,
                    height_grid[i, j],
                    agent_dir=agent_dir if agent_here else None,
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img.transpose(1, 0, 2)

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

        if close:
            if self.window:
                self.window.close()
            if self.window_target:
                self.window_target.close()
            return

        if mode == "rgb_mode" and not self.window:
            self.window = heightgrid.window.Window("heightgrid")
            self.window_target = heightgrid.window.Window("relative target")

        # Render the whole grid
        img = self.render_grid(
            tile_size, self.obs[:, :, 0], self.agent_pos, self.agent_dir
        )

        img_target = self.render_grid(
            tile_size,
            self.obs[:, :, 1],
            self.agent_pos,
            self.agent_dir,
            render_objects=True,
        )
        img = np.concatenate((img, img_target), axis=0)

        if mode == "rgb_mode":
            # self.window.set_caption(self.mission)
            self.window.show_img(img)
            # self.window_target.show_img(img_target)
            # manually controlled
            if key_handler:
                self.window.reg_key_handler(key_handler)
                # self.window_target.reg_key_handler(key_handler)
                self.window.show(block=block)
                # self.window_target.show(block=block)

        return img
