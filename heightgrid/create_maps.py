import numpy as np


def create_rectangle(map, hole_size, start_idx, value):
    # let's make the hole a nice 3x2 block
    coords = [np.arange(start_idx[i], start_idx[i] + hole_size[i]) for i in range(2)]
    i = np.repeat(coords[0], hole_size[1])
    j = np.tile(coords[1], hole_size[0])
    hole_idx = (i, j)
    map[hole_idx] = value
    return map


if __name__ == '__main__':
    target_map = np.zeros((5, 5))
    target_map = create_rectangle(target_map, (2, 3), (1, 1), value=-1)
    target_map = create_rectangle(target_map, (2, 3), (3, 1), value=1)
    print(target_map)
