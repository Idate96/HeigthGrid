import unittest
import numpy as np 
from heightgrid.heightgrid import Grid

class TestGrid(unittest.TestCase):

    def test_grid_init(self):
        zero_heightmap = np.zeros((6, 6))
        grid = Grid(zero_heightmap)



if __name__ == '__main__':
    unittest.main()