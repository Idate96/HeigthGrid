#!/usr/bin/env python3
from heightgrid.envs.random_height import RandomHeightEnv5x5
from matplotlib.pyplot import grid
from heightgrid.envs.empty import EmptyEnv5x5
import time
import argparse
import numpy as np
import gym
import heightgrid.heightgrid as heightgrid
from heightgrid.wrappers import *
from heightgrid.window import Window



def policy(obs):
    return np.random.randint(0, 6)


n_step = 100
env = RandomHeightEnv5x5()
obs = env.reset()

for i in range(n_step):
    obs, reward, done, info = env.step(policy(obs))
    # env.render()
    # time.sleep(0.3)