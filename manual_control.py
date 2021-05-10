#!/usr/bin/env python3

from heightgrid.envs.hole import Hole, HoleEnv5x5
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

def redraw():
    if not args.agent_view:
        img, img_target = env.render('rgb_array', tile_size=args.tile_size)

    env.window.show_img(img)
    env.window_target.show_img(img_target)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        env.window.set_caption(env.mission)

    redraw()

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw()

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        env.window.close()
        env.window_target.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        obs, reward, done, info = env.step(env.actions.left)
        parse_step(obs, reward, done, info)
        return
    if event.key == 'right':
        obs, reward, done, info = env.step(env.actions.right)
        parse_step(obs, reward, done, info)
        return
    if event.key == 'up':
        obs, reward, done, info = env.step(env.actions.forward)
        parse_step(obs, reward, done, info)

        return

    # Spacebar
    if event.key == ' ':
        obs, reward, done, info  = env.step(env.actions.toggle)
        parse_step(obs, reward, done, info)
        return

    if event.key == 'pageup':
        print(step(env.actions.dig))
        obs, reward, done, info  = env.step(env.actions.dig)
        parse_step(obs, reward, done, info)
        return
    if event.key == 'pagedown':
        obs, reward, done, info  = env.step(env.actions.drop)
        parse_step(obs, reward, done, info)
        return

    if event.key == 'enter':
        obs, reward, done, info = step(env.actions.done)
        return


def parse_step(obs, reward, done, info):
    redraw()
    # print("Observations [:, :, 0] \n", obs['image'][:, :, 0])
    # print("Observations [:, :, 1] \n", obs['image'][:, :, 1])
    # print("Observations [:, :, 2] \n", obs['image'][:, :, 2])
    # print("obs \n", obs) 
    # print("obs \n", obs)

    print("Reward :", reward)

    print("Done: ", done)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='HeightGrid-Empty-5x5-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

# grid_height = np.zeros((5,5))
# grid_height[1, 3] = 1
# env = gym.make(args.env)
# env = EmptyEnv5x5()
env = HoleEnv5x5()
env.reset()
if args.agent_view:
    env = FullyObsWrapper(env)
    # env = ImgObsWrapper(env)

# window = Window('heightgrid - ' + args.env)
env.render(block=True, key_handler=key_handler)
# env.window.reg_key_handler(key_handler)
# env.window_target.reg_key_handler(key_handler)

# reset()

# # Blocking event loop
# env.window.show(block=True)
# env.window_target.show(block=True)
