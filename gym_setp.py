import pip 

try:
    import gym
    import cv2
    import vizdoom
    from sympy import im
    from vizdoom import *
    import random
    import time
    import numpy as np
    import cv2
    from gym import Env
    from gym.spaces import Discrete, Box
    import os 
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3 import PPO
    from stable_baselines3.common import env_checker
    # from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecFrameStack
    from stable_baselines3.common.monitor import Monitor
except:
    pip.main(['install', 'gym',  'opencv-python', 'vizdoom', 'sympy', 'stable-baselines3'])
    import gym
    import cv2
    import gym

    
