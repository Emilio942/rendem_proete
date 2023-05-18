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



# # Setup game and actions
# game = DoomGame()
# game.load_config("/home/emilio/Documents/py/work/vizdoom/scenarios/basic.cfg")
# game.init()
# # This is the set of actions we can take in the environment
# actions = np.identity(3,dtype=np.uint8)
# random.choice(actions)

# episodes = 10
# for epi in range(episodes):
#     print("Episode #" + str(epi))
#     game.new_episode()
#     while not game.is_episode_finished():
#         state = game.get_state()
#         img = state.screen_buffer
#         misc = state.game_variables
#         action = random.choice(actions)
#         reward = game.make_action(action, 4)
#         print("State #" + str(state.number))
#         time.sleep(0.02)
#     print("Result:", game.get_total_reward())
#     time.sleep(2)


class VizdoomEnv(Env):

    def __init__(self,render=False):
        super(VizdoomEnv, self).__init__()
        self.game = DoomGame()
        self.game.load_config("/home/emilio/Documents/py/work/vizdoom/scenarios/defend_the_center.cfg")
        
        
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init()
        
        self.actions = np.identity(3, dtype=np.uint8)
        
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        
        reward = self.game.make_action(self.actions[action], 4)
        
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
        
            info = ammo
                   

        else:
            state = np.zeros(self.observation_space.shape)
            info:int= 0
        
        info = {'info':info}
        
        done = self.game.is_episode_finished()
        
        return state ,reward, done, info

    def render(self):
        pass

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

    

    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_RGB2GRAY)
        resize  = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_AREA)
        state = np.reshape(resize, (100, 160, 1))
        return state
    
    def close(self):
        return self.game.close()
    
class TrainCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(TrainCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.log_dir, "best_model_{}".format(self.n_calls))
            self.model.save(model_path)
        return True
          # Retrieve training reward


# model.learn(total_timesteps=1000000, callback=callback)
if  __name__ == "__main__":
    env = VizdoomEnv()
    CHECKPOINT = './train/train_defend_center'
    LOG_DIR = './train/logs_defend_center'
    callback = TrainCallback(check_freq=10000, log_dir=CHECKPOINT)
    model = PPO('CnnPolicy', env=env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001,n_steps=256)

    model.learn(total_timesteps=1000000 , callback=callback)    
    
    env.reset()
    time.sleep(2)
    env.close()

# from stable_baselines3.common.evaluation import evaluate_policy
# model = PPO.load('/home/emilio/Documents/py/work/vizdoom/my/train/train_basic/best_model_350000.zip')   
# env = VizdoomEnv()