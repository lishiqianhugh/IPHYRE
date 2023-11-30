import random
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from iphyre.simulator import IPHYRE
import numpy as np


class IPHYRE_inadvance(gym.Env):
    '''
    Plan in advance: generate one-step action times based on the initial scenes.
    '''
    def __init__(self, config):
        self.total_reward = 0
        self.game_list = config.get("game_list")
        self.seed = config.get("seed")
        random.seed(self.seed)
        self.game = None
        self.env = None
        self.action_list = None
        self.action_candidates = None
        self.cur_obs = None
        self.fps = 10
        self.game_time = 15.
        self.iter_len = 0
        self.action_space = Box(low=0., high=1., shape=(6,), dtype=np.float32)
        self.observation_space = Box(low=0., high=1., shape=(12 * 9 + 6 * 2,), dtype=np.float32)
        self.reset_num = 0

    def reset(self, *, seed=None, options=None):
        self.iter_len = 0
        self.game = random.choice(self.game_list)
        self.env = IPHYRE(game=self.game, fps=self.fps)
        self.total_reward = 0
        self.cur_obs = self.env.reset()
        self.action_list = self.env.get_action_space()[1:]
        self.action_candidates = np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        self.process()
        self.reset_num += 1
        return self.cur_obs, {}

    def step(self, action):
        '''
        :param action: the time sequence of executing each action (1 * 6)
        '''
        total_reward = 0
        terminated = False
        truncated = True
        tmp = np.round(action * self.game_time, 1)
        for time in np.round(np.arange(0, self.game_time, 1/self.fps), 1):
            if time > 0. and time in tmp:
                id = np.argwhere(tmp == time)[0][0]
                pos = self.action_list[id]
            else:
                pos = [0., 0.]
            self.cur_obs, reward, terminated = self.env.step(pos)
            total_reward += reward
            if terminated:
                truncated = False
                break
        self.process()
        return self.cur_obs, total_reward, terminated, truncated, {}

    def process(self):
        self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
        self.cur_obs[:, :5] /= 600
        self.cur_obs = self.cur_obs.clip(0., 1.)
        self.cur_obs = self.cur_obs.reshape(-1)
        self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))


class IPHYRE_onthefly(gym.Env):
    '''
    Plan on-the-fly: generate actions step by step based on the intermediate state.
    '''
    def __init__(self, config):
        self.total_reward = 0
        self.game_list = config.get("game_list")
        self.seed = config.get("seed")
        random.seed(self.seed)
        self.game = None
        self.env = None
        self.action_list = None
        self.action_candidates = None
        self.cur_obs = None
        self.fps = 10
        self.iter_len = 0
        self.action_space = Discrete(7)
        self.observation_space = Box(low=0., high=1., shape=(12 * 9 + 7 * 2,), dtype=np.float32)
        self.reset_num = 0

    def reset(self, *, seed=None, options=None):
        self.iter_len = 0
        self.game = random.choice(self.game_list)
        self.env = IPHYRE(game=self.game, fps=self.fps)
        self.total_reward = 0
        self.cur_obs = self.env.reset()
        self.action_list = self.env.get_action_space() 
        self.action_candidates = np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        self.process()
        self.reset_num += 1
        return self.cur_obs, {}

    def step(self, action):
        self.iter_len += 1
        pos = self.action_list[action]
        self.cur_obs, reward, terminated = self.env.step(pos)
        self.process()
        truncated = (self.iter_len >= 15 * self.fps)
        self.total_reward += reward
        return self.cur_obs, reward, terminated, truncated, {}

    def process(self):
        self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
        self.cur_obs[:, :5] /= 600
        self.cur_obs = self.cur_obs.clip(0., 1.)
        self.cur_obs = self.cur_obs.reshape(-1)
        self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))


class IPHYRE_combine(gym.Env):
    '''
    Combined strategy: generate one-step action times based on the initial scenes but update after each execution.
    '''
    def __init__(self, config):
        self.total_reward = 0
        self.game_list = config.get("game_list")
        self.seed = config.get("seed")
        random.seed(self.seed)
        self.game = None
        self.env = None
        self.action_list = None
        self.action_candidates = None
        self.cur_obs = None
        self.fps = 10
        self.game_time = 15.
        self.iter_len = 0
        self.action_space = Box(low=0., high=1., shape=(6,), dtype=np.float32)
        self.observation_space = Box(low=0., high=1., shape=(12 * 9 + 6 * 2,), dtype=np.float32)
        self.reset_num = 0
        self.mask = None

    def reset(self, *, seed=None, options=None):
        self.iter_len = 0
        self.game = random.choice(self.game_list)
        self.env = IPHYRE(game=self.game, fps=self.fps)
        self.total_reward = 0
        self.cur_obs = self.env.reset()
        self.action_list = self.env.get_action_space()[1:]
        self.action_candidates = np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        self.mask = np.ones((6,))
        self.process()
        self.reset_num += 1
        return self.cur_obs, {}

    def step(self, action):
        total_reward = 0
        terminated = False
        tmp = np.round(action * self.game_time, 1)
        for time in np.round(np.arange(self.iter_len/self.fps, self.game_time, 1/self.fps), 1):
            self.iter_len += 1
            truncated = (self.iter_len >= 15 * self.fps)
            if time > 0. and time in tmp:
                id = np.argwhere(tmp == time)[0][0]
                if self.mask[id]:
                    pos = self.action_list[id]
                    self.mask[id] = 0
                else:
                    pos = [0., 0.]
            else:
                pos = [0., 0.]
            self.cur_obs, reward, terminated = self.env.step(pos)
            total_reward += reward
            if terminated:
                truncated = False
                break
            if pos != [0., 0.]:
                break
        self.process()
        return self.cur_obs, total_reward, terminated, truncated, {}

    def process(self):
        self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
        self.cur_obs[:, :5] /= 600
        self.cur_obs = self.cur_obs.clip(0., 1.)
        self.cur_obs = self.cur_obs.reshape(-1)
        self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))
