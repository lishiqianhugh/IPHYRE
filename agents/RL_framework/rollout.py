import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils.environment.wrappers import *
import time
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pdb


class RolloutBuffer():
    def __init__(self, rollout_size, obs_size, device):
        self.rollout_size = rollout_size
        self.obs_size = obs_size
        self.device = device
        self.reset()
        
    def insert(self, step, done, action, log_prob, reward, obs,mask):    
        self.done[step].copy_(done)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.obs[step].copy_(obs)
        self.mask[step].copy_(mask)
        
    def reset(self):
        self.done = torch.zeros(self.rollout_size, 1).to(self.device)
        self.returns = torch.zeros(self.rollout_size + 1, 1, requires_grad=False).to(self.device)
        # Assuming Discrete Action Space
        self.actions = torch.zeros(self.rollout_size, 1, dtype=torch.int64).to(self.device)
        self.log_probs = torch.zeros(self.rollout_size, 1).to(self.device)
        self.rewards = torch.zeros(self.rollout_size, 1).to(self.device)
        self.obs = torch.zeros(self.rollout_size, self.obs_size).to(self.device)
        self.mask = torch.zeros(self.rollout_size, 4).to(self.device)
        
    def compute_returns(self, gamma):
        # Compute Returns until the last finished episode
        if((self.done==1).any()):
            self.last_done = (self.done == 1).nonzero().max()  
        else:
            self.last_done = self.rollout_size - 1
        self.returns[self.last_done + 1] = 0.

        # Accumulate discounted returns
        #so the return[step] = reward[step]+return[step+1]*gamma*(1-done[step]) a new way to calculate discounted reward?
        for step in reversed(range(self.last_done + 1)):
            self.returns[step] = self.returns[step + 1] * \
                gamma * (1 - self.done[step]) + self.rewards[step]
        
    def batch_sampler(self, batch_size, get_old_log_probs=False):
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.last_done)),
            batch_size,
            drop_last=True)
        for indices in sampler:
            if get_old_log_probs:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.log_probs[indices], self.mask[indices]
            else:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.mask[indices]
