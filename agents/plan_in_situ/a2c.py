
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import timm
import matplotlib.pyplot as plt
from copy import deepcopy
from iphyre.simulator import IPHYRE
from iphyre.games import PARAS as game_paras
import time
import logging
from train_online_RL import *
import pdb
import cv2


class Actor(nn.Module):
    def __init__(self,use_images,n_states,hidden_dim,n_actions):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        if use_images:
            self.fs = timm.create_model('resnet50d', pretrained=True, num_classes=self.hidden_dim).to(self.device)
        # self.fs.head = nn.Linear(768, hidden_dim)
        else:
            self.fs = nn.Sequential(
                nn.Linear(self.n_states, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                )

        self.fa = nn.Sequential(
            nn.Linear(self.n_actions * 2, self.hidden_dim),
            nn.ReLU(),
            )
        self.out = nn.Linear(self.hidden_dim, self.n_actions)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,states,actions): 
        states = self.fs(states)
        actions = self.fa(actions)
        x = states + actions
        out = self.out(x)
        probs = self.softmax(out)
        return probs

class Critic(nn.Module):
    def __init__(self,use_images,n_states,hidden_dim,n_actions):
        super(Critic, self).__init__()
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        if use_images:
            self.fs = timm.create_model('resnet50d', pretrained=True, num_classes=self.hidden_dim).to(self.device)
        # self.fs.head = nn.Linear(768, hidden_dim)
        else:
            self.fs = nn.Sequential(
                nn.Linear(self.n_states, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                )
        self.critic_back = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self,x):
        states = self.fs(x)
        value = self.critic_back(states)
        return value
    
class A2C(nn.Module):
    def __init__(self,device,config):
        super(A2C, self).__init__()
        self.device = device
        self.fps = config.fps
        self.max_iter = config.max_iter
        self.use_images = config.use_images
        self.game_time = config.game_time
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.hidden_dim = config.hidden_dim 
        self.lr = config.lr
        self.num_steps= config.num_steps
        self.max_frames = config.max_frames
        self.actor = Actor(self.use_images,self.n_states,self.hidden_dim,self.n_actions).to(self.device)
        self.critic = Critic(self.use_images,self.n_states,self.hidden_dim,self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam([{'params': self.actor.parameters(),'lr': self.lr}, {'params': self.critic.parameters(),'lr': self.lr}])
        self.max_frames = config.max_frames
        self.num_steps = config.num_steps
    
    def _forward(self, states, actions):
        value = self.critic(states)
        probs = self.actor(states,actions)
        dist  = Categorical(probs)
        return dist, value

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    
    def test(self,games):
        all_rewards = []
        all_done = []
        for game in games:
            env = IPHYRE(game=game,fps=self.fps)
            state = env.reset(self.use_images)
            if self.use_images:
                    state = cv2.resize(state, dsize=(224, 224)).transpose((2, 0, 1))
            else:
                state[:, :5] /= 600
            actions = env.get_action_space()
            input_actions = (torch.FloatTensor((deepcopy(actions))) / 600)
            input_actions = input_actions.reshape(1,-1).to(self.device)
            done = False
            total_reward = 0
            iter = 0
            while (not done) and iter < self.max_iter:
                iter+=1
                state = torch.FloatTensor(state).reshape(1,-1).to(self.device)
                dist, _ = self._forward(state,input_actions)
                a = dist.sample()
                pos = actions[a]
                next_state, reward, done = env.step(pos,use_images=self.use_images)
                state = next_state
                total_reward += reward
            all_rewards.append(total_reward)
        return all_rewards

    def train(self,train_split,mode):
        best_test_rewards = []
        for game in train_split:
            best_test_reward = -10000
            t1 = time.time()
            frame_idx = 0
            env = IPHYRE(game=game,fps=self.fps)
            state = env.reset(self.use_images)
            if self.use_images:
                state = cv2.resize(state, dsize=(224, 224)).transpose((2, 0, 1))
            else:
                state[:, :5] /= 600
            actions = env.get_action_space()
            input_actions = (torch.FloatTensor((deepcopy(actions))) / 600)
            input_actions = input_actions.reshape(1,-1).to(self.device)
            test_rewards = []
            while frame_idx < self.max_frames:
                log_probs = []
                values    = []
                rewards   = []
                masks     = []
                entropy = 0
                # rollout trajectory
                for _ in range(self.num_steps):
                    state = torch.FloatTensor(state).reshape(1,-1).to(self.device)
                    dist, value = self._forward(state,input_actions)
                    a = dist.sample()
                    pos = actions[a]
                    next_state, reward, done = env.step(pos,use_images=self.use_images)
                    if(done):
                        next_state = env.reset(self.use_images)
                    log_prob = dist.log_prob(a)
                    entropy += dist.entropy().mean()
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.FloatTensor([reward]).to(self.device))
                    masks.append(torch.FloatTensor([1 - done]).to(self.device))
                    
                    state = next_state
                    frame_idx += 1
                    if self.use_images:
                        state = cv2.resize(state, dsize=(224, 224)).transpose((2, 0, 1))
                    else:
                        state[:, :5] /= 600

                        
                next_state = torch.FloatTensor(next_state).reshape(1,-1).to(self.device)
                _, next_value = self._forward(next_state,input_actions)
                returns = self.compute_returns(next_value, rewards, masks)
                
                log_probs = torch.cat(log_probs)
                returns   = torch.cat(returns).detach()
                values    = torch.cat(values)

                advantage = returns - values

                actor_loss  = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if frame_idx % 100 == 0:
                    test_rewards+=self.test([game])
                    info = f'Testing Game: {game} | Frame idx: {frame_idx} | Reward: {test_rewards[-1]}'
                    print(info)
                    logging.info(info)
                    if best_test_reward < test_rewards[-1]:
                        best_test_reward = test_rewards[-1]
                    if len(test_rewards) > 5 and best_test_reward>0 and (best_test_reward == np.array(test_rewards[-5:])).all():
                        break
            
            best_test_rewards.append(best_test_reward)
        return best_test_rewards

                

            



def plot(frame_idx, rewards):
    plt.plot(rewards,'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.pause(0.0001)





