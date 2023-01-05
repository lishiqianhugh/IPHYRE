import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical
import timm
import matplotlib.pyplot as plt
from copy import deepcopy
from iphyre.simulator import IPHYRE
from iphyre.games import PARAS as game_paras
import time
from collections import deque
import random
from train_online_RL import *
from agents.plan_in_situ.a2c import *
import pdb
import cv2
from copy import deepcopy
torch.autograd.set_detect_anomaly(True)
import sys



# the action_dim is 1 x 7, representing the normalized timestep of each action taken. 
class DDPG_Actor(Actor):
    def __init__(self,use_images,n_states,hidden_dim,n_actions):
        super(DDPG_Actor, self).__init__(use_images,n_states,hidden_dim,n_actions)
        self.relu = nn.ReLU()
    
    @staticmethod    
    def normalize(x):
        min = x.min()
        max = x.max()
        norm_x = (max - x) / (max - min)
        return norm_x

    def forward(self,states,actions): 
        states = self.fs(states)
        actions = self.fa(actions)
        x = states + actions
        x = self.out(x)
        #out = normalize(self.relu(x))
        out = self.normalize(x)
        return out
        # the output represents the excecuting timestep of each action
class DDPG_Critic(Critic):
    def __init__(self,use_images,n_states,hidden_dim,n_actions):
        super(DDPG_Critic, self).__init__(use_images,n_states,hidden_dim,n_actions)
        self.fa = nn.Sequential(
            nn.Linear(self.n_actions, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            )
        
    def forward(self,states,action): 
        states = self.fs(states)
        action = self.fa(action)
        x = states + action
        q = self.critic_back(x)
        return q

class OUNoise(object):
    def __init__(self, device, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.device = device
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = torch.FloatTensor(self.evolve_state()).to(self.device)
        new_action = DDPG_Actor.normalize(ou_state+action)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return new_action
     
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        # index_list = list(range(self.max_size))
        # indices = np.random.choice(index_list, batch_size)
        # pdb.set_trace()
        # state_batch = self.buffer[0][indices]
        # action_batch = self.buffer[1][indices]
        # reward_batch = self.buffer[2][indices]
        # next_state_batch = self.buffer[3][indices]
        # done_batch = self.buffer[4][indices]
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class DDPG:
    def __init__(self,device,config):
        # Params
        self.device = device
        self.batch_size = config.batch_size
        self.memory_capacity = config.memory_capacity
        self.use_images = config.use_images
        self.game_time = config.game_time
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.gamma = config.gamma
        self.tau = config.tau
        self.hidden_dim = config.hidden_dim
        self.actor_lr = config.actor_lr
        self.critic_lr = config.critic_lr
        self.episode = config.episode
        self.max_iter = config.max_iter
        # Networks
        self.actor = DDPG_Actor(self.use_images,self.n_states,self.hidden_dim,self.n_actions).to(self.device)
        self.actor_target = DDPG_Actor(self.use_images,self.n_states,self.hidden_dim,self.n_actions).to(self.device)
        self.critic = DDPG_Critic(self.use_images,self.n_states,self.hidden_dim,self.n_actions).to(self.device)
        self.critic_target = DDPG_Critic(self.use_images,self.n_states,self.hidden_dim,self.n_actions).to(self.device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(self.memory_capacity)  
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
    
    def get_action(self, state,input_action):
        state = Variable(state)
        action = self.actor.forward(state,input_action)
        # action = action.detach().numpy()[0,0]
        return action
    
    def update(self,batch_size,action_space,mask):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).squeeze(1).to(self.device)
        actions = torch.FloatTensor(actions).squeeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).squeeze(1).to(self.device)
        action_space = action_space.repeat(batch_size,1).detach().clone()
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states,action_space)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        #first update the critic parameter 
        # Actor loss
        new_action = self.actor.forward(states,action_space)*mask
        policy_loss = -self.critic.forward(states, new_action).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def test(self,games):
        all_rewards = []
        for game in games:
            env = IPHYRE(game=game)
            state = env.reset(self.use_images)
            action_length = len(game_paras[game]['eli'])
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
            state = env.reset()
            noise = OUNoise(self.device,self.n_actions)
            noise.reset()
            state = torch.FloatTensor(state).reshape(1,-1).to(self.device)
            mask = torch.arange(0,self.n_actions) < action_length
            action_times = self.actor(state,input_actions)
            action_times = noise.get_action(action_times)* mask
            iter = 0
            time_step = self.game_time / self.max_iter
            for time in np.arange(0,self.game_time,time_step):
                if done: break
                if action_times[0][iter]!=0 and action_times[0][iter]*self.game_time >= time:
                    iter+=1
                    pos = actions[iter]
                else:
                    pos = [0., 0.]
                next_state, reward, done = env.step(pos,use_images=self.use_images)
                total_reward += reward
            all_rewards.append(total_reward)
        return all_rewards

    def train(self,train_split):
        for game in train_split:
            frame_idx = 0
            action_length = len(game_paras[game]['eli'])
            env = IPHYRE(game=game)
            state = env.reset(self.use_images)
            if self.use_images:
                state = cv2.resize(state, dsize=(224, 224)).transpose((2, 0, 1))
            else:
                state[:, :5] /= 600
            actions = env.get_action_space()
            input_actions = (torch.FloatTensor((deepcopy(actions))) / 600)
            input_actions = input_actions.reshape(1,-1).to(self.device)
            noise = OUNoise(self.device,self.n_actions)
            rewards = []
            avg_rewards = []
            test_rewards = []
            for episode in range(self.episode):
                state = env.reset()
                noise.reset()
                episode_reward = 0
                state = torch.FloatTensor(state).reshape(1,-1).to(self.device)
                mask = torch.arange(0,self.n_actions) < action_length
                action_times = self.actor(state,input_actions)
                action_times = noise.get_action(action_times)* mask
                iter = 0
                time_step = self.game_time / self.max_iter
                done = False
                for time in np.arange(0,self.game_time,time_step):
                    if done: break
                    if action_times[0][iter]!=0 and action_times[0][iter]*self.game_time >= time:
                        iter+=1
                        pos = actions[iter]
                    else:
                        pos = [0., 0.]
                    next_state, reward, done = env.step(pos,use_images=self.use_images)
                    
                next_state = torch.FloatTensor(next_state).reshape(1,-1).to(self.device)
                self.memory.push(state.detach().numpy(), action_times.detach().numpy(), reward, next_state.detach().numpy(), done)
                if len(self.memory) > self.batch_size:
                    self.update(self.batch_size,input_actions,mask)                    
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                
                rewards.append(reward)
                avg_rewards.append(np.mean(rewards[-10:]))
                if episode % 100 == 0:
                        test_rewards.append(np.mean([self.test([game])[0] for _ in range(10)]))
                        info = f'Testing Game: {game} | Frame idx: {frame_idx} | Reward: {test_rewards[-1]}'
                        print(info)
                        logging.info(info)
                if test_rewards[-1]<np.mean(test_rewards):
                    break
