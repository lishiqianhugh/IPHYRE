import sys
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import timm
import cv2
from games.simulator import IPHYRE
from games.game_paras import game_paras
import time
import logging
from main_situ import *
import pdb

class Net(nn.Module):
    def __init__(self, use_images, n_states, n_actions, hidden_dim):
        super(Net, self).__init__()
        self.use_images = use_images
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        if self.use_images:
            self.fs = timm.create_model('resnet50d', pretrained=True, num_classes=self.hidden_dim)
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


    def forward(self, states, actions):
        states = self.fs(states)
        actions = self.fa(actions)
        x = states + actions
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self,device,config):
        self.device = device
        self.epoch = config.epoch
        self.game_time = config.game_time
        self.use_images = config.use_images
        self.max_iter = config.max_iter
        self.learn_freq = config.learn_freq
        self.n_states = config.n_states
        if self.use_images:
            self.n_states = 224 * 224 * 3
        self.n_actions = config.n_actions
        self.hidden_dim = config.hidden_dim
        self.memory_capacity = config.memory_capacity
        self.lr = config.lr
        self.target_replace_iter = config.target_replace_iter
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_increment = config.epsilon_increment

        self.eval_net, self.target_net = \
                Net(self.use_images, self.n_states, self.n_actions, self.hidden_dim), \
                Net(self.use_images, self.n_states, self.n_actions, self.hidden_dim)
        self.eval_net.to(self.device)
        self.target_net.to(self.device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, self.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, states, actions, train):
        states = torch.unsqueeze(torch.FloatTensor(states), 0).to(device)
        actions = torch.unsqueeze(torch.FloatTensor(
            actions).reshape(-1), 0).to(self.device)
        if train == False or (train and np.random.uniform() < self.epsilon):
            actions_value = self.eval_net.forward(states, actions)
            action = torch.max(actions_value, 1)[1].cpu().detach().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)
        self.epsilon = min(0.99, self.epsilon + self.epsilon_increment)
        return action

    def store_transition(self, s, a, r, s_):
        if self.use_images:
            s = s.reshape(-1)
            s_ = s_.reshape(-1)
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, actions, done):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(self.device)
        b_a = torch.LongTensor(
            b_memory[:, self.n_states:self.n_states+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(self.device)
        if self.use_images:
            b_s = b_s.reshape((-1, 3, 224, 224))
            b_s_ = b_s_.reshape((-1, 3, 224, 224))

        actions = torch.unsqueeze(torch.FloatTensor(
            actions).reshape(-1), 0).to(self.device)
        q_eval = self.eval_net(b_s, actions).gather(1, b_a)
        q_next = self.target_net(b_s_, actions).detach()
        q_target = b_r + (done == False) * self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self,train_split):
        for game in train_split:
            t1 = time.time()
            env = IPHYRE(game=game)
            actions = env.get_action_space()
            input_actions = np.array(deepcopy(actions)) / 600
            print(input_actions)
            best_reward = -100000
            for i in range(self.epoch):
                total_reward = 0
                s = env.reset(self.use_images)
                if self.use_images:
                    s = cv2.resize(s, dsize=(224, 224)).transpose((2, 0, 1))
                else:
                    s[:, :5] /= 600
                    s = s.reshape(-1)
                iter = 0
                while iter < self.max_iter:
                    a = self.choose_action(s, input_actions, train=True)
                    pos = actions[a]
                    s_, r, done = env.step(pos, time_step=self.game_time / self.max_iter, use_images=self.use_images)
                    if self.use_images:
                        s_ = cv2.resize(s_, dsize=(224, 224)).transpose((2, 0, 1))
                    else:
                        s_[:, :5] /= 600
                        s_ = s_.reshape(-1)
                    # print(pos, r)
                    self.store_transition(s, a, r, s_)

                    total_reward += r
                    # total_reward += done * (args.max_iter - iter)
                    s = s_

                    if self.memory_counter > self.memory_capacity and (iter % self.learn_freq) == 0:
                        self.learn(input_actions, done)

                    iter += 1

                    if done:
                        break
                if total_reward > best_reward:
                    best_reward = total_reward
                info = f"Game: {game} | Epoch: {i} | Episode Reward: {total_reward}"
                print(info)

            t2 = time.time()
            info = f'Training Game: {game} | Learning Duration: {round(t2 - t1, 2)} | Best Reward: {best_reward}'
            print(info)
            logging.info(info)

    def test(self,test_split):
        rewards = []
        for game in test_split:
            t1 = time.time()
            env = IPHYRE(game=game)
            actions = env.get_action_space()
            input_actions = np.array(deepcopy(actions)) / 600
            best_reward = -100000
            total_reward = 0
            s = env.reset(self.use_images)
            if self.use_images:
                s = cv2.resize(s, dsize=(224, 224)).transpose((2, 0, 1))
            else:
                s[:, :5] /= 600
                s = s.reshape(-1)
            iter = 0
            while iter < self.max_iter:
                a = self.choose_action(s, input_actions, train=False)
                pos = actions[a]
                s_, r, done = env.step(pos, time_step=args.game_time / args.max_iter, use_images=use_images)
                if self.use_images:
                    s_ = cv2.resize(s_, dsize=(224, 224)).transpose((2, 0, 1))
                else:
                    s_[:, :5] /= 600
                    s_ = s_.reshape(-1)

                total_reward += r
                # total_reward += done * (args.max_iter - iter)
                s = s_

                iter += 1

                if done:
                    break
            if total_reward > best_reward:
                best_reward = total_reward

            t2 = time.time()
            rewards.append(best_reward)
            info = f'Testing Game: {game} | Testing Duration: {round(t2 - t1, 2)} | Reward: {best_reward}'
            print(info)
            logging.info(info)
        return rewards


        