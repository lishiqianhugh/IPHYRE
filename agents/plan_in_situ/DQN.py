import sys
sys.path.append('D:\Files\Research\Projects\Interactive_Physical_Reasoning\IPHYRE')
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import timm
import cv2

from games.simulator import IPHYRE
from games.game_paras import game_paras
from utils import setup_seed, write_csv
import time
import logging
import argparse
import optuna


def arg_parse():
    parser = argparse.ArgumentParser(description='DQN Parameters')
    parser.add_argument('--search', type=bool, help='whether searching hyperparameters', default=False)
    parser.add_argument('--mode', required=False, type=str, default='single',
                        choices=['single', 'within', 'cross'])
    parser.add_argument('--seed', type=int, help='training seed', default=0)
    parser.add_argument('--n_states', type=int, help='the dimension of game vectors', default=12 * 9)
    parser.add_argument('--use_images', type=bool, help='whether using image data', default=False)
    parser.add_argument('--n_actions', type=int, help='the number of candidate actions', default=7)
    parser.add_argument('--hidden_dim', type=int, help='of MLP', default=256)
    parser.add_argument('--game_time', type=float, help='total simulation time', default=15.)
    parser.add_argument('--max_iter', type=int, help='max iterations of games', default=150)
    parser.add_argument('--memory_capacity', type=int, help='the size of DQN pool', default=3000)
    parser.add_argument('--epsilon', type=float, help='balance r and Q', default=0.9)
    parser.add_argument('--epsilon_increment', type=float, help='reduce rate of epsilon', default=0)
    parser.add_argument('--gamma', type=float, help='balance explore and exploit', default=0.9)
    parser.add_argument('--target_replace_iter', type=int, help='interval to update Q-target Net', default=300)
    parser.add_argument('--learn_freq', type=int, help='learning frequency', default=5)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--epoch', type=int, help='training epoch', default=1000)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)

    return parser.parse_args()

args = arg_parse()

# device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print('Using', device)

# logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'logs/plan_in_situ/DQN_{args.mode}_{args.use_images}_{args.epoch}_{args.gamma}_{args.lr}.log', level=20,
                    format=LOG_FORMAT, datefmt=DATE_FORMAT)
logging.info(args)


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
    def __init__(self, use_images, n_states, n_actions, hidden_dim, memory_capacity, lr, target_replace_iter, batch_size, gamma, epsilon, epsilon_increment):
        self.use_images = use_images
        self.n_states = n_states
        if args.use_images:
            self.n_states = 224 * 224 * 3
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.memory_capacity = memory_capacity
        self.lr = lr
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_increment = epsilon_increment

        self.eval_net, self.target_net = \
                Net(self.use_images, self.n_states, self.n_actions, self.hidden_dim), \
                Net(self.use_images, self.n_states, self.n_actions, self.hidden_dim)
        self.eval_net.to(device)
        self.target_net.to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, self.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, states, actions, train):
        states = torch.unsqueeze(torch.FloatTensor(states), 0).to(device)
        actions = torch.unsqueeze(torch.FloatTensor(
            actions).reshape(-1), 0).to(device)
        if train == False or (train and np.random.uniform() < self.epsilon):
            actions_value = self.eval_net.forward(states, actions)
            action = torch.max(actions_value, 1)[1].cpu().detach().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)
        self.epsilon = min(0.99, self.epsilon + self.epsilon_increment)
        return action

    def store_transition(self, s, a, r, s_):
        if args.use_images:
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
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(device)
        b_a = torch.LongTensor(
            b_memory[:, self.n_states:self.n_states+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(device)
        if args.use_images:
            b_s = b_s.reshape((-1, 3, 224, 224))
            b_s_ = b_s_.reshape((-1, 3, 224, 224))

        actions = torch.unsqueeze(torch.FloatTensor(
            actions).reshape(-1), 0).to(device)
        q_eval = self.eval_net(b_s, actions).gather(1, b_a)
        q_next = self.target_net(b_s_, actions).detach()
        q_target = b_r + (done == False) * self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(model, train_split, use_images):
    for game in train_split:
        t1 = time.time()
        env = IPHYRE(game=game)
        actions = env.get_action_space()
        input_actions = np.array(deepcopy(actions)) / 600
        print(input_actions)
        best_reward = -100000
        for i in range(args.epoch):
            total_reward = 0
            s = env.reset(use_images)
            if use_images:
                s = cv2.resize(s, dsize=(224, 224)).transpose((2, 0, 1))
            else:
                s[:, :5] /= 600
                s = s.reshape(-1)
            iter = 0
            while iter < args.max_iter:
                a = model.choose_action(s, input_actions, train=True)
                pos = actions[a]
                s_, r, done = env.step(pos, time_step=args.game_time / args.max_iter, use_images=use_images)
                if args.use_images:
                    s_ = cv2.resize(s_, dsize=(224, 224)).transpose((2, 0, 1))
                else:
                    s_[:, :5] /= 600
                    s_ = s_.reshape(-1)
                # print(pos, r)
                model.store_transition(s, a, r, s_)

                total_reward += r
                # total_reward += done * (args.max_iter - iter)
                s = s_

                if model.memory_counter > args.memory_capacity and (iter % args.learn_freq) == 0:
                    model.learn(input_actions, done)

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

def test(model, test_split, use_images):
    rewards = []
    for game in test_split:
        t1 = time.time()
        env = IPHYRE(game=game)
        actions = env.get_action_space()
        input_actions = np.array(deepcopy(actions)) / 600
        best_reward = -100000
        total_reward = 0
        s = env.reset(use_images)
        if use_images:
            s = cv2.resize(s, dsize=(224, 224)).transpose((2, 0, 1))
        else:
            s[:, :5] /= 600
            s = s.reshape(-1)
        iter = 0
        while iter < args.max_iter:
            a = model.choose_action(s, input_actions, train=False)
            pos = actions[a]
            s_, r, done = env.step(pos, time_step=args.game_time / args.max_iter, use_images=use_images)
            if args.use_images:
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

def run(use_images=args.use_images,
        n_states=args.n_states,
        n_actions=args.n_actions,
        hidden_dim=args.hidden_dim,
        memory_capacity=args.memory_capacity,
        lr=args.lr,
        target_replace_iter=args.target_replace_iter,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_increment=args.epsilon_increment):
    setup_seed(args.seed)
    all_test_rewards = []
    FOLD_LIST = ['basic', 'compositional', 'noisy', 'multi_ball']
    for FOLD in FOLD_LIST:
        FOLD_ID = FOLD_LIST.index(FOLD)
        GAMES = list(game_paras.keys())
        NUM_GAMES = len(GAMES)
        NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
        TEST_SPLIT = GAMES[FOLD_ID * NUM_PER_GROUP: (FOLD_ID + 1) * NUM_PER_GROUP]
        TRAIN_SPLIT = GAMES[:FOLD_ID * NUM_PER_GROUP] + GAMES[(FOLD_ID + 1) * NUM_PER_GROUP:]

        logging.info(f'Fold: {FOLD}, Seed: {args.seed}')
        if args.mode == 'within':
            dqn = DQN(use_images, n_states, n_actions, hidden_dim, memory_capacity, lr, target_replace_iter, batch_size, gamma, epsilon, epsilon_increment)
            train(dqn, TEST_SPLIT, use_images)
            test_rewards = test(dqn, TEST_SPLIT, use_images)
            all_test_rewards += test_rewards
        elif args.mode == 'cross':
            dqn = DQN(use_images, n_states, n_actions, hidden_dim, memory_capacity, lr, target_replace_iter, batch_size, gamma, epsilon, epsilon_increment)
            train(dqn, TRAIN_SPLIT, use_images)
            test_rewards = test(dqn, TEST_SPLIT, use_images)
            all_test_rewards += test_rewards
        elif args.mode == 'single':
            for game in TEST_SPLIT:
                dqn = DQN(use_images, n_states, n_actions, hidden_dim, memory_capacity, lr, target_replace_iter, batch_size, gamma, epsilon, epsilon_increment)
                train(dqn, [game], use_images)
                test_rewards = test(dqn, [game], use_images)
                all_test_rewards += test_rewards
        else:
            raise ValueError(f'No such mode {args.mode}')
    
    return all_test_rewards

def objective(trial):
    gamma = trial.suggest_float('gamma', 0.5, 1.)
    epsilon = trial.suggest_float('epsilon', 0.5, 1.)
    lr = trial.suggest_float('lr', 0.001, 0.01)
    target_replace_iter = trial.suggest_int('target_replace_iter', 50, 200, 50)
    all_test_rewards = run(use_images=args.use_images,
                        n_states=args.n_states,
                        n_actions=args.n_actions,
                        hidden_dim=args.hidden_dim,
                        memory_capacity=args.memory_capacity,
                        lr=lr,
                        target_replace_iter=target_replace_iter,
                        batch_size=args.batch_size,
                        gamma=gamma,
                        epsilon=epsilon,
                        epsilon_increment=args.epsilon_increment)

    return sum(all_test_rewards)


if __name__ == '__main__':
    if args.search == False:
        all_test_rewards = run()
        write_csv(path=f'logs/plan_in_situ/DQN_{args.mode}_{args.use_images}_{args.epoch}_{args.gamma}_{args.lr}_rewards.csv', contents=[list(game_paras.keys()), all_test_rewards])
    else:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        print('best params:', study.best_trial.params,
              '\n', 'best reward:', study.best_trial.values)
        