import sys
sys.path.append('D:\Files\Research\Projects\Interactive_Physical_Reasoning\IPHYRE')
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from games.simulator import IPHYRE
from games.game_paras import game_paras
from utils import setup_seed, write_csv
import time
import logging

MODE = 'single'  # single, within or cross
SEED = 0
N_STATES = 12 * 9
N_ACTIONS = 7
HIDDEN_DIM = 256
GAME_TIME = 15.
MAX_ITER = 150
MEMORY_CAPACITY = 200
EPSILON = 0.7
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
BATCH_SIZE = 32
LR = 0.01
EPOCH = 10
# device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print('Using', device)

# logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'logs/plan_in_situ/DQN_{MODE}_{EPOCH}.log', level=20,
                    format=LOG_FORMAT, datefmt=DATE_FORMAT)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fs = nn.Linear(N_STATES, HIDDEN_DIM)
        self.fs.weight.data.normal_(0, 0.1)
        self.fa = nn.Linear(N_ACTIONS * 2, HIDDEN_DIM)
        self.fa.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(HIDDEN_DIM, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, states, actions):
        states = F.relu(self.fs(states))
        actions = F.relu(self.fa(actions))
        x = states + actions
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.to(device)
        self.target_net.to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, states, actions, epsilon):
        states = torch.unsqueeze(torch.FloatTensor(states), 0).to(device)
        actions = torch.unsqueeze(torch.FloatTensor(
            actions).reshape(-1), 0).to(device)
        if np.random.uniform() < epsilon:
            actions_value = self.eval_net.forward(states, actions)
            action = torch.max(actions_value, 1)[1].cpu().detach().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, actions):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(
            b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        actions = torch.unsqueeze(torch.FloatTensor(
            actions).reshape(-1), 0).to(device)
        q_eval = self.eval_net(b_s, actions).gather(1, b_a)
        q_next = self.target_net(b_s_, actions).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(model, train_split):
    for game in train_split:
        t1 = time.time()
        env = IPHYRE(game=game)
        actions = env.get_action_space()
        # print(actions)
        input_actions = deepcopy(actions)
        best_reward = -100000
        for i in range(EPOCH):
            total_reward = 0
            s = env.reset()
            s[:, :5] /= 600
            s = s.reshape(-1)
            iter = 0
            while iter < MAX_ITER:
                a = model.choose_action(s, input_actions, EPSILON)
                pos = actions[a]
                s_, r, done = env.step(pos, time_step=GAME_TIME / MAX_ITER)
                s_[:, :5] /= 600
                s_ = s_.reshape(-1)
                # print(pos, r)
                model.store_transition(s, a, r, s_)

                total_reward += r
                s = s_

                if model.memory_counter > MEMORY_CAPACITY:
                    model.learn(input_actions)

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

def test(model, test_split):
    rewards = []
    for game in test_split:
        t1 = time.time()
        env = IPHYRE(game=game)
        actions = env.get_action_space()
        # print(actions)
        input_actions = deepcopy(actions)
        best_reward = -100000
        total_reward = 0
        s = env.reset()
        s[:, :5] /= 600
        s = s.reshape(-1)
        iter = 0
        while iter < MAX_ITER:
            a = model.choose_action(s, input_actions, 1)
            pos = actions[a]
            s_, r, done = env.step(pos, time_step=GAME_TIME / MAX_ITER)
            s_[:, :5] /= 600
            s_ = s_.reshape(-1)

            total_reward += r
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


if __name__ == '__main__':
    setup_seed(SEED)
    all_test_rewards = []
    FOLD_LIST = ['basic', 'compositional', 'noisy', 'multi_ball']
    for FOLD in FOLD_LIST:
        FOLD_ID = FOLD_LIST.index(FOLD)
        GAMES = list(game_paras.keys())
        NUM_GAMES = len(GAMES)
        NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
        TEST_SPLIT = GAMES[FOLD_ID * NUM_PER_GROUP: (FOLD_ID + 1) * NUM_PER_GROUP]
        TRAIN_SPLIT = GAMES[:FOLD_ID * NUM_PER_GROUP] + GAMES[(FOLD_ID + 1) * NUM_PER_GROUP:]

        logging.info(f'Fold: {FOLD}, Seed: {SEED}')
        if MODE == 'within':
            dqn = DQN()
            train(dqn, TEST_SPLIT)
            test_rewards = test(dqn, TEST_SPLIT)
            all_test_rewards += test_rewards
        elif MODE == 'cross':
            dqn = DQN()
            train(dqn, TRAIN_SPLIT)
            test_rewards = test(dqn, TEST_SPLIT)
            all_test_rewards += test_rewards
        elif MODE == 'single':
            for game in TEST_SPLIT:
                dqn = DQN()
                train(dqn, [game])
                test_rewards = test(dqn, [game])
                all_test_rewards += test_rewards
        else:
            raise ValueError(f'No such mode {MODE}')
        

    write_csv(path=f'logs/plan_in_situ/DQN_{MODE}_{EPOCH}_rewards.csv', contents=[list(game_paras.keys()), all_test_rewards])
        