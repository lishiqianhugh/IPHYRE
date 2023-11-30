'''
Generate the offline dataset
'''

import numpy as np
from torch.utils.data import Dataset
import cv2
import tqdm
from iphyre.games import PARAS
from iphyre.simulator import IPHYRE


ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 2, 12 * 9, 256, 64, 64

class IPHYREData(Dataset):
    def __init__(self, action_data_path, game_data_path, num_succeed, num_fail, fold, train=True):
        self.action_data_path = action_data_path
        self.game_data_path = game_data_path
        self.num_succeed = num_succeed
        self.num_fail = num_fail
        self.action_num = self.num_succeed + self.num_fail
        self.fold = fold
        self.fold_list = ['basic', 'noisy', 'compositional', 'multi_ball']
        assert self.fold in self.fold_list
        self.fold_id = self.fold_list.index(self.fold)
        self.num_games = len(PARAS)
        self.num_per_group = int(self.num_games / len(self.fold_list))
        self.test_split = list(PARAS.keys())[self.fold_id * self.num_per_group:
                                                  (self.fold_id + 1) * self.num_per_group]
        self.train_split = list(PARAS.keys())[:self.num_per_group]
        if train:
            self.split = self.train_split
        else:
            self.split = self.test_split
        self.game_names = []
        self.initial_scenes = []
        self.body_property = []
        self.actions = []
        self.labels = []

        for game in self.split:
            self.game_names += [game] * self.action_num
            initial_scene = cv2.imread(self.game_data_path + game + f'/{game}.jpg')
            initial_scene = cv2.resize(initial_scene, dsize=(224, 224)).transpose((2, 0, 1))
            self.initial_scenes.append([initial_scene] * self.action_num)
            body_property = np.load(self.game_data_path + game + '/vectors.npy')
            body_property[:, :5] /= 600  # normalize
            self.body_property.append([body_property] * self.action_num)
            succeed_actions = np.load(self.action_data_path + game + f'/succeed_actions.npy')
            fail_actions = np.load(self.action_data_path + game + f'/fail_actions.npy')
            actions = np.concatenate((succeed_actions[:, :-3], fail_actions[:, :-3]))
            self.labels.append(np.concatenate((succeed_actions[:, -3:], fail_actions[:, -3:])))
            # complete action
            complete_actions = np.zeros((self.action_num, body_property.shape[0]))
            for i, acts in enumerate(actions):
                k = 0
                for j, bps in enumerate(body_property):
                    if bps[-4] == 1:
                        complete_actions[i][j] = acts[k]
                        k += 1
            self.actions.append(complete_actions)

        self.initial_scenes = np.concatenate(self.initial_scenes, dtype=np.float32)
        self.body_property = np.concatenate(self.body_property, dtype=np.float32)
        self.actions = np.concatenate(self.actions, dtype=np.float32)
        self.labels = np.concatenate(self.labels, dtype=np.float32)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return self.game_names[idx],\
               self.initial_scenes[idx], \
               self.body_property[idx], \
               self.actions[idx], \
               self.labels[idx]


class IPHYREData_inadvance(Dataset):
    def __init__(self, data_path, transform, num_succeed, num_fail, fold, max_len=1, train=True):
        self._transform = transform
        self.data_path = data_path
        self.num_succeed = num_succeed
        self.num_fail = num_fail
        self.action_num = self.num_succeed + self.num_fail
        self.max_len = max_len
        self.fold = fold
        self.fold_list = ['basic', 'noisy', 'compositional', 'multi_ball']
        assert self.fold in self.fold_list
        self.fold_id = self.fold_list.index(self.fold)
        self.num_games = len(PARAS)
        self.num_per_group = int(self.num_games / len(self.fold_list))
        self.test_split = list(PARAS.keys())[self.fold_id * self.num_per_group:
                                             (self.fold_id + 1) * self.num_per_group]
        self.train_split = list(PARAS.keys())[:self.num_per_group]
        if train:
            self.split = self.train_split
        else:
            self.split = self.test_split
        self.game_names = []
        self.sequence_scenes = []
        self.body_property = []
        self.actions = []
        self.returns_to_go = []
        self.time_steps = []
        self.body_num = []
        self.done = []

        self.target_return = 1000

        for game in self.split:
            env = IPHYRE(game=game, fps=10)
            self.game_names += [game] * self.action_num
            for i in range(self.action_num):
                path = f'{data_path}/{game}/{i}'
                property_path = path + '/' + 'vectors.npy'
                actions_path = path + '/' + 'actions.npy'
                body_property = np.load(property_path)[:1]
                actions = np.load(actions_path)

                positions = env.get_action_space()[1:]
                action_time = np.zeros(6)
                for a in actions.tolist():
                    index = positions.index(a[:2])
                    action_time[index] = a[-1]

                # normalize
                body_property[:, :, :5] /= 600
                action_time /= 15
                positions = np.array(positions).reshape(-1) / 600
                body_property = np.concatenate((body_property.reshape(-1), positions))

                self.sequence_scenes.append(0)
                self.body_property.append([body_property])
                self.actions.append([action_time])
                if i < self.num_succeed:
                    self.returns_to_go.append(
                        [[0]])
                    self.done.append([[1]])
                else:
                    self.returns_to_go.append([[self.target_return]])
                    self.done.append([[0]])

                self.time_steps.append([[0 / self.max_len]])  # norm time steps
                self.body_num.append(0)

        self.body_property = np.array(self.body_property, dtype=np.float32)
        self.sequence_scenes = self.body_property
        self.actions = np.array(self.actions, dtype=np.float32)
        self.returns_to_go = np.array(self.returns_to_go, dtype=np.float32)
        self.time_steps = np.array(self.time_steps, dtype=np.float32)
        self.body_num = np.array(self.body_num, dtype=np.float32)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return self.game_names[idx], \
               self.sequence_scenes[idx], \
               self.body_property[idx], \
               self.actions[idx], \
               self.returns_to_go[idx], \
               self.time_steps[idx], \
               self.body_num[idx]


class IPHYREData_onthefly(Dataset):
    def __init__(self, data_path, transform, num_succeed, num_fail, fold, max_len=150, train=True):
        self._transform = transform
        self.data_path = data_path
        self.num_succeed = num_succeed
        self.num_fail = num_fail
        self.action_num = self.num_succeed + self.num_fail
        self.max_len = max_len
        self.fold = fold
        self.fold_list = ['basic', 'noisy', 'compositional', 'multi_ball']
        assert self.fold in self.fold_list
        self.fold_id = self.fold_list.index(self.fold)
        self.num_games = len(PARAS)
        self.num_per_group = int(self.num_games / len(self.fold_list))
        self.test_split = list(PARAS.keys())[self.fold_id * self.num_per_group:
                                             (self.fold_id + 1) * self.num_per_group]
        self.train_split = list(PARAS.keys())[:self.num_per_group]
        if train:
            self.split = self.train_split
        else:
            self.split = self.test_split
        self.game_names = []
        self.sequence_scenes = []
        self.body_property = []
        self.actions = []
        self.returns_to_go = []
        self.time_steps = []
        self.body_num = []
        self.done = []

        self.target_return = 1

        for game in self.split:
            env = IPHYRE(game, fps=10)
            self.game_names += [game] * self.action_num
            for i in range(self.action_num):
                path = f'{data_path}/{game}/{i}'
                property_path = path + '/' + 'vectors.npy'
                actions_path = path + '/' + 'actions.npy'
                body_property = np.load(property_path)
                actions = np.load(actions_path)
                positions = env.get_action_space()

                assert self.max_len == body_property.shape[0]
                # convert actions (x, y, t) to [...,[x, y],....]
                actions_seq = np.zeros((self.max_len, 2))
                for a in actions:
                    actions_seq[int(a[-1] * 10)] = a[0:2]

                # convert actions_seq to one hot
                one_hot_actions_seq = np.zeros((self.max_len, 7))
                for i, a in enumerate(actions_seq):
                    id = positions.index(a.tolist())
                    one_hot_actions_seq[i][id] = 1.

                # normalize
                body_property[:, :, :5] /= 600
                positions = np.array(positions).reshape(-1) / 600
                body_property = np.concatenate(
                    (body_property.reshape(self.max_len, -1), np.tile(positions, (self.max_len, 1))), -1)

                seq_len = 0
                for seq in body_property:
                    if seq.sum() != 0:
                        seq_len += 1

                self.sequence_scenes.append(0)
                self.body_property.append(body_property.reshape(self.max_len, -1))
                self.actions.append(one_hot_actions_seq)
                if i < self.num_succeed:
                    self.returns_to_go.append(
                        [[self.target_return] for i in range(seq_len)] + [[0] for _ in range(self.max_len - seq_len)])
                    self.done.append(
                        [[0] for _ in range(seq_len)] + [[1]] + [[1] for _ in range(self.max_len - seq_len - 1)])
                else:
                    self.returns_to_go.append([[self.target_return] for _ in range(self.max_len)])
                    self.done.append([[0] for _ in range(self.max_len)])

                self.time_steps.append([[t / self.max_len] for t in range(self.max_len)])  # norm time steps
                self.body_num.append(0)

        self.body_property = np.array(self.body_property, dtype=np.float32)
        self.sequence_scenes = self.body_property
        self.actions = np.array(self.actions, dtype=np.float32)
        self.returns_to_go = np.array(self.returns_to_go, dtype=np.float32)
        self.time_steps = np.array(self.time_steps, dtype=np.float32)
        self.body_num = np.array(self.body_num, dtype=np.float32)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return self.game_names[idx], \
               self.sequence_scenes[idx], \
               self.body_property[idx], \
               self.actions[idx], \
               self.returns_to_go[idx], \
               self.time_steps[idx], \
               self.body_num[idx]


class IPHYREData_combine(Dataset):
    def __init__(self, data_path, transform, num_succeed, num_fail, fold, max_len=6, train=True):
        self._transform = transform
        self.data_path = data_path
        self.num_succeed = num_succeed
        self.num_fail = num_fail
        self.action_num = self.num_succeed + self.num_fail
        self.max_len = max_len
        self.fold = fold
        self.fold_list = ['basic', 'noisy', 'compositional', 'multi_ball']
        assert self.fold in self.fold_list
        self.fold_id = self.fold_list.index(self.fold)
        self.num_games = len(PARAS)
        self.num_per_group = int(self.num_games / len(self.fold_list))
        self.test_split = list(PARAS.keys())[self.fold_id * self.num_per_group:
                                             (self.fold_id + 1) * self.num_per_group]
        self.train_split = list(PARAS.keys())[:self.num_per_group]
        if train:
            self.split = self.train_split
        else:
            self.split = self.test_split
        self.game_names = []
        self.sequence_scenes = []
        self.body_property = []
        self.actions = []
        self.returns_to_go = []
        self.time_steps = []
        self.body_num = []
        self.done = []

        self.target_return = 1

        for game in self.split:
            env = IPHYRE(game, fps=10)
            self.game_names += [game] * self.action_num * self.max_len
            for i in range(self.action_num):
                path = f'{data_path}/{game}/{i}'
                property_path = path + '/' + 'vectors.npy'
                actions_path = path + '/' + 'actions.npy'
                body_property = np.load(property_path)
                actions = np.load(actions_path)
                positions = env.get_action_space()[1:]

                # convert actions (x, y, t) to [...,[0, 0, t, 0],....]
                actions_seq = np.zeros((self.max_len, 6))
                valid_id = 0
                for a in actions:
                    if np.sum(a[:2]) == 0:
                        continue
                    else:
                        if a[-1] == 0:
                            a[-1] = 1e-1  # a small offset
                        id = positions.index(a[:2].tolist())
                        actions_seq[valid_id][id] = a[-1] / 2  # norm
                        valid_id += 1

                # normalize
                body_property[:, :, :5] /= 600
                positions = np.array(positions).reshape(-1) / 600
                frame_index = np.max(actions_seq[:-1], -1) * 10  # the frame after execution
                body_property = np.concatenate(
                    (body_property[0:1], body_property[frame_index.astype(int)]))
                body_property = np.concatenate(
                    (body_property.reshape(self.max_len, -1), np.tile(positions, (self.max_len, 1))), -1)
                body_property[valid_id + 1:] *= 0

                body_num = 0
                seq_len = 0
                for seq in body_property:
                    if seq.sum() != 0:
                        seq_len += 1

                self.sequence_scenes.append(0)

                self.body_property.append(body_property.reshape(self.max_len, -1))
                self.actions.append(actions_seq)
                if i < self.num_succeed:
                    self.returns_to_go.append(
                        [[self.target_return] for i in range(seq_len)] + [[0] for _ in range(self.max_len - seq_len)])
                    self.done.append(
                        [[0] for _ in range(seq_len)] + [[1]] + [[1] for _ in range(self.max_len - seq_len - 1)])
                else:
                    self.returns_to_go.append([[self.target_return] for _ in range(self.max_len)])
                    self.done.append([[0] for _ in range(self.max_len)])

                self.time_steps.append([[t / self.max_len] for t in range(self.max_len)])  # norm time steps
                self.body_num.append(body_num)

        self.body_property = np.array(self.body_property, dtype=np.float32)
        self.sequence_scenes = self.body_property
        self.actions = np.array(self.actions, dtype=np.float32)
        self.returns_to_go = np.array(self.returns_to_go, dtype=np.float32)
        self.time_steps = np.array(self.time_steps, dtype=np.float32)
        self.body_num = np.zeros_like(self.time_steps)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return self.game_names[idx], \
               self.sequence_scenes[idx], \
               self.body_property[idx], \
               self.actions[idx], \
               self.returns_to_go[idx], \
               self.time_steps[idx], \
               self.body_num[idx]


class IPHYRERnnData(Dataset):
    def __init__(self, data_path, seq_len, transform, num_succeed, num_fail, fold, max_len=150, train=True,
                 single=True):
        self._transform = transform
        self.data_path = data_path
        self._seq_len = seq_len
        self.num_succeed = num_succeed
        self.num_fail = num_fail
        self.action_num = self.num_succeed + self.num_fail
        self.max_len = max_len
        self.fold = fold
        self.fold_list = ['basic', 'noisy', 'compositional', 'multi_ball']
        if single:
            self.test_split = [fold]
            self.train_split = [fold]
        else:
            self.num_games = len(PARAS)
            self.num_games = len(PARAS)
            self.num_per_group = int(self.num_games / len(self.fold_list))
            basic_id = 0

            self.train_split = list(PARAS.keys())[basic_id * self.num_per_group:
                                                  (basic_id + 1) * self.num_per_group]

            self.test_split = list(PARAS.keys())[:basic_id * self.num_per_group] \
                              + list(PARAS.keys())[(basic_id + 1) * self.num_per_group:]

        if train:
            self.split = self.train_split
        else:
            self.split = self.test_split
        print(f"train:{train}")
        print(len(self.split))
        self.game_names = []
        self.vectors = []
        self.body_property = []
        self.actions = []
        self.rewards = []
        self.time_steps = []
        self.body_num = []
        self.done = []
        for game in self.split:
            self.game_names += [game] * self.action_num
            for i in tqdm(range(self.action_num)):
                path = f'{data_path}/{game}/{i}'
                actions_path = path + '/' + 'actions.npy'
                rewards_path = path + '/' + 'rewards.npy'
                vectors_path = path + '/' + 'vectors.npy'
                rewards = np.load(rewards_path)
                actions = np.load(actions_path)
                vectors = np.load(vectors_path)
                done = np.zeros((self.max_len,))
                # convert actions (x, y, t) to [...,[x, y],....]
                actions_seq = np.zeros((self.max_len, 2))
                for a in actions:
                    actions_seq[int(a[-1] * 10)] = a[0:2]

                actions_seq /= 600
                vectors /= 600
                seq_len = sum(rewards < 0)
                done[seq_len:] = 1

                if seq_len < self.max_len - 1:
                    rewards[seq_len] = 499.
                if rewards.shape[0] > 150 and rewards[-1] == 499:
                    rewards = rewards[:150]
                    rewards[-1] = 499

                self.actions.append(actions_seq)
                self.rewards.append(rewards)
                self.done.append(done)
                self.vectors.append(vectors)
        self.vectors = np.array(self.vectors, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        self.rewards = np.array(self.rewards, dtype=np.float32)
        self.done = np.array(self.done, dtype=np.float32)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        rollout_len = self.actions.shape[1]
        seq_idx = np.random.randint(low=0, high=rollout_len - self._seq_len)
        obs_data = self.vectors[idx][seq_idx:seq_idx + self._seq_len + 1]
        obs, next_obs = obs_data[:-1], obs_data[1:]
        return obs, \
               self.actions[idx][seq_idx + 1:seq_idx + self._seq_len + 1], \
               self.rewards[idx][seq_idx + 1:seq_idx + self._seq_len + 1], \
               self.done[idx][seq_idx + 1:seq_idx + self._seq_len + 1], \
               next_obs