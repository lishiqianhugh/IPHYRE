import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

from iphyre.games import PARAS
from utils import setup_seed


class IPHYREData(Dataset):
    def __init__(self, action_data_path, game_data_path, num_succeed, num_fail, fold, train=True):
        self.action_data_path = action_data_path
        self.game_data_path = game_data_path
        self.num_succeed = num_succeed
        self.num_fail = num_fail
        self.action_num = self.num_succeed + self.num_fail
        self.fold = fold
        self.fold_list = ['basic', 'compositional', 'noisy', 'multi_ball']
        assert self.fold in self.fold_list
        self.fold_id = self.fold_list.index(self.fold)
        self.num_games = len(PARAS)
        self.num_per_group = int(self.num_games / len(self.fold_list))
        self.test_split = list(PARAS.keys())[self.fold_id * self.num_per_group:
                                                  (self.fold_id + 1) * self.num_per_group]
        self.train_split = list(PARAS.keys())[:self.fold_id * self.num_per_group] \
                           + list(PARAS.keys())[(self.fold_id + 1) * self.num_per_group:]
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
            succeed_actions = np.load(self.action_data_path + game + f'/succeed_actions_{self.num_succeed}.npy')
            fail_actions = np.load(self.action_data_path + game + f'/fail_actions_{self.num_fail}.npy')
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


class IPHYRESeqData(Dataset):
    def __init__(self, data_path, num_succeed, num_fail, fold, max_len=150, train=True):
        self.data_path = data_path
        self.num_succeed = num_succeed
        self.num_fail = num_fail
        self.action_num = self.num_succeed + self.num_fail
        self.max_len = max_len
        self.fold = fold
        self.fold_list = ['basic', 'compositional', 'noisy', 'multi_ball']
        assert self.fold in self.fold_list
        self.fold_id = self.fold_list.index(self.fold)
        self.num_games = len(PARAS)
        self.num_per_group = int(self.num_games / len(self.fold_list))
        self.test_split = list(PARAS.keys())[self.fold_id * self.num_per_group:
                                                  (self.fold_id + 1) * self.num_per_group]
        self.train_split = list(PARAS.keys())[:self.fold_id * self.num_per_group] \
                           + list(PARAS.keys())[(self.fold_id + 1) * self.num_per_group:]
        if train:
            self.split = self.train_split
        else:
            self.split = self.test_split
        self.game_names = []
        self.initial_scenes = []
        self.body_property = []
        self.actions = []
        self.returns_to_go = []
        self.time_steps = []
        self.body_num = []

        for game in self.split[1:2]:
            self.game_names += [game] * self.action_num
            for i in range(self.action_num):
                path = f'{data_path}{game}/{i}'
                property_path = path + '/' + 'vectors.npy'
                actions_path = path + '/' + 'actions.npy'
                body_property = np.load(property_path)
                actions = np.load(actions_path)

                assert self.max_len == body_property.shape[0]
                # convert actions (x, y, t) to [...,[x, y],....]
                actions_seq = np.zeros((self.max_len, 2))
                for a in actions:
                    actions_seq[int(a[-1] * 10)] = a[0:2]
                
                # normalize
                body_property[:, :, :5] /= 600
                actions_seq /= 600

                body_num = 0
                for body in body_property[0]:
                    if body.sum() != 0:
                        body_num += 1
                
                seq_len = 0
                for seq in body_property:
                    if seq.sum() != 0:
                        seq_len += 1

                self.initial_scenes.append([[0] for _ in range(self.max_len)])
                self.body_property.append(body_property)
                self.actions.append(actions_seq)
                if i < self.num_succeed:
                    self.returns_to_go.append([[1] for _ in range(seq_len)] + [[0] for _ in range(self.max_len - seq_len)])
                else:
                    self.returns_to_go.append([[1] for _ in range(self.max_len)])
                self.time_steps.append([[t / self.max_len] for t in range(self.max_len)])  # norm time steps
                self.body_num.append(body_num)

        self.initial_scenes = np.array(self.initial_scenes, dtype=np.float32)
        self.body_property = np.array(self.body_property, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        self.returns_to_go = np.array(self.returns_to_go, dtype=np.float32)
        self.time_steps = np.array(self.time_steps, dtype=np.float32)
        self.body_num = np.array(self.body_num, dtype=np.float32)
                

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return self.game_names[idx],\
               self.initial_scenes[idx], \
               self.body_property[idx], \
               self.actions[idx], \
               self.returns_to_go[idx], \
               self.time_steps[idx], \
               self.body_num[idx]


if __name__ == '__main__':
    # setup_seed(0)
    # train_set = IPHYREData(action_data_path='action_data/',
    #                        game_data_path='game_initial_data/',
    #                        num_succeed=50,
    #                        num_fail=50,
    #                        fold='compositional',
    #                        train=True)
    # kwargs = {'pin_memory': True, 'num_workers': 0}
    # train_loader = DataLoader(train_set, batch_size=16, shuffle=True, **kwargs)
    # for initial_scenes, body_property, actions, label in train_loader:
    #     print(initial_scenes.shape, body_property.shape, actions.shape, label.shape)
    #     break

    setup_seed(0)
    train_set = IPHYRESeqData(data_path='data/offline_data/', num_succeed=50, num_fail=50, fold='compositional')
    kwargs = {'pin_memory': True, 'num_workers': 0}
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, **kwargs)
    for game_names, initial_scenes, body_property, actions, returns_to_go, time_steps, body_num in train_loader:
        print(len(game_names), initial_scenes.shape, body_property.shape, actions.shape, returns_to_go.shape, time_steps.shape, len(body_num))
        # for a in actions[0]:
        #     print(a)
        # print(game_names)
        # print(body_num)
        print(returns_to_go.sum(-2))
        break
