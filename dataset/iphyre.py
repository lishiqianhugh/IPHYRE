import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

from games.game_paras import game_paras
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
        self.num_games = len(game_paras)
        self.num_per_group = int(self.num_games / len(self.fold_list))
        self.test_split = list(game_paras.keys())[self.fold_id * self.num_per_group:
                                                  (self.fold_id + 1) * self.num_per_group]
        self.train_split = list(game_paras.keys())[:self.fold_id * self.num_per_group] \
                           + list(game_paras.keys())[(self.fold_id + 1) * self.num_per_group:]
        if train:
            self.split = self.train_split
        else:
            self.split = self.test_split
        self.game_names = []
        self.initial_scenes = []
        self.body_property = []
        self.actions = []

        for game in self.split:
            self.game_names += [game] * self.action_num
            initial_scene = cv2.imread(self.game_data_path + game + f'/{game}.jpg')
            self.initial_scenes.append([initial_scene] * self.action_num)
            body_property = np.load(self.game_data_path + game + '/vectors.npy')
            self.body_property.append([body_property] * self.action_num)
            succeed_actions = np.load(self.action_data_path + game + f'/succeed_actions_{self.num_succeed}.npy')
            fail_actions = np.load(self.action_data_path + game + f'/fail_actions_{self.num_fail}.npy')
            actions = np.concatenate((succeed_actions, fail_actions))
            self.actions.append(actions)

        self.initial_scenes = np.concatenate(self.initial_scenes, dtype=np.float32)
        self.body_property = np.concatenate(self.body_property, dtype=np.float32)
        self.actions = np.concatenate(self.actions, dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.game_names[idx],\
               self.initial_scenes[idx], \
               self.body_property[idx], \
               self.actions[:, :-3][idx], \
               self.actions[:, -3:][idx]  # label


if __name__ == '__main__':
    setup_seed(0)
    train_set = IPHYREData(action_data_path='action_data/',
                           game_data_path='game_initial_data/',
                           num_succeed=50,
                           num_fail=50,
                           fold='compositional',
                           train=True)
    kwargs = {'pin_memory': True, 'num_workers': 0}
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, **kwargs)
    for initial_scenes, body_property, actions, label in train_loader:
        print(initial_scenes.shape, body_property.shape, actions.shape, label.shape)
        break
