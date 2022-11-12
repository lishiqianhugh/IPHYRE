import numpy as np
from torch.utils.data import Dataset, DataLoader
from game_paras import game_paras
from utils import setup_seed


class IPHYREData(Dataset):
    def __init__(self, data_path, fold, train=True):
        self.data_path = data_path
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
        self.actions = []
        self.game_data = []
        for game in self.split:
            succeed_actions = np.load(data_path + game + '/succeed_actions_50.npy')
            fail_actions = np.load(data_path + game + '/fail_actions_50.npy')
            actions = np.concatenate((succeed_actions, fail_actions))
            self.actions.append(actions)
            self.game_data.append(actions)  # TODO: use game data (image, body_properties)
        self.actions = np.concatenate(self.actions)
        self.game_data = np.concatenate(self.game_data)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.game_data[idx], self.actions[:, :-3][idx], self.actions[:, -3:][idx]


if __name__ == '__main__':
    setup_seed(0)
    # train_set = IPHYRE_Data(data_path='data/', fold='compositional', train=True)
    test_set = IPHYREData(data_path='data/', fold='compositional', train=False)
    kwargs = {'pin_memory': True, 'num_workers': 0}
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, **kwargs)
    for game_data, actions, label in test_loader:
        print(game_data.shape, actions.shape, label.shape)
        break
