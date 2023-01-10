import os
import tempfile
import shutil
import csv
import seaborn as sns
import pandas as pd
import json

import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import cv2
from iphyre.games import PARAS


from functools import partial
from torch.optim import Optimizer

class EarlyStopping(object): # pylint: disable=R0902
    """
    Gives a criterion to stop training when a given metric is not
    improving anymore
    Args:
        mode (str): One of `min`, `max`. In `min` mode, training will
            be stopped when the quantity monitored has stopped
            decreasing; in `max` mode it will be stopped when the
            quantity monitored has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after
            which training is stopped. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only stop learning after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.

    """

    def __init__(self, mode='min', patience=10, threshold=1e-4, threshold_mode='rel'):
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        """ Updates early stopping state """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    @property
    def stop(self):
        """ Should we stop learning? """
        return self.num_bad_epochs > self.patience


    def _cmp(self, mode, threshold_mode, threshold, a, best): # pylint: disable=R0913, R0201
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        """ Returns early stopping state """
        return {key: value for key, value in self.__dict__.items() if key != 'is_better'}

    def load_state_dict(self, state_dict):
        """ Loads early stopping state """
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold,
                             threshold_mode=self.threshold_mode)



############################################################
#### WARNING : THIS IS A TEMPORARY CODE WHICH HAS      #####
####  TO BE REMOVED WITH PYTORCH 0.5                   #####
#### IT IS COPY OF THE 0.5 VERSION OF THE LR SCHEDULER #####
############################################################
class ReduceLROnPlateau(object): # pylint: disable=R0902
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, # pylint: disable=R0913
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        """ Updates scheduler state """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        """ Are we on CD? """
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best): # pylint: disable=R0913,R0201
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        """ Returns scheduler state """
        return {key: value for key, value in self.__dict__.items()
                if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        """ Loads scheduler state """
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold,
                             threshold_mode=self.threshold_mode)

def plot_trajectory(traj_x, traj_y):
    plt.scatter(traj_x, traj_y)
    plt.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def draw_bbox(im_path, vectors_path):
    """
    There is a -1 bug in x1, y1, x2, y2, r
    """
    image = cv2.imread(im_path)
    vectors = np.load(vectors_path)
    for vector in vectors:
        x1, y1, x2, y2, r = vector[:5]
        x1, y1, x2, y2, r = int(y1) - 1, int(x1) - 1, int(y2) - 1, int(x2) - 1, int(r) - 1
        x1 = min(x1, x2)
        x2 = max(x1, x2)
        y1 = min(y1, y2)
        y2 = max(y1, y2)
        print(x1, y1, x2, y2, r)
        image[x1 - r: x2 + r, y1 - r] = [0, 0, 0]
        image[x1 - r: x2 + r, y2 + r] = [0, 0, 0]
        image[x1 - r, y1 - r: y2 + r] = [0, 0, 0]
        image[x2 + r, y1 - r: y2 + r] = [0, 0, 0]
    cv2.imshow('bbox_image', image)
    cv2.waitKey(0)


def reorganize_images(generated_dir='dataset/game_initial_data/',
                      save_dir='games/images/'):
    fold_list = ['basic', 'compositional', 'noisy', 'multi_ball']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, fold in enumerate(fold_list):
        for game in list(PARAS.keys())[i * 10: (i + 1) * 10]:
            save_path = save_dir + f'{fold}_{game}.jpg'
            im_path = generated_dir + f'{game}/' + f'{game}.jpg'
            image = cv2.imread(im_path)
            cv2.imwrite(save_path, image)


def print_generated_actions(dataset_path='dataset/action_data/'):
    print('##################### succeed #####################')
    for game in PARAS.keys():
        succeed_path = dataset_path + f'{game}/succeed_actions_50.npy'
        succeed_data = np.load(succeed_path)
        print(game)
        print(succeed_data)
    print('##################### fail #####################')
    for game in PARAS.keys():
        fail_path = dataset_path + f'{game}/fail_actions_50.npy'
        fail_data = np.load(fail_path)
        print(game)
        print(fail_data)

def jpg2gif(path='dataset\offline_data'):
    for game in os.listdir(path):
        for i in range(100):
            os.system(f'ffmpeg -i {path}/{game}/{i}/images/%d.jpg -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {path}/{game}/{i}/{game}.{i}.gif')


def check(path='dataset\offline_data'):
    for game in os.listdir(path):
        for i in range(100):
            if not os.path.exists(f'{path}/{game}/{i}'):
                print(game, i)

def write_json(path, contents, mode='a'):
    with open(path, mode) as file:
        file.writelines(json.dumps(contents)+'\n')

def read_json(path, mode='r'):
    with open(path, mode) as file:
        contents = []
        for line in file:
            contents.append[json.loads(line)]
    return contents

def write_csv(path, contents, mode='a'):
    with open(path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(contents)

def read_csv(path, mode='r'):
    with open(path, mode) as csvfile:
        reader = csv.reader(csvfile)
        contents = []
        for line in reader:
            contents.append(line)
    return contents

def avg_reward_from_csv(path):
    FOLD_LIST = ['basic', 'compositional', 'noisy', 'multi_ball']
    for i, fold in enumerate(FOLD_LIST):
        rewards = read_csv(path)[1][i * 10: (i + 1) * 10]
        int_rewards = [int(r) for r in rewards]
        avg_rewards = sum(int_rewards) / 10
        print(fold, avg_rewards)

def analyze_games_to_csv(path='./analysis.csv'):
    FOLD_LIST = ['basic', 'compositional', 'noisy', 'multi_ball']
    contents = [['Game', 'Fold', 'Body Type', 'Num']]
    for FOLD in FOLD_LIST:
        FOLD_ID = FOLD_LIST.index(FOLD)
        GAMES = list(PARAS.keys())
        NUM_GAMES = len(GAMES)
        NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
        SPLIT = GAMES[FOLD_ID * NUM_PER_GROUP: (FOLD_ID + 1) * NUM_PER_GROUP]
        for game in SPLIT:
            num_blocks = len(PARAS[game]['block'])
            num_eli_blocks = sum(PARAS[game]['eli'])
            num_balls = len(PARAS[game]['ball'])
            contents.append([game, FOLD.capitalize(), 'Blocks', num_blocks])
            contents.append([game, FOLD.capitalize(), 'Gray Blocks', num_eli_blocks])
            contents.append([game, FOLD.capitalize(), 'Balls', num_balls])
    write_csv(path, contents)

def draw_barplot(csv_path='./analysis.csv', save_path='./analysis.pdf'):
    contents = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.5)
    g = sns.catplot(
        data=contents, kind="bar",
        x="Fold", y="Num", hue="Body Type",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Number")
    g.legend.set_title("Body Type")
    g.figure.set_size_inches(12, 6)
    sns.move_legend(g, "upper center", ncol=3, title=None)
    g.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    # draw_bbox('../dataset/game_initial_data/spring_flick/spring_flick.jpg',
    #           '../dataset/game_initial_data/spring_flick/vectors.npy')
    # reorganize_images()
    # print_generated_actions()
    jpg2gif(path='data/seq_data')
    # check()
    # write_csv(path='./test.csv', contents=[[1,2], [3,4]])
    # contents = read_csv(path='./test.csv')
    # print(contents)
    # avg_reward_from_csv(path='logs\plan_in_situ\Random_rewards.csv')
    # avg_reward_from_csv(path='logs\plan_in_situ\DQN_single_False_1000_0.9_0.01_rewards.csv')
    # analyze_games_to_csv()
    # draw_barplot()

