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
from games.game_paras import game_paras


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
        for game in list(game_paras.keys())[i * 10: (i + 1) * 10]:
            save_path = save_dir + f'{fold}_{game}.jpg'
            im_path = generated_dir + f'{game}/' + f'{game}.jpg'
            image = cv2.imread(im_path)
            cv2.imwrite(save_path, image)


def print_generated_actions(dataset_path='dataset/action_data/'):
    print('##################### succeed #####################')
    for game in game_paras.keys():
        succeed_path = dataset_path + f'{game}/succeed_actions_50.npy'
        succeed_data = np.load(succeed_path)
        print(game)
        print(succeed_data)
    print('##################### fail #####################')
    for game in game_paras.keys():
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
        GAMES = list(game_paras.keys())
        NUM_GAMES = len(GAMES)
        NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
        SPLIT = GAMES[FOLD_ID * NUM_PER_GROUP: (FOLD_ID + 1) * NUM_PER_GROUP]
        for game in SPLIT:
            num_blocks = len(game_paras[game]['block'])
            num_eli_blocks = sum(game_paras[game]['eli'])
            num_balls = len(game_paras[game]['ball'])
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
    # jpg2gif()
    # check()
    # write_csv(path='./test.csv', contents=[[1,2], [3,4]])
    # contents = read_csv(path='./test.csv')
    # print(contents)
    # avg_reward_from_csv(path='logs\plan_in_situ\Random_rewards.csv')
    # avg_reward_from_csv(path='logs\plan_in_situ\DQN_single_False_1000_0.9_0.01_rewards.csv')
    # analyze_games_to_csv()
    draw_barplot()

