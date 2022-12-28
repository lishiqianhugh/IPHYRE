import numpy as np
import random
from random import sample

from iphyre.simulator import IPHYRE
from iphyre.games import GAMES, PARAS


def generate_actions(game, num_succeed, num_fail, interval, max_game_time, max_action_time, max_step, seed=0):
    def time_order(a):
        return a[-1]

    def return_center(p):
        x = (p[0][0] + p[1][0]) / 2
        y = (p[0][1] + p[1][1]) / 2
        return [x, y]

    np.random.seed(seed)
    random.seed(seed)

    succeed_list, fail_list = [], []
    sampled_action = []
    assert max_action_time <= max_game_time
    time_steps = [t for t in np.arange(0, max_action_time, interval)]
    ns, nf, iteration = 0, 0, 0
    demo = IPHYRE(game)
    eli_idx = np.where(np.array(PARAS[game]['eli']) == 1)
    eli_blocks = np.array(PARAS[game]['block'])[eli_idx]
    n_step = len(eli_idx[0])
    while ns < num_succeed or nf < num_fail:
        iteration += 1
        demo.reset()
        act_time = sample(time_steps, n_step)
        while act_time in sampled_action:
            act_time = sample(time_steps, n_step)
        sampled_action.append(act_time)
        # the action should be sorted according to the time order
        action = [return_center(eli_blocks[i]) + [t] for i, t in enumerate(act_time) if t != 0]
        action.sort(key=time_order)
        succeed, step, time_count = demo.simulate(action)
        real_act_time = [t * (t <= time_count) for t in act_time]
        real_act_time += [0] * (max_step - len(real_act_time))
        if succeed:
            if ns < num_succeed:
                succeed_list.append(real_act_time + [succeed] + [step] + [time_count])
                ns += 1
        else:
            if nf < num_fail:
                fail_list.append(real_act_time + [succeed] + [step] + [time_count])
                nf += 1
        log = f'\r{game}: Found {ns} / {num_succeed} succeed and {nf} / {num_fail} fail after {iteration} iterations.'
        print(log, end='')
    
    return succeed_list, fail_list, iteration

def play_all():
    for game in GAMES:
        demo = IPHYRE(game)
        demo.play()

def collect_initial_all(save_path='./dataset/game_initial_data/'):
    for game in GAMES:
        demo = IPHYRE(game)
        demo.collect_initial_data(save_path=save_path)

def collect_seq_all(action_path='dataset/action_data_7s/'):
    def return_center(p):
        x = (p[0][0] + p[1][0]) / 2
        y = (p[0][1] + p[1][1]) / 2
        return [x, y]
    def time_order(a):
        return a[-1]
    for game in GAMES:
        succeed_actions = np.load(action_path + game + '/succeed_actions.npy')
        fail_actions = np.load(action_path + game + '/fail_actions.npy')
        actions = np.concatenate((succeed_actions[:, :-3], fail_actions[:, :-3]))
        eli_idx = np.where(np.array(PARAS[game]['eli']) == 1)
        eli_blocks = np.array(PARAS[game]['block'])[eli_idx]
        act_lists = []
        for act_time in actions:
            act = [return_center(eli_blocks[i]) + [t] for i, t in enumerate(act_time) if t != 0]
            act.sort(key=time_order)
            act_lists.append(act)
        demo = IPHYRE(game)
        demo.collect_seq_data(act_lists=act_lists)

def collect_play_all(player_name, max_episode, save_path):
    for game in GAMES:
        demo = IPHYRE(game)
        demo.collect_while_play(player_name=player_name, max_episode=max_episode, save_path=save_path)
