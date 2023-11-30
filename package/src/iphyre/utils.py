import numpy as np
import random
from random import sample

from iphyre.simulator import IPHYRE
from iphyre.games import GAMES, PARAS


def generate_actions(game, num_succeed, num_fail, interval, max_game_time, max_action_time, max_step, seed=0, epsilon=0.2):
    np.random.seed(seed)
    random.seed(seed)

    succeed_list, fail_list = [], []
    sampled_action = []
    assert max_action_time <= max_game_time
    time_steps = [t for t in np.arange(0, max_action_time, interval)]
    ns, nf, iteration = 0, 0, 0
    demo = IPHYRE(game)
    positions = demo.get_action_space()
    eli_idx = np.where(np.array(PARAS[game]['eli']) == 1)
    n_step = len(eli_idx[0])
    while ns < num_succeed or nf < num_fail:
        iteration += 1
        demo.reset()
        act_time = sample(time_steps, n_step)
        while act_time in sampled_action:
            act_time = sample(time_steps, n_step)
        act_time = [t * (np.random.uniform() > epsilon) for t in act_time]
        sampled_action.append(act_time)
        act_time += [0.] * (max_step - len(act_time))
        act_list = [positions[i + 1] + [act_time[i]] for i in range(max_step)]
        succeed, valid_step, time_count = demo.simulate(act_list)
        real_act_time = [t * (t <= time_count) for t in act_time]  # mask the time-out actions
        if succeed:
            if ns < num_succeed:
                succeed_list.append(real_act_time + [succeed, valid_step, time_count])
                ns += 1
        else:
            if nf < num_fail:
                fail_list.append(real_act_time + [succeed, valid_step, time_count])
                nf += 1
        log = f'\r{game}: Found {ns} / {num_succeed} succeed and {nf} / {num_fail} fail after {iteration} iterations.'
        print(log, end='')
    print()
    
    return succeed_list, fail_list, iteration


def play_all():
    for game in GAMES:
        demo = IPHYRE(game)
        demo.play()


def collect_initial_all(save_path='./dataset/game_initial_data/'):
    for game in GAMES:
        demo = IPHYRE(game)
        demo.collect_initial_data(save_path=save_path)


def collect_play_all(player_name, max_episode, save_path):
    for game in GAMES:
        demo = IPHYRE(game)
        demo.collect_while_play(player_name=player_name, max_episode=max_episode, save_path=save_path)
