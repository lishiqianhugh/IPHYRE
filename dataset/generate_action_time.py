import numpy as np
from random import sample
import os
import logging
import sys
sys.path.append('../games')

from games.simulator import IPHYRE
from games.game_paras import game_paras
from utils import setup_seed


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'generate.log', level=20, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def generate_action_time(game, num_succeed, num_fail, interval, max_game_time, max_action_time, max_step):
    def time_order(a):
        return a[-1]

    def return_center(p):
        x = (p[0][0] + p[1][0]) / 2
        y = (p[0][1] + p[1][1]) / 2
        return [x, y]

    setup_seed(0)
    succeed_list, fail_list = [], []
    sampled_action = []
    assert max_action_time <= max_game_time
    time_steps = [t for t in np.arange(0, max_action_time, interval)]
    ns, nf, iteration = 0, 0, 0
    demo = IPHYRE(game, 'simulate')
    eli_idx = np.where(np.array(game_paras[game]['eli']) == 1)
    eli_blocks = np.array(game_paras[game]['block'])[eli_idx]
    n_step = len(eli_idx[0])
    while ns < num_succeed or nf < num_fail:
        iteration += 1
        demo.reset()  # self.add_all() is contained in reset() function
        act_time = sample(time_steps, n_step)
        while act_time in sampled_action:
            act_time = sample(time_steps, n_step)
        sampled_action.append(act_time)
        # the action should be sorted according to the time order
        action = [return_center(eli_blocks[i]) + [t] for i, t in enumerate(act_time) if t != 0]
        action.sort(key=time_order)
        succeed, step, time_count = demo.run(action)
        # time_count = min(time_count, max_game_time)
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
        info = f'\r{game}: Found {ns} / {num_succeed} succeed and {nf} / {num_fail} fail after {iteration} iterations.'
        print(info, end='')
    logging.info(info)
    return succeed_list, fail_list


if __name__ == '__main__':
    for i, game in enumerate(game_paras.keys()):
        save_path = f'action_data/{game}/'
        if not os.path.exists(save_path):
            print(f'{i+1} / {len(game_paras)}')
            num_succeed, num_fail = 50, 50
            slist, flist = generate_action_time(game,
                                                num_succeed=num_succeed,
                                                num_fail=num_fail,
                                                interval=0.1,
                                                max_game_time=15.,
                                                max_action_time=10.,
                                                max_step=6)
            print(f'\nsucceed_list: {slist}')
            print(f'fail_list: {flist}\n')

            os.makedirs(save_path)
            np.save(save_path + f'succeed_actions_{num_succeed}.npy', slist)
            np.save(save_path + f'fail_actions_{num_fail}.npy', flist)

