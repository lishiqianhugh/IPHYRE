import numpy as np
from random import sample

from run_game import IPHYRE
from game_paras import game_paras
from utils import setup_seed
import pdb
def generate_action_time(game, num_succeed, num_fail, interval, max_time, max_step):
    def time_order(a):
        return a[-1]

    setup_seed(0)
    succeed_list, fail_list = [], []
    sampled_action = []
    time_steps = [t for t in np.arange(0, max_time, interval)]
    ns, nf = 0, 0
    demo = IPHYRE(game, 'simulate')
    while ns < num_succeed or nf < num_fail:
        demo.reset()  # self.add_all() is contained in reset() function
        n_step = len(game_paras[game]['block'])
        act_time = sample(time_steps, n_step)
        while act_time in sampled_action:
            act_time = sample(time_steps, n_step)
        sampled_action.append(act_time)
        # the action should be sorted according to the time order
        action = [return_center(game_paras[game]['block'][i]) + [t] for (i, t) in enumerate(act_time)]
        action.sort(key=time_order)
        succeed, step, time_count = demo.run(action)
        time_count = min(time_count, max_time)
        real_act_time = [t for t in act_time if t <= time_count] + [0]*(max_step - len(act_time))
        if succeed:
            if ns < num_succeed:
                succeed_list.append([real_act_time, succeed, step, time_count])
                ns += 1
        else:
            if nf < num_fail:
                fail_list.append([real_act_time, succeed, step, time_count])
                nf += 1
    return succeed_list, fail_list

def return_center(p):
    x = (p[0][0]+p[1][0])/2
    y = (p[0][1] + p[1][1]) / 2
    return [x,y]
if __name__ == '__main__':
    succeed_list, fail_list = generate_action_time('hinder', num_succeed=5, num_fail=5, interval=0.1, max_time=15., max_step=6)
    print(f'succeed_list: {succeed_list}')
    print(f'fail_list: {fail_list}')
