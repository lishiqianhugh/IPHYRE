import numpy as np

from run_game import IPHYRE
from utils import setup_seed


def generate_action_time(game, num_succeed, num_fail, interval, max_time):
    setup_seed(0)
    succeed_list, fail_list = [], []
    ns, nf = 0, 0
    while ns < num_succeed or nf < num_fail:
        demo = IPHYRE(game, 'simulate')
        n = 6
        action = np.random.random(n)
        #TODO: save action list and steps and time (use dictionary)
        if demo.run(action):
            if ns < num_succeed:
                succeed_list.append(action)
                ns += 1
        else:
            if nf < num_fail:
                fail_list.append(action)
                nf += 1

    return succeed_list, fail_list


if __name__ == '__main__':
    succeed_list, fail_list = generate_action_time('support', num_succeed=0, num_fail=10, interval=0.1, max_time=10.)
    print(succeed_list, fail_list)
