from iphyre.games import MAX_ELI_OBJ_NUM
from iphyre.utils import generate_actions
from iphyre.simulator import IPHYRE
from iphyre.games import GAMES
import sys
import os
sys.path.append("…")

if not os.path.exists('dataset/game_seq_data/'):
    os.makedirs('dataset/game_seq_data/')
if not os.path.exists('dataset/game_initial_data/'):
    os.makedirs('dataset/game_initial_data/')
for game in GAMES: 
    succeed_list, fail_list, _ = generate_actions(game='hole',
                                                num_succeed=50,
                                                num_fail=50,
                                                interval=0.1,
                                                max_game_time=15.,
                                                max_action_time=7.,
                                                max_step=MAX_ELI_OBJ_NUM,
                                                seed=0)

    env = IPHYRE(game='hole')
    env.collect_initial_data(save_path='dataset/game_initial_data/')  # save the initial data and image

    positions = env.get_action_space()
    act_lists = []
    for a in succeed_list + fail_list:
        act_lists.append([positions[i + 1] + [a[i]] for i in range(MAX_ELI_OBJ_NUM)])
    env.collect_seq_data(save_path='dataset/game_seq_data/', act_lists=act_lists)  # save the sequential data and images
