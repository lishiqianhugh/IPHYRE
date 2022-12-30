from iphyre.games import MAX_ELI_OBJ_NUM
from iphyre.utils import generate_actions
from iphyre.simulator import IPHYRE

succeed_list, fail_list, _ = generate_actions(game='hole',
                                              num_succeed=1,
                                              num_fail=1,
                                              interval=0.1,
                                              max_game_time=15.,
                                              max_action_time=7.,
                                              max_step=MAX_ELI_OBJ_NUM,
                                              seed=0)

env = IPHYRE(game='hole')
env.collect_initial_data(save_path='./game_initial_data/')  # save the initial data and image

positions = env.get_action_space()
act_lists = []
for a in succeed_list + fail_list:
    act_lists.append([positions[i + 1] + [a[i]] for i in range(MAX_ELI_OBJ_NUM)])
env.collect_seq_data(save_path='./game_seq_data/', act_lists=act_lists)  # save the sequential data and images
