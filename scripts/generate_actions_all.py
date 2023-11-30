import numpy as np
import os
import logging
 
from iphyre.utils import generate_actions
from iphyre.games import GAMES, MAX_ELI_OBJ_NUM


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'generate.log', level=20, format=LOG_FORMAT, datefmt=DATE_FORMAT)

for i, game in enumerate(GAMES):
    num_succeed, num_fail = 50, 50
    save_path = f'data/action_data_s{num_succeed}_f{num_fail}/{game}/'
    if not os.path.exists(save_path):
        print(f'{i+1} / {len(GAMES)}')
        slist, flist, log = generate_actions(game,
                                            num_succeed=num_succeed,
                                            num_fail=num_fail,
                                            interval=0.1,
                                            max_game_time=15.,
                                            max_action_time=7.,
                                            max_step=MAX_ELI_OBJ_NUM,
                                            seed=0)
        print(f'\nsucceed_list: {slist}')
        print(f'fail_list: {flist}\n')

        logging.info(f'{game} iter: {log}')
        os.makedirs(save_path)
        np.save(save_path + f'succeed_actions.npy', slist)
        np.save(save_path + f'fail_actions.npy', flist)

