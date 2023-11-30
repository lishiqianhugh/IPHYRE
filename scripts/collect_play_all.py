import os
import sys

from iphyre.utils import collect_play_all

# config
try:
    player_name = sys.argv[1]
except IndexError:
    raise ValueError('Please specify your name in the command! such as: \npython collect_play_all.py your_name')

folder_path = './player_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

save_path = f'{folder_path}/{player_name}.json'

# play
collect_play_all(player_name, max_episode=5, save_path=save_path)


