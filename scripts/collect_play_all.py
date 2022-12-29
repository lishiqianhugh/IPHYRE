import sys
from iphyre.utils import collect_play_all
# player_name = sys.argv[1]
collect_play_all('lsq', max_episode=1, save_path='../data/player_data.json')
