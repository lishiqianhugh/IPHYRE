import sys
from iphyre.utils import collect_play_all
# player_name = sys.argv[1]
player_name = 'your_name'
collect_play_all(player_name, max_episode=1, save_path=f'../data/{player_name}.json')
