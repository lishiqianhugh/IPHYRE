import sys
from games.simulator import collect_play_all

collect_play_all(sys.argv[1], max_episode=2, save_path ='./dataset/player_data.json')
