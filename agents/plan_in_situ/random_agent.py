import sys
sys.path.append('D:\Files\Research\Projects\Interactive_Physical_Reasoning\IPHYRE')
import numpy as np
from iphyre.simulator import IPHYRE
from iphyre.games import PARAS
from utils import setup_seed, write_csv
import time
import logging


SEED = 0
N_ACTIONS = 7
GAME_TIME = 15.
MAX_ITER = 150

# logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'logs/plan_in_situ/Random.log', level=20,
                    format=LOG_FORMAT, datefmt=DATE_FORMAT)


class RandomAgent(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self):
        action = np.random.randint(0, self.n_actions)
        return action


def test(model, test_split):
    rewards = []
    for game in test_split:
        t1 = time.time()
        env = IPHYRE(game=game)
        actions = env.get_action_space()
        best_reward = -100000
        total_reward = 0
        _ = env.reset()
        iter = 0
        while iter < MAX_ITER:
            a = model.choose_action()
            pos = actions[a]
            _, r, done = env.step(pos, time_step=GAME_TIME / MAX_ITER)

            total_reward += r
            iter += 1

            if done:
                break
        if total_reward > best_reward:
            best_reward = total_reward

        t2 = time.time()
        rewards.append(best_reward)
        info = f'Testing Game: {game} | Testing Duration: {round(t2 - t1, 2)} | Reward: {best_reward}'
        print(info)
        logging.info(info)
    return rewards


if __name__ == '__main__':
    setup_seed(SEED)
    all_test_rewards = []
    FOLD_LIST = ['basic', 'compositional', 'noisy', 'multi_ball']
    for FOLD in FOLD_LIST:
        FOLD_ID = FOLD_LIST.index(FOLD)
        GAMES = list(PARAS.keys())
        NUM_GAMES = len(GAMES)
        NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
        TEST_SPLIT = GAMES[FOLD_ID * NUM_PER_GROUP: (FOLD_ID + 1) * NUM_PER_GROUP]
        TRAIN_SPLIT = GAMES[:FOLD_ID * NUM_PER_GROUP] + GAMES[(FOLD_ID + 1) * NUM_PER_GROUP:]

        logging.info(f'Fold: {FOLD}, Seed: {SEED}')
        random_agent = RandomAgent(N_ACTIONS)
        test_rewards = test(random_agent, TEST_SPLIT)
        all_test_rewards += test_rewards
    write_csv(path=f'logs/plan_in_situ/Random_rewards.csv', contents=[list(PARAS.keys()), all_test_rewards])
        