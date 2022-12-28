import torch.nn.functional as F
import torch
from iphyre.games import GAMES
from utils import setup_seed, write_csv
import logging
import argparse
import optuna
from agents.config.DQN_config import BaseConfig as DQNConfig
from agents.config.A2C_config import BaseConfig as A2CConfig
from agents.plan_in_situ.DQN import *
from agents.plan_in_situ.A2C import *
import os
import pdb


def arg_parse():
    parser = argparse.ArgumentParser(description='Plan situ Parameters')
    parser.add_argument('--search', type=bool, help='whether searching hyperparameters', default=False)
    parser.add_argument('--mode', required=False, type=str, default='single',
                        choices=['single', 'within', 'cross'])
    parser.add_argument('--seed', type=int, help='training seed', default=0)
    parser.add_argument('--model', type=str, help='model name', default='A2C')

    return parser.parse_args()

args = arg_parse()
# device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using', device)

def run(config):
    use_images = config.use_images
    all_test_rewards = []
    FOLD_LIST = ['basic', 'compositional', 'noisy', 'multi_ball']
    for FOLD in FOLD_LIST:
        FOLD_ID = FOLD_LIST.index(FOLD)
        NUM_GAMES = len(GAMES)
        NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
        TEST_SPLIT = GAMES[FOLD_ID * NUM_PER_GROUP: (FOLD_ID + 1) * NUM_PER_GROUP]
        TRAIN_SPLIT = GAMES[:FOLD_ID * NUM_PER_GROUP] + GAMES[(FOLD_ID + 1) * NUM_PER_GROUP:]

        logging.info(f'Fold: {FOLD}, Seed: {args.seed}')
        if args.mode == 'within':
            model = eval(args.model)(device,config)
            model.train(TEST_SPLIT)
            test_rewards = model.test(TEST_SPLIT)
            all_test_rewards += test_rewards
        elif args.mode == 'cross':
            model = eval(args.model)(device,config)
            model.train(TRAIN_SPLIT)
            test_rewards = model.test(TEST_SPLIT)
            all_test_rewards += test_rewards
        elif args.mode == 'single':
            for game in TEST_SPLIT:
                model = eval(args.model)(device,config)
                model.train([game])
                test_rewards = model.test([game])
                all_test_rewards += test_rewards
        else:
            raise ValueError(f'No such mode {args.mode}')
    
    return all_test_rewards

def objective(trial):
    gamma = trial.suggest_float('gamma', 0.5, 1.)
    epsilon = trial.suggest_float('epsilon', 0.5, 1.)
    lr = trial.suggest_float('lr', 0.001, 0.01)
    target_replace_iter = trial.suggest_int('target_replace_iter', 50, 200, 50)
    all_test_rewards = run(config)

    return sum(all_test_rewards)

# logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
# logging.basicConfig(filename=f'logs/plan_in_situ/DQN_{args.mode}_{args.use_images}_{args.epoch}_{args.gamma}_{args.lr}.log', level=20,
#                     format=LOG_FORMAT, datefmt=DATE_FORMAT)
if not os.path.exists('logs/plan_in_situ/'):
    os.makedirs('logs/plan_in_situ/')
logging.basicConfig(filename=f'logs/plan_in_situ/{args.model}_{args.mode}.log', level=20,
                     format=LOG_FORMAT, datefmt=DATE_FORMAT)
logging.info(args)

if __name__ == '__main__':
    setup_seed(args.seed)
    all_test_rewards = []
    if(args.model == 'DQN'):
        config = DQNConfig()
    if(args.model == 'A2C'):
        config = A2CConfig()
    FOLD_LIST = ['basic', 'compositional', 'noisy', 'multi_ball']
    for FOLD in FOLD_LIST:
        FOLD_ID = FOLD_LIST.index(FOLD)
        NUM_GAMES = len(GAMES)
        NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
        TEST_SPLIT = GAMES[FOLD_ID * NUM_PER_GROUP: (FOLD_ID + 1) * NUM_PER_GROUP]
        TRAIN_SPLIT = GAMES[:FOLD_ID * NUM_PER_GROUP] + GAMES[(FOLD_ID + 1) * NUM_PER_GROUP:]

        logging.info(f'Fold: {FOLD}, Seed: {args.seed}')
    if args.search == False:
        all_test_rewards = run(config)
        write_csv(path=f'logs/plan_in_situ/{args.model}_{args.mode}_{args.use_images}_{args.epoch}_{args.gamma}_{args.lr}_rewards.csv', contents=[GAMES, all_test_rewards])
    else:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        print('best params:', study.best_trial.params,
              '\n', 'best reward:', study.best_trial.values)
