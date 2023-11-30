import numpy as np
import ray
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from IPHYREEnv import IPHYRE_inadvance, IPHYRE_onthefly, IPHYRE_combine
from utils import setup_seed
from iphyre.games import GAMES
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Plan situ Parameters')
    parser.add_argument('--strategy', required=False, type=str, default='onthefly', choices=['inadvance', 'onthefly', 'combine'],)
    parser.add_argument('--seed', type=int, help='training seed', default=0)
    parser.add_argument('--model', type=str, help='name', default='PPO', choices=['DDPG', 'DQN', 'A2C', 'PPO', 'SAC'])
    parser.add_argument('--checkpoint_path', type=str, help='where to save checkpoint', default='./checkpoints')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--id', type=int, default=200)

    return parser.parse_args()


ray.init()
args = arg_parse()
setup_seed(args.seed)
FOLD_LIST = ['basic', 'noisy', 'compositional', 'multi_ball']
NUM_GAMES = len(GAMES)
NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
TRAIN_SPLIT = GAMES[: NUM_PER_GROUP]
env = f"IPHYRE_{args.strategy}"

config = (
    eval(f"{args.model}Config")()
    .environment(
        eval(env),
        env_config={
            "game_list": TRAIN_SPLIT,
            "seed": args.seed
        },
    )
    .rollouts(num_rollout_workers=1)
    .framework('torch')
    .debugging(seed=args.seed)
    .resources(num_gpus=0)
)

algo = config.build()
best_iter = 0
best_basic_reward = -1000
path = f"{args.checkpoint_path}/{args.strategy}/{args.model}/{args.lr}/{args.seed}/"

for iter in range(5):
    id = (6-len(str(args.id))) * '0' + str(args.id)
    model_path = f"{args.checkpoint_path}/{args.strategy}/{args.model}/{args.lr}/{args.seed}/checkpoint_{id}"
    algo.restore(model_path)
    print(f"Loading {model_path}")
    eval_rewards = []
    for FOLD in FOLD_LIST:
        print(f"Evaluating {FOLD}")
        FOLD_ID = FOLD_LIST.index(FOLD)
        TEST_SPLIT = GAMES[FOLD_ID * NUM_PER_GROUP: (FOLD_ID + 1) * NUM_PER_GROUP]
        for game in TEST_SPLIT:
            ENV = eval(env)({"game_list": [game]})
            obs, _ = ENV.reset()
            terminated = truncated = False
            total_reward = 0.0
            # Play one episode.
            while not terminated and not truncated:
                # Compute a single action, given the current observation from the environment.
                action = algo.compute_single_action(obs)
                # Apply the computed action in the environment.
                obs, reward, terminated, truncated, _ = ENV.step(action)
                # Sum up rewards for reporting purposes.
                total_reward += reward
            # Report results.
            eval_rewards.append(total_reward)
            print(f"{game}: {total_reward}")

    fold_rewards = []
    for i, fold in enumerate(FOLD_LIST):
        rewards = eval_rewards[i * 10: (i + 1) * 10]
        int_rewards = [float(r) for r in rewards]
        avg_rewards = sum(int_rewards) / 10
        fold_rewards.append(avg_rewards)
    success_num = (np.array(eval_rewards) > 0).sum()
    if fold_rewards[0] > best_basic_reward:
        best_iter = iter
        best_basic_reward = fold_rewards[0]
    print("Iter: ", iter, " ", fold_rewards, " ", success_num, "\n", eval_rewards)
print(f"Best iter: {best_iter} {best_basic_reward}")
ray.shutdown()
