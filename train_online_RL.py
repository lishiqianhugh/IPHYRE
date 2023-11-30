import argparse
from iphyre.games import GAMES
import ray
from ray import air, tune
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from IPHYREEnv import IPHYRE_inadvance, IPHYRE_onthefly, IPHYRE_combine
from utils import setup_seed


def arg_parse():
    parser = argparse.ArgumentParser(description='Plan situ Parameters')
    parser.add_argument('--strategy', required=False, type=str, default='inadvance',
                        choices=['inadvance', 'onthefly', 'combine', 'onthefly_continuous'], )
    parser.add_argument('--model', type=str, help='name', default='PPO',
                        choices=['DDPG', 'DQN', 'A2C', 'PPO', "SAC"])
    parser.add_argument('--seed', type=int, help='training seed', default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--checkpoint_path', type=str, help='where to save checkpoint', default='./checkpoints')
    parser.add_argument('-resume', type=bool, default=False)
    parser.add_argument('-resume_ep', type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    ray.init()
    args = arg_parse()
    setup_seed(args.seed)
    FOLD_LIST = ['basic', 'noisy', 'compositional', 'multi_ball']
    NUM_GAMES = len(GAMES)
    NUM_PER_GROUP = int(NUM_GAMES / len(FOLD_LIST))
    TRAIN_SPLIT = GAMES[: NUM_PER_GROUP]
    env = f"IPHYRE_{args.strategy}"
    # Training
    config = (
        eval(f"{args.model}Config")()
        .environment(
            eval(env),
            env_config={
                "game_list": TRAIN_SPLIT,
                "seed": args.seed
            },
        )
        .rollouts(num_rollout_workers=args.num_workers)
        .framework('torch')
        .debugging(
            seed=args.seed,
        )
        .resources(num_gpus=0)
        .training(lr=args.lr)
    )

    algo = config.build()
    if args.resume:
        model_path = f"{args.checkpoint_path}/{args.strategy}/{args.model}/{args.lr}/{args.seed}/checkpoint_000{args.resume_ep}"
        algo.restore(model_path)
        for i in range(args.resume_ep, args.resume_ep + 200):
            results = algo.train()
            checkpoint_path = algo.save(
                f'{args.checkpoint_path}/{args.strategy}/{args.model}/{args.lr}/{args.seed}')
            print(f"Saving {checkpoint_path}")
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
    else:
        for i in range(200):
            results = algo.train()
            checkpoint_path = algo.save(
                f'{args.checkpoint_path}/{args.strategy}/{args.model}/{args.lr}/{args.seed}')
            print(f"Saving {checkpoint_path}")
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
