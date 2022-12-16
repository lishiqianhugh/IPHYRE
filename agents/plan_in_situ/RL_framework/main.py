import numpy as np
import matplotlib.pyplot as plt 
import gym
import cv2
import time
import copy
import os
from tqdm import tqdm
from gym.wrappers import RecordVideo
#from IPython.display import Video
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils.environment.wrappers import *
from utils.environment.minigrid import *   # the minigrid is different from the original one(add new object 'line')
#from IPython.display import clear_output

from dataProcess import *

from micemaze import MiceMazeEnv as MiceMazeEnv_origin    #The implementation code is too long so I put it into a file
from train import *
from test import *
from policies import Policy
from utils.util import *
import warnings
warnings.filterwarnings('ignore')
from gym import spaces
from utils.environment.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
import pandas as pd
import argparse
import wandb
import pdb

from stable_baselines3 import A2C,DDPG,PPO,DQN,SAC

class MiceMazeEnv(MiceMazeEnv_origin):
    
    def key_reward(self):
        """
        Compute the reward to be given upon success
        """     
        #return np.tanh((2*self.max_steps)/self.step_count)
        return 1 - self.kr_c*(self.step_count/self.max_steps)    
    
    def home_reward(self):
        
        return max(0, 1 - self.hr_c*(self.max_steps/(100*self.step_count)))
    
    def intristic_reward(self,Nt,Nt_next,all_index):
        
        return 0
        # if(level>5):
        #     node_reward = self.level_of_all_index(all_index)*0.01
        # return node_reward
    

class MiceMazeEnv1(MiceMazeEnv_origin):
    
    def key_reward(self):
        """
        Compute the reward to be given upon success
        """     
        #return np.tanh((2*self.max_steps)/self.step_count)
        return 1 - self.kr_c*(self.step_count/self.max_steps)    
    
    def home_reward(self):
        if(self.key_count==0): return 0
        else:
            pdb.set_trace()
            return max(0, 1 - self.hr_c*(self.max_steps/(100*self.step_count)))
    
    def intristic_reward(self,Nt,Nt_next,all_index):
        
        return 0



class MiceMazeEnv_intristic(MiceMazeEnv_origin):
    
    def key_reward(self):
        """
        Compute the reward to be given upon success
        """     
        #the more you take the water, the less results will get
        #return np.tanh((2*self.max_steps)/self.step_count)
        return 1 - self.kr_c*(self.step_count/self.max_steps)    
    
    def home_reward(self):
        if(self.key_count==0): return 0
        else:
            pdb.set_trace()
            return max(0, 1 - self.hr_c*(self.max_steps/(100*self.step_count)))
    
    def intristic_reward(self,Nt,Nt_next,all_index):
        if(Nt_next<2):
            return max(0,0.5*(1/Nt_next -1/Nt))
        else:
            return 0

class MiceMazeEnv_test(MiceMazeEnv_origin):
    
    def key_reward(self):
        """
        Compute the reward to be given upon success
        """     
        if(self.key_count==1): 
            print(f'step_count {self.step_count}')
            return max(2, 10-0.1*(self.step_count-30))
        else:
            interval_step = self.findKey_steps[-1]-self.findKey_steps[-2]
            if(interval_step>90):
                return 10
            else:
                return 0
        #the more you take the water, the less results will get
        #return np.tanh((2*self.max_steps)/self.step_count)
        # if(len(self.findKey_steps)==1): 
        #     interval_step = self.findKey_steps[-1]
        # else: interval_step = self.findKey_steps[-1]-self.findKey_steps[-2]
        # return max(0,1 - self.kr_c*(self.step_record[self.agent_end_pos]/interval_step))    
    
    def home_reward(self):
        if(self.key_count==0): 
            return 0
        else:
            return 0.1
    
    def intristic_reward(self,Nt,Nt_next,all_index):
        node_reward = 0
        exp_reward = 0
        exp_reward = 1/Nt_next -1/Nt 
        # if(Nt_next==1):
        #     exp_reward = 1/Nt_next -1/Nt 
        #     level = self.level_of_all_index(all_index)
        #     if(level>5):
        #         node_reward = level*0.1
        return node_reward+exp_reward
     
def arg_parse():
    parser = argparse.ArgumentParser(description='MiceMaze')
    parser.add_argument('--goal', required=False, type=str, default='stable baseline exp offline')
    parser.add_argument('--env', required=False, type=str, default='MiceMazeEnv')
    parser.add_argument('--polices', required=False, type=list, default=['DDPG'])
    parser.add_argument('--hidden_dim', required=False, type=int, default=10)
    parser.add_argument('--learning_rate', required=False, type=int, default=1e-3)
    parser.add_argument('--batch_size', required=False, type=int, default=1000)
    parser.add_argument('--policy_epochs', required=False, type=int, default=10)
    parser.add_argument('--entropy_coef', required=False, type=int, default=0.001)
    parser.add_argument('--rollout_size', required=False, type=int, default=10000)
    parser.add_argument('--num_updates', required=False, type=int, default=50)
    parser.add_argument('--discount', required=False, type=int, default=0.99)
    parser.add_argument('--plotting_iters', required=False, type=int, default=100)
    parser.add_argument('--env_name', required=False, type=str, default=None)
    parser.add_argument('--max_steps', required=False, type=int, default=10000)
    parser.add_argument('--seeds', required=False, type=list, default = [123])#[0,24,45,67,89,35,74,45,93])#[125,2,27,50,68,99,36,76,49,98])#123,
    parser.add_argument('--kr_c', required=False, type=float, default=0.09)
    parser.add_argument('--hr_c', required=False, type=float, default=0.09)
    parser.add_argument('--cache_path', required=False, type=str, default='cached_policy_for_StableBaseline')
    parser.add_argument('--test', required=False, type=bool, default=False)
    return parser.parse_args()



class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)
        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSpace.shape[0] * imgSpace.shape[1]* imgSpace.shape[2],),
            dtype="uint8",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]

        obs = image.flatten()

        return obs


def instantiate(params_in, nonwrapped_env=None):
    
    params = copy.deepcopy(params_in)

    if nonwrapped_env is None:
        nonwrapped_env = gym.make(params.env_name)

    env = None
    env = FlatObsWrapper(nonwrapped_env)    
    obs_size = env.observation_space.shape[0]
    print(obs_size)
    if(isinstance(env.action_space, spaces.Box)):
        num_actions = env.action_space.high-env.action_space.low+1
    else:
        num_actions = env.action_space.n
    rollouts = RolloutBuffer(params.rollout_size, obs_size,params.policy_params.device)
    policy_class = params.policy_params.pop('policy_class')
    if(isinstance(policy_class, str)):
        policy_class = eval(policy_class)
    #policy = policy_class(obs_size, num_actions, **params.policy_params)
        policy = policy_class("MlpPolicy", env, verbose=1,n_steps=params.rollout_size)
        print(policy.action_space)
    else:
        policy = policy_class(obs_size, num_actions, **params.policy_params)

    return env, rollouts, policy



def sweep(args):
    #with wandb.init(config = args) as run:
        # Overwrite the random run names chosen by wandb
    print(args.kr_c, args.hr_c)
    for policy in args.polices:
        print(policy)
        policy_params = ParamDict(
            policy_class = policy,    # Policy class to use (replaced later)     
            hidden_dim =args.hidden_dim,          # dimension of the hidden state in actor network
            learning_rate = args.learning_rate,     # learning rate of policy update
            batch_size = args.batch_size,        # batch size for policy update
            policy_epochs = args.policy_epochs,        # number of epochs per policy update
            entropy_coef = args.entropy_coef,     # hyperparameter to vary the contribution of entropy loss
            device = device
        )

        params = ParamDict(
            policy_params = policy_params,
            rollout_size = args.rollout_size,      # number of collected rollout steps per policy update
            num_updates = args.num_updates,         # number of training policy iterations
            discount = args.discount,          # discount factor
            plotting_iters = args.plotting_iters,      # interval for logging graphs and policy rollouts
            env_name = args.env_name,
            max_steps = args.max_steps, 
            kr_c = args.kr_c,
            hr_c = args.hr_c
        )
         #---creat_path--
        exp_dir = f'{args.goal}/{args.env}_{args.kr_c}_{args.hr_c}/{policy}'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        df_pre = pd.DataFrame(columns=['seed','SF','SA','BF','BS'])
        for seed in args.seeds:
            setup_seed(seed)
            cache_dir = f'{args.cache_path}/{policy}/{seed}'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            nonwrapped_env = eval(args.env)(max_steps = params.max_steps,hr_c = params.hr_c, kr_c = params.kr_c, agent_view_size=7,action_space_type ='box' )
            env, rollouts, policy = instantiate(params,nonwrapped_env)
            if(args.test):
                for file in os.listdir(cache_dir):
                    policy.actor.load_state_dict(torch.load(f'{cache_dir}/{file}'))
                    rewards, success_times, all_traj, bouts, key_count = test(env, rollouts, policy, params, cache_dir, seed=seed)
            if(args.goal =='stable baseline exp offline'):
                policy.learn(total_timesteps=50*args.rollout_size)
            else: 
                rewards, success_times, all_traj, bouts, key_count = train(env, rollouts, policy, params, cache_dir, seed=seed)
            sort_id = sort_traj(policy.node_traj)
            select_id = sort_id[:100]
            preference = caculate_preference(policy.node_traj,env,select_id)
            preference['seed']  = seed
            df_pre = df_pre.append(preference,ignore_index=True)
            #wandb.log(preference)
        # wandb.log({'P_BF vs P_SF': wandb.plot([bi[:10,0],bi[10:,0],[2/3]],[bi[:10,2],bi[10:,2],[2/3]],fmts=['r.','k+'],markersize=5,
        # xlim=[0,1],ylim=[0,1],equal=True,legend=['rewarded','random'],
        # xlabel='$P_{\mathrm{SF}}$',ylabel='$P_{\mathrm{BF}}$',loc='lower left')})
        data = {'all_traj':policy.all_traj,'bout':policy.node_traj,'key_count':policy.key_count}
        df = pd.DataFrame(data)
        df.to_csv(f'{exp_dir}/{seed}.csv')

    print("Training completed!")
    print(df_pre)
    #wandb.log({'table':df_pre})


args = arg_parse()
# ----wandb----
# sweep_config = {
#     "name": "reward_coef_test",
#     "method": "grid",
#     "metric": {"name": "preference", "goal": "maximize"},
#     "parameters": {
#         "kr_c":{
#             "values":[0.009,0.09,0.9],
#         },
#         "hr_c":{
#             "values":[0.009,0.09],
#         },
#     },
# }
#TODO: 

# run = wandb.init(config=args)
# args = wandb.config
#run.name = f'{args.goal}_{args.env}_{args.kr_c}_{args.hr_c}'


# Hyperparameters
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'using device {device}')

sweep(args)

if(args.test):
    pred_model.load_state_dict(torch.load(f'{args.pred_path}'))
    policy.load()
    torch.save(pred_model.state_dict(), save_path + f'_pred_model{i+1}.pt')