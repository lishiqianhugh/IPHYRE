import numpy as np
import torch
import torch.nn as nn
from rollout import RolloutBuffer

def train(env, rollouts, policy, params, cache_path, seed=123, other_env = False):
    # SETTING ||m: it is good practice to set seeds when running experiments to keep results comparable
    env.seed(seed)
    device = params.policy_params.device
    rollout_time, update_time = AverageMeter(), AverageMeter()  # Loggers
    rewards, success_times = [], []
    all_traj, node_traj, key_count=[], [], []
    #print("Training model with {} parameters...".format(policy.num_params))

    # Training Loop
    for j in range(params.num_updates):
        ## Initialization
        avg_eps_reward = AverageMeter()
        trunctrated = False
        terminated = False
        water_reward = False
        prev_obs = env.reset()
        #pdb.set_trace()
        env.reset_memory()
        prev_obs = torch.tensor(prev_obs, dtype=torch.float32).to(device)
        eps_reward = 0.
        bouts_time = 0
        start_time = time.time()
        
        ## Collect rollouts
        ## when return_back is True, there is no success signal, the agent is expected to wander between home and key
        for step in (range(rollouts.rollout_size)):
            if terminated:
                avg_eps_reward.update(eps_reward)
                bouts_time+=1
                all_traj.append((env.all_trajectories))
                node_traj.append((env.end_trajectories))
                key_count.append(env.key_count)
                # if(env.key_count):
                #     print(f'traj:{len(key_count)} key_count: {env.key_count}')
                obs = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
                eps_reward = 0.
                # Reset Environment
            else:
                obs = prev_obs
            mask = env.act_mask().to(device)
            action, log_prob = policy.act(obs,mask)
            obs, reward, terminated, tructrated = env.step(action)
            # if terminated:
            #     rollouts.insert(step, torch.tensor(terminated, dtype=torch.float32), action, log_prob, 
            #                     torch.tensor(reward, dtype=torch.float32), 
            #                     prev_obs,mask)
            # else:
            rollouts.insert(step, torch.tensor(terminated, dtype=torch.float32), action, log_prob, 
                            torch.tensor(reward, dtype=torch.float32), 
                            prev_obs,mask)
            
            prev_obs = torch.tensor(obs, dtype=torch.float32).to(device)
            eps_reward += reward
            avg_eps_reward.update(eps_reward)


        # Use the rollout buffer's function to compute the returns for all stored rollout steps. (requires just 1 line)
        rollouts.compute_returns(params['discount'])
        rollout_done_time = time.time()

        
        # Call the policy's update function using the collected rollouts        
        policy.update(rollouts)
        update_done_time = time.time()
        rollouts.reset()
        #pdb.set_trace()
        # log metrics
        torch.save(policy.actor.state_dict(), f'{cache_path}/' + f'policy_model{j+1}.pt')
        rewards.append(avg_eps_reward.avg)
        success_times.append(bouts_time)
        rollout_time.update(rollout_done_time - start_time)
        update_time.update(update_done_time - rollout_done_time)
        # print('it {}: avgR: {:.3f} --bouts_time: {:.3f} -- rollout_time: {:.3f}sec -- update_time: {:.3f}sec'.format(j, 
        #                                                                                         avg_eps_reward.avg,bouts_time, 
        #                                                                                         rollout_time.avg,
        #                                                                                        update_time.avg))
        print('it {}: avgR: {:.3f} --bouts_time: {:.3f}'.format(j, avg_eps_reward.avg,bouts_time))
        # if j % params.plotting_iters == 0 and j != 0:
        #     plot_learning_curve(rewards,params.num_updates)
        #     log_policy_rollout(policy, params.env_name, pytorch_policy=True, other_env = other_env)
       # this removes all training outputs to keep the notebook clean, DON'T REMOVE THIS LINE!
    return rewards, success_times, all_traj, node_traj, key_count


