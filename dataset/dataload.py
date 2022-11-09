import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import itertools
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from tqdm import tqdm





class PHYREO(Dataset):
    def __init__(self, split):

        self.total_size = cfg.INPUT.INPUT_SIZE + cfg.NET.PRED_SIZE
        self.split = split
        self.input_height, self.input_width = cfg.INPUT.INPUT_HEIGHT, cfg.INPUT.INPUT_WIDTH

        protocal = cfg.PHYRE_PROTOCAL
        fold = cfg.PHYRE_FOLD

        num_pos = 100 if split == 'train' else 1
        num_neg = 100 if split == 'train' else 1

        eval_setup = f'ball_{protocal}_template'
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold)
        tasks = train_tasks + dev_tasks if split == 'train' else test_tasks
        #TODO:
        tasks = tasks[:5]
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)

        # all the actions
        cache = phyre.get_default_100k_cache('ball')
        training_data = cache.get_sample(tasks, None)
        # (100000 x 3)
        actions = training_data['actions']
        # (num_tasks x 100000)
        sim_statuses = training_data['simulation_statuses']

        self.simulator = phyre.initialize_simulator(tasks, action_tier)

        self.video_info = np.zeros((0, 4))
        for t_id, t in enumerate(tqdm(tasks)):
            sim_status = sim_statuses[t_id]
            pos_acts = actions[sim_status == 1].copy()
            neg_acts = actions[sim_status == -1].copy()
            np.random.shuffle(pos_acts)
            np.random.shuffle(neg_acts)
            pos_acts = pos_acts[:num_pos]
            neg_acts = neg_acts[:num_neg]
            acts = np.concatenate([pos_acts, neg_acts])
            video_info = np.zeros((acts.shape[0], 4))
            video_info[:, 0] = t_id
            video_info[:, 1:] = acts
            self.video_info = np.concatenate([self.video_info, video_info])

    def __len__(self):
        return self.video_info.shape[0]

    def __getitem__(self, idx):
        task_id, acts = self.video_info[idx, 0], self.video_info[idx, 1:]
        sim = self.simulator.simulate_action(
            int(task_id), acts, stride=60, need_images=True, need_featurized_objects=True
        )
        images = sim.images
        tmps = []
        for i in range(self.total_size):
            try:
                tmp = cv2.resize(images[i], (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                tmps.append(tmp)
            except:
                for j in range(self.total_size-i):
                    tmp = cv2.resize(images[i-1], (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)

                    tmps.append(tmp)
                break

        data = np.array([phyre.observations_to_float_rgb(image) for image in tmps],
                            dtype=np.float).transpose((0, 3, 1, 2))

        data = torch.from_numpy(data.astype(np.float32))
        labels = torch.from_numpy(np.array(int(sim.status == 1), dtype=np.float32))

        return data, labels