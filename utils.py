import numpy as np
import random
# import torch
import matplotlib.pyplot as plt


def plot_trajectory(traj_x, traj_y):
    plt.scatter(traj_x, traj_y)
    plt.show()


def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True