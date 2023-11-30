import torch
import numpy as np
import random
import csv


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def write_csv(path, contents, mode='a'):
    with open(path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(contents)
 