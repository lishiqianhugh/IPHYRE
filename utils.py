import numpy as np
import random
# import torch
import matplotlib.pyplot as plt
import cv2


def plot_trajectory(traj_x, traj_y):
    plt.scatter(traj_x, traj_y)
    plt.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def draw_bbox(im_path, vectors_path):
    """
    There is a -1 bug in x1, y1, x2, y2, r
    """
    image = cv2.imread(im_path)
    vectors = np.load(vectors_path)
    for vector in vectors:
        x1, y1, x2, y2, r = vector[:5]
        x1, y1, x2, y2, r = int(y1)-1, int(x1)-1, int(y2)-1, int(x2)-1, int(r)-1
        x1 = min(x1, x2)
        x2 = max(x1, x2)
        y1 = min(y1, y2)
        y2 = max(y1, y2)
        print(x1, y1, x2, y2, r)
        image[x1 - r: x2 + r, y1 - r] = [0, 0, 0]
        image[x1 - r: x2 + r, y2 + r] = [0, 0, 0]
        image[x1 - r, y1 - r: y2 + r] = [0, 0, 0]
        image[x2 + r, y1 - r: y2 + r] = [0, 0, 0]
    cv2.imshow('bbox_image', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    draw_bbox('../dataset/game_initial_data/spring_flick/spring_flick.jpg',
              '../dataset/game_initial_data/spring_flick/vectors.npy')
