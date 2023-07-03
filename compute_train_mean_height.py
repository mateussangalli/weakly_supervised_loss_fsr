import numpy as np
import os
import argparse

from utils.data_loading import read_dataset
from utils.size_regularization import get_mean_height
from labeled_images import LABELED_IMAGES

SC = 0
LED = 2

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="../prp_loreal_data")
parser.add_argument("--num_images_labeled", type=int, default=3)
args = parser.parse_args()

# load data


data_train = read_dataset(args.data_root, "train", LABELED_IMAGES)

for name, (_, gt) in zip(LABELED_IMAGES, data_train):
    gt = np.eye(3)[gt]
    mean_height_sc = float(get_mean_height(gt[np.newaxis, ...], SC))
    mean_height_led = float(get_mean_height(gt[np.newaxis, ...], LED))
    print(f'image: {name}')
    print(f'{mean_height_sc=}')
    print(f'{mean_height_led=}')
