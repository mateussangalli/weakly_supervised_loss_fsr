import numpy as np
import os
import argparse

from utils.data_loading import read_dataset
from utils.size_regularization import get_mean_height

SC = 0
LED = 2

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="../prp_loreal_data")
parser.add_argument("--num_images_labeled", type=int, default=3)
args = parser.parse_args()

# load data
train_images = os.listdir(os.path.join(args.data_root, "train", "images"))
if args.num_images_labeled > 0:
    train_images_labeled = [
        train_images[3],
        train_images[34],
        train_images[64],
        train_images[4],
        train_images[5],
    ][: args.num_images_labeled]
else:
    train_images_labeled = train_images


data_train = read_dataset(args.data_root, "train", train_images_labeled)

for name, (_, gt) in zip(train_images_labeled, data_train):
    gt = np.eye(3)[gt]
    mean_height_sc = float(get_mean_height(gt[np.newaxis, ...], SC))
    mean_height_led = float(get_mean_height(gt[np.newaxis, ...], LED))
    print(f'image: {name}')
    print(f'{mean_height_sc=}')
    print(f'{mean_height_led=}')
