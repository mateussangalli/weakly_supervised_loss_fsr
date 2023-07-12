import numpy as np
import os
import argparse

from utils.data_loading import read_dataset
from utils.size_regularization import get_mean_height
from labeled_images import LABELED_IMAGES

SC = 0
LED = 2

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
parser.add_argument("--num_images_labeled", type=int, default=3)
args = parser.parse_args()

# load data

def get_height(gt, class_num):
    mask = gt == class_num
    return np.sum(mask, 0)



data_train = read_dataset(args.data_root, "train")

mean_heights_sc = list()
mean_heights_led = list()
mean_ratios = list()
for i, (_, gt) in enumerate(data_train):
    heights_sc = get_height(gt, SC)
    heights_led = get_height(gt, LED)
    min_sc = np.min(heights_sc)
    max_sc = np.max(heights_sc)
    mean_sc = np.mean(heights_sc)
    min_led = np.min(heights_led)
    max_led = np.max(heights_led)
    mean_led = np.mean(heights_led)
    min_ratio = np.min(heights_led / heights_sc)
    max_ratio = np.max(heights_led / heights_sc)
    mean_ratio = np.mean(heights_led / heights_sc)

    mean_heights_sc.append(mean_sc)
    mean_heights_led.append(mean_led)
    mean_ratios.append(mean_ratio)

    print(f'image {i}:')
    print(f'{min_ratio=}')
    print(f'{max_ratio=}')
    print(f'{mean_ratio=}')
    print(f'{min_led=}')
    print(f'{max_led=}')
    print(f'{mean_led=}')
    print(f'{min_sc=}')
    print(f'{max_sc=}')
    print(f'{mean_sc=}')

print(f'min sc: {np.min(mean_heights_sc)}, max sc: {np.max(mean_heights_sc)}')
print(f'min led: {np.min(mean_heights_led)}, max led: {np.max(mean_heights_led)}')
print(f'min ratios: {np.min(mean_ratios)}, max ratios: {np.max(mean_ratios)}')
