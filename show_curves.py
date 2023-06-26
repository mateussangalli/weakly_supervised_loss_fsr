import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread


def get_average_df(pattern, runs_dir):
    run_names = os.listdir(runs_dir)
    run_names = filter(lambda s: pattern in s, run_names)
    df_list = [pd.read_csv(os.path.join(runs_dir, name, 'training_history.csv')) for name in run_names]
    df = pd.concat(df_list)
    return df.groupby(df.index).mean()

df_labeled = pd.read_csv('labeled_runs/weight0p0000_2023-06-22_17-00-10/training_history.csv')
iou_labeled = df_labeled['val_one_hot_mean_io_u'][len(df_labeled)-1]
reg_labeled = df_labeled['val_directional_loss_metric'][len(df_labeled)-1]


weights = [0.0, 1.0, 5.0, 10.0]
plt.figure()
plt.axhline(iou_labeled, color='black', linestyle='--', label='only labeled, no reg, val')
for weight in weights:
    weight_str = f"{weight:.4f}".replace(".", "p")
    df = get_average_df(weight_str, "pseudo_labels_runs")
    plt.plot(df['val_one_hot_mean_io_u'], label=f'weight = {weight:.2f}, val')
    # plt.plot(df['val_one_hot_mean_io_u'], label=f'weight = {weight:.2f}, train', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('IoU')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.axhline(reg_labeled, color='black', linestyle='--', label='only labeled, no reg, val')
for weight in weights:
    weight_str = f"{weight:.4f}".replace(".", "p")
    df = get_average_df(weight_str, "pseudo_labels_runs")
    plt.plot(df['val_directional_loss_metric'], label=f'weight = {weight:.2f}')
    # plt.plot(df['val_directional_loss_metric'], label=f'weight = {weight:.2f}, train', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('Directional Regularization')
plt.grid()
plt.legend()
plt.show()
