import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("runs_dir", type=str)
args = parser.parse_args()

for run_id in os.listdir(args.runs_dir):
    run_path = os.path.join(args.runs_dir, run_id)
    try:
        df = pd.read_csv(os.path.join(run_path, 'training_history.csv'))
    except FileNotFoundError as e:
        print(e)
        print(f'Did not find history on {run_path}. Skipping.')
        continue

    iou_path = os.path.join(run_path, 'mIoU_train_eval.pdf')
    plt.figure()
    plt.plot(df['epoch'], df['val_one_hot_mean_io_u'], label='validation')
    plt.plot(df['epoch'], df['one_hot_mean_io_u'], label='train')
    plt.grid()
    plt.legend()
    plt.savefig(iou_path, bbox_inches='tight')

    reg_path = os.path.join(run_path, 'reg_train_eval.pdf')
    plt.figure()
    plt.plot(df['epoch'], df['val_directional_loss_metric'], label='validation')
    plt.plot(df['epoch'], df['directional_loss_metric'], label='train')
    plt.grid()
    plt.legend()
    plt.savefig(reg_path, bbox_inches='tight')

