import argparse
import json
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("runs_dir", type=str)
args = parser.parse_args()

best_ious = dict()

for run_id in os.listdir(args.runs_dir):
    run_path = os.path.join(args.runs_dir, run_id)
    try:
        df = pd.read_csv(os.path.join(run_path, 'training_history.csv'))
    except FileNotFoundError as e:
        print(e)
        print(f'Did not find history on {run_path}. Skipping.')
        continue
    params_path = os.path.join(run_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)

    best_val_iou = df['val_one_hot_mean_io_u'].max()
    weight = run_id[6:12]
    if weight not in best_ious or best_val_iou > best_ious[weight]['best_iou']:
        best_ious[weight] = params
        best_ious[weight]['best_iou'] = best_val_iou

print(best_ious)

best_runs_path = os.path.join('best_runs.json')
with open(best_runs_path, 'w') as fp:
    json.dump(best_ious, fp)
