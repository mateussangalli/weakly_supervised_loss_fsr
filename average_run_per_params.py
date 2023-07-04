import argparse
import json
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("runs_dir", type=str)
args = parser.parse_args()

ious = dict()

for run_id in os.listdir(args.runs_dir):
    run_path = os.path.join(args.runs_dir, run_id)
    try:
        df = pd.read_csv(os.path.join(run_path, 'average_results_test.csv'))
    except FileNotFoundError as e:
        print(e)
        print(f'Did not find history on {run_path}. Skipping.')
        continue
    params_path = os.path.join(run_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)

    iou = df['0'][0]
    weight = params['max_weight']
    height_weight = params['height_reg_weight']

    key = f'({weight:.4f},{height_weight:.4f})'
    if key not in ious:
        ious[key] = []
    ious[key].append(iou)

for k, v in ious.items():
    v = np.mean(v)
    print(f'{k}: {v}')
