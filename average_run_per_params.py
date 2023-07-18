import argparse
import json
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("runs_dir", type=str)
args = parser.parse_args()

results = dict()

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

    try:
        weight = params['max_weight']
        height_weight = params['height_reg_weight']
        key = f'({weight:.5f},{height_weight:.5f})'
    except KeyError as e:
        key = 'fully_labeled'
    if key not in results:
        results[key] = {}
        for column in df:
            results[key][column] = []
            results[key]['num_runs'] = 0

    for column in df:
        results[key][column].append(df[column][0])
    results[key]['num_runs'] += 1

for k, v in results.items():
    print(f'{k}:')
    num_runs = v['num_runs']
    print(f'   num_runs: {num_runs}')
    for k2, v2 in v.items():
        if k2 == 'num_runs':
            continue
        v2 = np.mean(v2)
        print(f'   {k2}: {v2}')
