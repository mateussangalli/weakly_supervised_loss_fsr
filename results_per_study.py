import argparse
import json
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("runs_dir", type=str)
args = parser.parse_args()

results = dict()


def get_results_per_study(results_per_image: pd.DataFrame):
    df = results_per_image.copy()
    studies = df['image'].map(lambda s: s[:6])
    unique_studies = pd.unique(studies)

    results_per_study = pd.DataFrame()

    dtypes = ['float32', 'float64']
    for study in unique_studies:
        study_results = df[studies == study]
        numeric_results = study_results.select_dtypes(dtypes)
        mean_results = dict(numeric_results.mean(0))
        for k, v in mean_results.items():
            mean_results[k] = [v]
        mean_results['study'] = study
        results_per_study = pd.concat([results_per_study, pd.DataFrame(mean_results)], ignore_index=True)
    return results_per_study


for run_id in os.listdir(args.runs_dir):
    run_path = os.path.join(args.runs_dir, run_id)
    try:
        results_per_image = pd.read_csv(os.path.join(run_path, 'results_test.csv'))
    except FileNotFoundError as e:
        print(e)
        print(f'Did not find results for run {run_path}. Skipping.')
        continue

    results_per_study = get_results_per_study(results_per_image)

    params_path = os.path.join(run_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)

    weight = params['max_weight']
    height_weight = params['height_reg_weight']
    key = f'({weight:.5f},{height_weight:.5f})'
    if key not in results:
        results[key] = {'results_per_study': results_per_study, 'num_runs': 1}
    else:
        for column in results_per_study.columns:
            if column == 'study':
                continue
            results[key]['results_per_study'][column] += results_per_study[column]
        results[key]['num_runs'] += 1

for k, v in results.items():
    print(f'{k}:')
    num_runs = v['num_runs']
    results_per_study = v['results_per_study']
    for column in results_per_study.columns:
        if column == 'study':
            continue
        results_per_study[column] /= num_runs
    print(f'{num_runs=}')
    print(results_per_study)
