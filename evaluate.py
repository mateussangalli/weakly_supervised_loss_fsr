import argparse
import os
import pandas as pd
import numpy as np
from skimage.io import imread

from utils.data_loading import read_label, read_dataset
from utils.jaccard_loss import mean_iou
from utils.utils import one_hot
from utils.directional_relations import PRPDirectionalPenalty

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("run_id", type=str)
parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
parser.add_argument("--runs_dir", type=str, default="labeled_runs")
parser.add_argument("--subset", type=str, default="val")
args = parser.parse_args()


run_dir = os.path.join(args.runs_dir, args.run_id)

prp_penalty = PRPDirectionalPenalty(20, 1)

# load data
filenames = os.listdir(os.path.join(args.data_root, args.subset, "images"))

data = read_dataset(args.data_root, args.subset, filenames, min_size=(0, 0))

proba_dir = os.path.join(run_dir, 'probability_maps', args.subset)
pred_dir = os.path.join(run_dir, 'predictions', args.subset)
post_dir = os.path.join(run_dir, 'postproc', args.subset)
results = list()
for filename, (im, gt) in zip(filenames, data):
    proba_path = os.path.join(proba_dir, filename)
    proba = imread(proba_path)
    dir_penalty = np.array(prp_penalty(proba[np.newaxis, ...]))[0]
    pred_path = os.path.join(pred_dir, filename)
    pred = read_label(pred_path, one_hot=True)
    post_path = os.path.join(post_dir, filename)
    post = read_label(post_path, one_hot=True)
    gt = one_hot(gt)
    miou_pred = mean_iou(gt, pred)
    miou_post = mean_iou(gt, post)
    tmp = {'image': filename, 'mean_iou_pred': miou_pred, 'mean_iou_post': miou_post, 'dir_penalty': dir_penalty}
    results.append(tmp)

df = pd.DataFrame(results)
df.to_csv(os.path.join(run_dir, f'results_{args.subset}.csv'))

average_results = dict()
for column in df:
    average_results[column] = [df[column].mean()]
df_average = pd.DataFrame(average_results)
df_average.to_csv(os.path.join(run_dir, f'average_results_{args.subset}.csv'))
