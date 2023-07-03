import argparse
import os
import pandas as pd


from utils.data_loading import read_label, read_dataset
from utils.jaccard_loss import mean_iou
from utils.utils import one_hot

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("run_id", type=str)
parser.add_argument("--data_root", type=str, default="../prp_loreal_data")
parser.add_argument("--runs_dir", type=str, default="labeled_runs")
parser.add_argument("--subset", type=str, default="val")
args = parser.parse_args()


run_dir = os.path.join(args.runs_dir, args.run_id)

# load data
filenames = os.listdir(os.path.join(args.data_root, args.subset, "images"))

data = read_dataset(args.data_root, args.subset, filenames, min_size=(0, 0))

results = list()
for filename, (im, gt) in zip(filenames, data):
    pred_path = os.path.join(args.data_root, args.subset, "predictions", filename)
    pred = read_label(pred_path)
    post_path = os.path.join(args.data_root, args.subset, "postproc", filename)
    post = read_label(post_path)
    gt = one_hot(gt)
    miou_pred = mean_iou(gt, pred)
    miou_post = mean_iou(gt, post)
    tmp = {'image': filename, 'mean_iou_pred': miou_pred, 'mean_iou_post': miou_post}
    results.append(tmp)

df = pd.DataFrame(results)
df.to_csv(os.path.join(args.data_root, args.subset, 'results.csv'))

pd.DataFrame(df.mean()).to_csv(os.path.join(args.data_root, args.subset, 'average_results.csv'))
