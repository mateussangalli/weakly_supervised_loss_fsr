import argparse
import os

import numpy as np
from skimage.io import imsave

from utils.data_loading import read_dataset, save_label
from utils.im_tools import TwoLayers
from utils.utils import pad_to_multiple_of, load_model

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("run_id", type=str)
parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
parser.add_argument("--runs_dir", type=str, default="labeled_runs")
parser.add_argument("--subset", type=str, default="val")
args = parser.parse_args()


run_dir = os.path.join(args.runs_dir, args.run_id)

# load data
filenames = os.listdir(os.path.join(args.data_root, args.subset, "images"))

data = read_dataset(args.data_root, args.subset, filenames, min_size=(0, 0))


def gen_val():
    for image, label in data:
        image = image.astype(np.float32) / 255.0
        yield image, label


model = load_model(os.path.join(run_dir, 'saved_model'))

postproc_func = TwoLayers(3, ordered=True)


def postproc(pred):
    pred = np.array([2, 1, 3])[pred]
    pred = postproc_func(pred)
    pred = np.array([1, 0, 2])[pred-1]
    return pred


proba_dir = os.path.join(run_dir, 'probability_maps', args.subset)
pred_dir = os.path.join(run_dir, 'predictions', args.subset)
post_dir = os.path.join(run_dir, 'postproc', args.subset)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(proba_dir, exist_ok=True)
os.makedirs(post_dir, exist_ok=True)
for i, ((im, _), name) in enumerate(zip(data, filenames)):
    im, (pad1, pad2) = pad_to_multiple_of(im, 32)
    im = im.astype(np.float32) / 255.0
    proba = model(im[np.newaxis, ...])
    proba = np.array(proba)[0, :-pad1, :-pad2, ...]
    proba = (proba * 255.).astype(np.uint8)
    pred = np.argmax(proba, -1)
    post = postproc(pred)
    imsave(os.path.join(proba_dir, name), proba)
    save_label(pred, os.path.join(pred_dir, name))
    save_label(post, os.path.join(post_dir, name))
