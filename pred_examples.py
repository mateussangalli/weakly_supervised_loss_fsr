import argparse
import os

import keras
import numpy as np
from keras.losses import CategoricalCrossentropy
from keras.models import load_model
from skimage.io import imsave

from utils.combined_loss import CombinedLoss
from utils.data_loading import read_dataset
from utils.directional_relations import PRPDirectionalPenalty
from utils.jaccard_loss import OneHotMeanIoU
from utils.utils import crop_to_multiple_of

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("run_id", type=str)
parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
parser.add_argument("--runs_dir", type=str, default="labeled_runs")
parser.add_argument("--subset", type=str, default="val")
args = parser.parse_args()


run_dir = os.path.join(args.runs_dir, args.run_id)

# load data
images = os.listdir(os.path.join(args.data_root, args.subset, "images"))

data = read_dataset(args.data_root, args.subset, images)
data = [
    (crop_to_multiple_of(im, 32),
     crop_to_multiple_of(gt, 32)) for (im, gt) in data
]


def gen_val():
    for image, label in data:
        image = image.astype(np.float32) / 255.0
        yield image, label


directional_loss = PRPDirectionalPenalty(3, 2, 5)


def directional_loss_metric(y, y_pred, **kwargs):
    return directional_loss(y_pred)


crossentropy = CategoricalCrossentropy(from_logits=False)


def crossentropy_metric(y_true, y_pred, **kwargs):
    return crossentropy(y_true, y_pred)


loss_fn = CombinedLoss(
    CategoricalCrossentropy(from_logits=False),
    PRPDirectionalPenalty(3, 2, 5),
    50,
    0.,
)

# create run dir
custom_objects = {'OneHotMeanIoU': OneHotMeanIoU(3),
                  'directional_loss_metric': directional_loss_metric,
                  'crossentropy_metric': crossentropy_metric,
                  'CombinedLoss': loss_fn}
model = load_model(os.path.join(run_dir, 'saved_model'),
                   custom_objects=custom_objects,
                   compile=False)

pred_dir = os.path.join(run_dir, 'predictions', args.subset)
os.makedirs(pred_dir, exist_ok=True)
for i, (im, _) in enumerate(data):
    im = im.astype(np.float32) / 255.0
    pred = model(im[np.newaxis, ...])
    pred = np.array(pred)[0, ...]
    pred = (pred * 255.).astype(np.uint8)
    imsave(os.path.join(pred_dir, f'pred{i}.png'), pred)
