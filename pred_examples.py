import argparse
import os

import numpy as np
from keras.models import load_model
from keras.losses import CategoricalCrossentropy
import keras
from skimage.io import imsave

from utils.data_loading import read_dataset
from utils.utils import crop_to_multiple_of
from utils.jaccard_loss import OneHotMeanIoU
from utils.combined_loss import CombinedLoss
from utils.directional_relations import PRPDirectionalPenalty

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("run_id", type=str)
parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
parser.add_argument("--runs_dir", type=str, default="labeled_runs")

args = parser.parse_args()


run_dir = os.path.join(args.runs_dir, args.run_id)

# load data
val_images = os.listdir(os.path.join(args.data_root, "val", "images"))
val_images = [val_images[3], val_images[4], val_images[5]]

data_val = read_dataset(args.data_root, "val", val_images)
data_val = [
    (crop_to_multiple_of(im, 32),
     crop_to_multiple_of(gt, 32)) for (im, gt) in data_val
]


def gen_val():
    for image, label in data_val:
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

pred_dir = os.path.join(run_dir, 'predictions')
os.mkdirs(pred_dir)
for i, (im, _) in enumerate(data_val):
    im = im.astype(np.float32) / 255.0
    imsave(os.path.join(pred_dir, f'pred{i}.png'), im)
