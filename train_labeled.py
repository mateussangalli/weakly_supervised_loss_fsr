import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger, LearningRateScheduler
from keras.losses import CategoricalCrossentropy

from utils.combined_loss import CombinedLoss
from utils.data_generation import get_tf_train_dataset, get_tf_val_dataset
from utils.data_loading import read_dataset
from utils.directional_relations import PRPDirectionalPenalty
from utils.jaccard_loss import OneHotMeanIoU, jaccard_loss_mean_wrapper
from utils.unet import UNetBuilder
from utils.utils import crop_to_multiple_of
from labeled_images import LABELED_IMAGES, LABELED_IMAGES_VAL


def parse_args():
    parser = argparse.ArgumentParser()
    # data dir arguments
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
    parser.add_argument("--runs_dir", type=str, default="labeled_runs")

    # training arguments
    parser.add_argument("--num_images_labeled", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--starting_lr", type=float, default=1e-4)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay_rate", type=float, default=0.003)
    parser.add_argument("--rotation_angle", type=float, default=np.pi / 8.0)
    parser.add_argument("--loss_fn", choices=['iou', 'crossentropy'], default='crossentropy')
    parser.add_argument("--hue_jitter", type=float, default=0.)
    parser.add_argument("--sat_jitter", type=float, default=0.)
    parser.add_argument("--val_jitter", type=float, default=0.)

    # crop generator arguments
    parser.add_argument("--crop_size", type=int, default=160)
    parser.add_argument("--crops_per_image", type=int, default=16)
    parser.add_argument("--min_scale", type=float, default=-1.0)
    parser.add_argument("--max_scale", type=float, default=1.3)

    # architecture arguments
    parser.add_argument("--filters_start", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--bn_momentum", type=float, default=0.85)

    # loss function arguments
    parser.add_argument("--max_weight", type=float, default=1.0)
    parser.add_argument("--increase_epochs", type=int, default=50)
    parser.add_argument("--strel_size", type=int, default=3)
    parser.add_argument("--strel_spread", type=int, default=2)
    parser.add_argument("--strel_iterations", type=int, default=10)

    # verbose
    parser.add_argument("--verbose", type=int, default=2)

    args = parser.parse_args()
    return args

    return ds_train, ds_val


def get_model(args):
    # create model
    model = UNetBuilder(
        (None, None, 3),
        args.filters_start,
        args.depth,
        normalization="batch",
        normalize_all=False,
        batch_norm_momentum=args.bn_momentum,
    ).build()
    directional_loss = PRPDirectionalPenalty(
        args.strel_size, args.strel_spread, args.strel_iterations
    )

    def directional_loss_metric(y, y_pred, **kwargs):
        return directional_loss(y_pred)


    if args.loss_fn == 'iou':
        loss_labeled = jaccard_loss_mean_wrapper()
    else:
        loss_labeled = CategoricalCrossentropy(from_logits=False)

    def loss_labeled_metric(y_true, y_pred, **kwargs):
        return loss_labeled(y_true, y_pred)

    loss_fn = CombinedLoss(
        loss_labeled,
        PRPDirectionalPenalty(
            args.strel_size, args.strel_spread, args.strel_iterations
        ),
        args.increase_epochs,
        args.max_weight,
    )

    model.compile(
        optimizer=tf.keras.optimizers.experimental.Adam(
            args.starting_lr, weight_decay=args.weight_decay
        ),
        loss=loss_fn,
        metrics=[
            OneHotMeanIoU(3),
            loss_labeled_metric,
            directional_loss_metric,
        ],
    )

    return model, loss_fn


if __name__ == "__main__":
    args = parse_args()

    # load data
    data_train = read_dataset(args.data_root, "train", LABELED_IMAGES[:args.num_images_labeled])
    data_val = read_dataset(args.data_root, "val", LABELED_IMAGES_VAL)
    data_val = [
        (
            crop_to_multiple_of(im, 2**args.depth),
            crop_to_multiple_of(gt, 2**args.depth),
        )
        for (im, gt) in data_val
    ]

    samples_per_epoch = len(data_train) * args.crops_per_image
    steps_per_epoch = int(np.ceil(samples_per_epoch / args.batch_size))

    ds_train = get_tf_train_dataset(data_train, vars(args))
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_val = get_tf_val_dataset(data_val)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    if args.run_id == "":
        weight_str = f"{args.max_weight:.4f}".replace(".", "p")
        run_name = f'weight{weight_str}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    else:
        run_name = args.run_id
    run_dir = os.path.join(args.runs_dir, run_name)

    # create run dir
    os.makedirs(run_dir, exist_ok=True)
    params_path = os.path.join(run_dir, "params.json")
    args_dict = vars(args)
    with open(params_path, "w") as fp:
        json.dump(args_dict, fp)

    def schedule(epoch, lr):
        if epoch > 0:
            return lr * tf.exp(-args.lr_decay_rate)
        return lr

    model, loss_fn = get_model(args)

    model.fit(
        ds_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val,
        validation_steps=len(data_val),
        epochs=args.epochs,
        callbacks=[
            loss_fn.callback,
            CSVLogger(os.path.join(run_dir, "training_history.csv")),
            LearningRateScheduler(schedule),
        ],
        verbose=args.verbose,
        validation_freq=args.val_freq,
    )

    model.save(os.path.join(run_dir, "saved_model"))
