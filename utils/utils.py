import keras
import numpy as np
from keras.losses import CategoricalCrossentropy

from .combined_loss import CombinedLoss
from .directional_relations import PRPDirectionalPenalty
from .jaccard_loss import OneHotMeanIoU
from .size_regularization import QuadraticPenaltyHeight
from .unlabeled_training import SemiSupModel, SemiSupModelPseudo


def crop(im, size, start=(0, 0)):
    top = 0
    left = 0
    bottom = size[0]
    right = size[1]

    im_out = im[top:bottom, left:right, ...]
    return im_out


def pad(im, size):
    pad00 = max((size[0] - im.shape[0]) // 2, 0)
    pad01 = max((size[0] - im.shape[0]) - pad00, 0)
    pad10 = max((size[1] - im.shape[1]) // 2, 0)
    pad11 = max((size[1] - im.shape[1]) - pad00, 0)
    if len(im.shape) == 3:
        im_pad = np.pad(
            im, [[pad00, pad01], [pad10, pad11], [0, 0]], mode='reflect')
    else:
        im_pad = np.pad(im, [[pad00, pad01], [pad10, pad11]], mode='reflect')
    return im_pad


def crop_pad(im, size, start=(0, 0)):
    im_tmp, gt_tmp = crop(im, size, start)
    return pad(im_tmp, gt_tmp, size)


def one_hot(arr, num_classes='auto'):
    if num_classes == 'auto':
        num_classes = np.max(arr) + 1
    eye = np.eye(num_classes)
    return eye[arr]


def crop_to_multiple_of(image, k):
    shape = image.shape
    crop1 = shape[0] % k
    crop2 = shape[1] % k
    return image[:shape[0] - crop1, :shape[1] - crop2, ...]


def pad_to_multiple_of(image, k):
    shape = image.shape
    pad1 = k - shape[0] % k
    pad2 = k - shape[1] % k
    if len(shape) == 3:
        return np.pad(image, [(0, pad1), (0, pad2), (0, 0)], mode='reflect'), (pad1, pad2)
    elif len(shape) == 2:
        return np.pad(image, [(0, pad1), (0, pad2)], mode='reflect'), (pad1, pad2)
    else:
        raise ValueError('image should have rank 2 or 3')


def load_model(path, size=3, spread=2, iterations=5):
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

    custom_objects = {
        'OneHotMeanIoU': OneHotMeanIoU(3),
        'directional_loss_metric': directional_loss_metric,
        'crossentropy_metric': crossentropy_metric,
        'CombinedLoss': loss_fn,
        'SemiSupModel': SemiSupModel,
        'SemiSupModelPseudo': SemiSupModelPseudo,
        'QuadraticPenaltyHeight': QuadraticPenaltyHeight
    }
    return keras.models.load_model(path,
                                   custom_objects=custom_objects,
                                   compile=False)
