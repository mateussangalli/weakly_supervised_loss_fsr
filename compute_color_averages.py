import numpy as np

from utils.data_loading import read_dataset


DATA_ROOT = '~/weak_supervision_data'
WHITE_THRESHOLD_MIN = .96
WHITE_THRESHOLD_MAX = .99


def smoothstep(value, tmin, tmax):
    return np.minimum(np.maximum((value - tmin) / (tmax - tmin), 0.), 1.)


def compute_mean(image):
    mask = smoothstep(np.min(image, -1, keepdims=True),
                      WHITE_THRESHOLD_MIN,
                      WHITE_THRESHOLD_MAX)
    mask = 1. - mask
    out = np.sum(mask * image, 1)
    out = np.sum(out, 0) / (np.sum(mask) + 1e-10)
    return out


dataset = read_dataset(DATA_ROOT, 'train')
images = [x.astype(np.float32) / 255. for x, _ in dataset]

means = np.stack(list(map(compute_mean, images)), 0)
print(means)

np.save('means_training_set.npy', means)
