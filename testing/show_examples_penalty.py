import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from utils.directional_relations import PRPDirectionalPenalty
from utils.data_loading import read_label


preds = [
    read_label('testing/image_name', one_hot=True),
    read_label('testing/image_name', one_hot=True),
    read_label('testing/image_name', one_hot=True),
]

def label2grey(label):
    label = np.argmax(label, -1)
    colors = np.array([.5, 0, 1.], np.float32) 
    colors = np.stack([colors]*3, 1)
    return colors[label]

colors = [(0., 0., 0., 0.), (0.9, 0.4, 0., 1.)]
cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors)
 
fn1 = PRPDirectionalPenalty(20, 1, return_map=True, dilation_type='maxplus')
fn2 = PRPDirectionalPenalty(20, 1, return_map=True, dilation_type='flat_line')

for i, pred in enumerate(preds):
    penalty = np.array(fn1(pred[np.newaxis, ...]))[0, ...]
    pred = label2grey(pred)

    ax = plt.subplot(2, 3, i+1)
    ax.imshow(pred, alpha=1.)
    ax.imshow(penalty, cmap=cmap, vmin=0., vmax=1., alpha=.7)

for i, pred in enumerate(preds):
    penalty = np.array(fn2(pred[np.newaxis, ...]))[0, ...]
    pred = label2grey(pred)

    ax = plt.subplot(2, 3, i+4)
    ax.imshow(pred, alpha=1.)
    ax.imshow(penalty, cmap=cmap, vmin=0., vmax=1., alpha=.7)
plt.show()
