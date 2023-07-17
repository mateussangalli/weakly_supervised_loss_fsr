from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

from compute_color_averages import WHITE_THRESHOLD_MIN, WHITE_THRESHOLD_MAX, smoothstep


def transfer_mean(im_tgt, im_src):
    mask_tgt = smoothstep(np.min(im_tgt, -1, keepdims=True),
                          WHITE_THRESHOLD_MIN,
                          WHITE_THRESHOLD_MAX)
    mask_tgt = 1. - mask_tgt
    mean_tgt = np.sum(mask_tgt * im_tgt, 1)
    mean_tgt = np.sum(mean_tgt, 0) / (np.sum(mask_tgt) + 1e-10)
    mean_tgt = mean_tgt[np.newaxis, np.newaxis, :]

    mask_src = smoothstep(np.min(im_src, -1, keepdims=True),
                          WHITE_THRESHOLD_MIN,
                          WHITE_THRESHOLD_MAX)
    mask_src = 1. - mask_src
    mean_src = np.sum(mask_src * im_src, 1)
    mean_src = np.sum(mean_src, 0) / (np.sum(mask_src) + 1e-10)
    mean_src = mean_src[np.newaxis, np.newaxis, :]

    im_out = im_src * (1. - mask_src) + (im_src - mean_src + mean_tgt) * mask_src
    return im_out


im_tgt = imread('testing/image_name')
im_tgt = im_tgt.astype(np.float32) / 255.
im_src = imread('testing/image_name')
im_src = im_src.astype(np.float32) / 255.

im_out = transfer_mean(im_tgt, im_src)

plt.figure()
plt.subplot(131)
plt.imshow(im_src)
plt.subplot(132)
plt.imshow(im_tgt)
plt.subplot(133)
plt.imshow(im_out)
plt.show()

im_out = np.minimum(np.maximum(im_out, 0.), 1.)
im_out = (im_out * 255.).astype(np.uint8)

imsave('transferred_example.png', im_out)
