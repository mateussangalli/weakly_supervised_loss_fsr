import matplotlib.pyplot as plt

import numpy as np
from skimage.io import imsave
from utils.directional_relations import LeftOf

r = 30
size = r * 2 + 1
sqr_side = 10
sqr_r = sqr_side // 2
im = np.zeros([size, size], np.float32)
im[r-sqr_r:r+sqr_r, r-sqr_r:r+sqr_r] = 1.
im = im[np.newaxis, :, :, np.newaxis]
layer_maxplus = LeftOf(30, 1, dilation_type='maxplus')
layer_product = LeftOf(30, 1, dilation_type='product')
layer_minimum = LeftOf(30, 1, dilation_type='minimum')
out_p = layer_product(im) + im
out_min = layer_minimum(im) + im
out_l = layer_maxplus(im) + im
out_p = out_p[0, :, :, 0]
out_min = out_min[0, :, :, 0]
out_l = out_l[0, :, :, 0]
plt.subplot(131)
plt.imshow(out_p)
plt.subplot(132)
plt.imshow(out_min)
plt.subplot(133)
plt.imshow(out_l)
plt.show()

out_p = np.array(out_p)
out_p = (255. * out_p).astype(np.uint8)
out_l = np.array(out_l)
out_l = (255. * out_l).astype(np.uint8)
out_min = np.array(out_min)
out_min = (255. * out_min).astype(np.uint8)

imsave('left_of_product.png', out_p)
imsave('left_of_minimum.png', out_min)
imsave('left_of_maxplus.png', out_l)
