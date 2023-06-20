import numpy as np
import matplotlib.pyplot as plt
from utils.directional_relations import Below, Above

if __name__ == '__main__':

    im = np.zeros([31, 31], np.float32)
    im[15, 15] = 1.
    im = im[np.newaxis, :, :, np.newaxis]

    layer = Below(5, 5, 6)
    out = layer(im)[0, :, :, 0]
    plt.subplot(131)
    plt.imshow(im[0, :, :, 0])
    plt.subplot(132)
    plt.imshow(out)
    plt.subplot(133)
    plt.imshow(layer.kernel[:, :, 0])
    plt.show()

    layer = Above(5, 5, 6)
    out = layer(im)[0, :, :, 0]
    plt.subplot(131)
    plt.imshow(im[0, :, :, 0])
    plt.subplot(132)
    plt.imshow(out)
    plt.subplot(133)
    plt.imshow(layer.kernel[:, :, 0])
    plt.show()
