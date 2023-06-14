import tensorflow as tf
from keras.metrics import kullback_leibler_divergence


def class_line_probability(softmax_labels):
    return tf.reduce_mean(softmax_labels, 2)


def horizontal_kl_loss(y_true, y_pred):
    lines_true = class_line_probability(y_true)
    lines_pred = class_line_probability(y_pred)
    return kullback_leibler_divergence(lines_true, lines_pred)


class CombinedLosses:
    """
    Combines two loss functions loss1 and loss2 according to
    loss(y_true, y_pred) = loss1(y_true, y_pred) * (1 - alpha) +  loss1(y_true, y_pred) * alpha
    """

    def __init__(self, loss1, loss2, alpha):
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        return self.loss1(y_true, y_pred) * (1 - self.alpha) + self.loss2(y_true, y_pred) * self.alpha

    def get_config(self):
        return {'loss1': self.loss1, 'loss2': self.loss2, 'alpha': self.alpha}


if __name__ == '__main__':
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread

    workdir = 'sweeps_runs/unet_2023-02-17_13-51-09/predictions-000/images/'
    image_name1 = 'F14_14_3a_PRP_FM_20220916_143542_X001_Y001_A001_Sub02'
    image_name2 = 'F21_07_04_PRP_FM_20220916_150557_X001_Y001_A016_Sub01'

    gt1 = imread(os.path.join(workdir, image_name1 + '_gt.png')).astype(np.float32) / 255.
    pred1 = imread(os.path.join(workdir, image_name1 + '_proba.png')).astype(np.float32) / 255.
    loss1 = float(horizontal_kl_loss(gt1, pred1))
    lines_true1 = class_line_probability(gt1[np.newaxis, ...])
    lines_pred1 = class_line_probability(pred1[np.newaxis, ...])
    print(loss1)

    gt2 = imread(os.path.join(workdir, image_name2 + '_gt.png')).astype(np.float32) / 255.
    pred2 = imread(os.path.join(workdir, image_name2 + '_proba.png')).astype(np.float32) / 255.
    loss2 = float(horizontal_kl_loss(gt2, pred2))
    lines_true2 = class_line_probability(gt2[np.newaxis, ...])
    lines_pred2 = class_line_probability(pred2[np.newaxis, ...])
    print(loss2)

    plt.subplot(221)
    plt.title('ground truth')
    plt.imshow(gt1)
    plt.subplot(222)
    plt.title(f'prediction, \n KL={loss1}')
    plt.imshow(pred1)

    plt.subplot(223)
    plt.title('ground truth')
    plt.imshow(gt2)
    plt.subplot(224)
    plt.title(f'prediction, \n KL={loss2}')
    plt.imshow(pred2)

    plt.savefig('../../KLLoss_test_images.pdf', dpi=300, bbox_inches='tight')
    plt.show()


    def plot_lines_probs(lines):
        lines = np.array(lines)[0, ...]
        lines = lines[:, [1, 0, 2]]
        lines = np.cumsum(lines, 1)
        x = np.linspace(1., 0., lines.shape[0])
        plt.fill_between(x, np.zeros_like(x), lines[:, 0], color='green', label='BG')
        plt.fill_between(x, lines[:, 0], lines[:, 1], color='red', label='SC')
        plt.fill_between(x, lines[:, 1], lines[:, 2], color='blue', label='LE')
        plt.xlabel('height')
        plt.legend()


    plt.figure()
    plt.subplot(121)
    plt.title('prob. of class per line, GT')
    plot_lines_probs(lines_true1)
    plt.subplot(122)
    plt.title('prediction')
    plot_lines_probs(lines_pred1)

    plt.savefig('../../KLLoss_test_lines_im1.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.title('prob. of class per line, GT')
    plot_lines_probs(lines_true2)
    plt.subplot(122)
    plt.title('prediction')
    plot_lines_probs(lines_pred2)

    plt.savefig('../../KLLoss_test_lines_im2.pdf', dpi=300, bbox_inches='tight')
    plt.show()
