import os
import matplotlib.pyplot as plt
from utils.data_loading import read_dataset
from utils.data_generation import crop_generator

DATA_ROOT = '../prp_loreal_data'

train_images = os.listdir(os.path.join(DATA_ROOT, "train", "images"))
train_images = [train_images[3], train_images[4], train_images[5]]
data = read_dataset(DATA_ROOT, 'train', train_images)

gen = crop_generator(data, 128, 3, (0.5, 2.))

for i, (im_crop, gt_crop) in enumerate(gen):
    plt.subplot(121)
    plt.imshow(im_crop)
    plt.subplot(122)
    plt.imshow(gt_crop)
    plt.show()
    if i > 5:
        break
