import numpy as np

from .crop_selection import select_crop


def crop_generator(
    data, crop_size, crops_per_image, scale_range,
    yield_label=True, normalize=True
):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    for image, label in data:
        if normalize:
            image = image.astype(np.float32) / 255.0
        for i in range(crops_per_image):
            s = np.random.uniform(low=np.log(scale_range[0]),
                                  high=np.log(scale_range[1]))
            s = np.exp(s)
            new_size = (int(crop_size[0] * s), int(crop_size[1] * s))
            im_crop, gt_crop = select_crop(image, label, new_size)
            if yield_label:
                yield im_crop, gt_crop
            else:
                yield im_crop
