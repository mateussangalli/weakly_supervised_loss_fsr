import numpy as np

from .crop_selection import select_crop


def crop_generator(
    data, crop_size, crops_per_image, scale_range,
    yield_label=True, normalize=True
):
    """
    Takes a list of (image, label) pairs and yields crops of the images.
    Parameters:
        data (List[Tuple[np.ndarray, np.ndarray]]): a list of (image, label)
            pairs, where image is a float32 or integer array of rank 3 and
            label is an integer array of rank 2.
        crop_size (int or Tuple[int, int]): height and width of the cropped
            region.
            if integer, cropped region is a square.
        crops_per_image (int): how many crops per image.
        scale_range (Tuple[float, float]): minimum and maximum value for res-
            scaling cropped region.
        yield_label (bool): where to yield only the image or the label too.
        normalize (bool): whether to normalize the image from [0..255] valued
            to [0., 1.] valued.
    Yields:
        im_crop (np.ndarray): cropped image
        gt_crop (np.ndarray): optional. Cropped label.
    """
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
