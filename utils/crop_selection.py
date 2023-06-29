import numpy as np
from skimage.transform import resize

TOL_WHITE = 0.05
EPS_WHITE = 0.01


def _is_valid_crop(im_crop):
    nonwhite = np.abs(np.max(im_crop, (0, 1), keepdims=True) - im_crop) > TOL_WHITE
    nonwhite = nonwhite.max(-1).astype(np.float32)
    return nonwhite.mean() > EPS_WHITE


def random_crop(im, gt=None, size=None):
    """
    Crops a square of specified size from an image and label pair
    Args:
        im: image to be cropped, 3-dimensional numpy ndarray
        gt: label corresponding to image, 2-dimensional numpy ndarray
        size: tuple of integer, size of the rectangular cropped region
    Returns:
        im_out: cropped square from image
        gt_out: cropped square from label

    """
    top = np.random.randint(0, max(im.shape[0] - size[0], 1))
    left = np.random.randint(0, max(im.shape[1] - size[1], 1))
    bottom = top + size[0]
    right = left + size[1]

    im_out = im[top:bottom, left:right, :]
    if gt is None:
        return im_out
    gt_out = gt[top:bottom, left:right]
    if gt_out.shape[0] != size[0] or gt_out.shape[1] != size[1]:
        raise ValueError(
            f"Invalid crop shape for labels. Label crop shape: {gt_out.shape}, Label shape: {gt.shape}, "
            f"Image shape: {im.shape}"
        )
    return im_out, gt_out


def pad(im, gt=None, size=None):
    """
    Pads image and ground truth until they have size at least size x size
    Args:
        im: image to be padded, 3-dimensional ndarray
        gt: label to be padded, 2-dimensional ndarray
        size: tuple of integer, minimum size of the output

    Returns:
        im_pad: padded image, array of the same type as im
        gt_pad: padded label, array of the same type as gt

    """
    pad00 = max((size[0] - im.shape[0]) // 2, 0)
    pad01 = max((size[0] - im.shape[0]) - pad00, 0)
    pad10 = max((size[1] - im.shape[1]) // 2, 0)
    pad11 = max((size[1] - im.shape[1]) - pad00, 0)
    im_pad = np.pad(im, [[pad00, pad01], [pad10, pad11], [0, 0]], mode="reflect")
    if gt is None:
        return im_pad
    gt_pad = np.pad(gt, [[pad00, pad01], [pad10, pad11]], mode="reflect")
    return im_pad, gt_pad


def random_crop_pad(im, gt, size):
    """
    Crops or pads size of image/label pairs until they are a square of side = size
    Args:
        im: image to be padded, 3-dimensional ndarray
        gt: label to be padded, 2-dimensional ndarray
        size: integer, size of the output

    Returns:
        im_out: cropped/padded square from image
        gt_out: cropped/padded square from label

    """
    im_tmp, gt_tmp = pad(im, gt, size)
    return random_crop(im_tmp, gt_tmp, size)


def select_crop(im, gt, size):
    """
    Randomly crops an image and gt pair
    Args:
        im: image to be cropped, ndarray with shape [H, W, c]
        gt: label to be cropped, ndarray with shape [H, W]
        size: integer or tuple of integer, size of the crop
    Returns:
        im_crop: cropped square from image
        gt_crop: cropped square from label
    """
    if isinstance(size, int):
        size = (size, size)
    valid = False
    while not valid:
        im_crop, gt_crop = random_crop_pad(im, gt, size)
        valid = _is_valid_crop(im_crop)
    return im_crop, gt_crop
