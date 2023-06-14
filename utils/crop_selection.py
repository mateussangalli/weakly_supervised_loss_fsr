import numpy as np
from skimage.transform import resize


def _is_valid_crop(gt_crop):
    return max(np.sum((gt_crop == 0).astype(np.int32)), np.sum((gt_crop == 1).astype(np.int32)),
               np.sum((gt_crop == 2).astype(np.int32))) < (gt_crop.shape[0] * gt_crop.shape[1])


def _is_valid_crop_full(gt_crop):
    """
    stronger definition that requires all three classes to be present and the
    background to be present in the top and bottom of the crop
    """
    if np.sum((gt_crop == 0).astype(np.int32)) == 0:
        return False
    if np.sum((gt_crop == 2).astype(np.int32)) == 0:
        return False
    if 1 not in gt_crop[0, :] or 1 not in gt_crop[-1, :]:
        return False
    return True

def _is_valid_point(p, shape):
    return (0 <= p[0] < shape[0]) and (0 <= p[1] <= shape[1])


def _transform_point(p, center, angle, scale, v):
    p2 = p - center
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    p2 = rotation_matrix.dot(p2) * scale + v
    p2 += center
    return p2


def _crop(im, gt, size):
    """
    Crops a square of specified size from an image and label pair
    Args:
        im: image to be cropped, 3-dimensional numpy ndarray
        gt: label corresponding to image, 2-dimensional numpy ndarray
        size: integer, size of the square cropped region
    Returns:
        im_out: cropped square from image
        gt_out: cropped square from label

    """
    top = np.random.randint(0, im.shape[0] - size)
    left = np.random.randint(0, im.shape[1] - size)
    bottom = top + size
    right = left + size

    im_out = im[top:bottom, left:right, :]
    gt_out = gt[top:bottom, left:right]
    if gt_out.shape[0] != size or gt_out.shape[1] != size:
        raise ValueError(f'Invalid crop shape for labels. Label crop shape: {gt_out.shape}, Label shape: {gt.shape}, '
                         f'Image shape: {im.shape}')
    return im_out, gt_out


def _pad(im, gt, size):
    """
    Pads image and ground truth until they have size at least size x size
    Args:
        im: image to be padded, 3-dimensional ndarray
        gt: label to be padded, 2-dimensional ndarray
        size: integer, minimum size of the output

    Returns:
        im_pad: padded image, array of the same type as im
        gt_pad: padded label, array of the same type as gt

    """
    pad00 = max((size - im.shape[0]) // 2, 0)
    pad01 = max((size - im.shape[0]) - pad00, 0)
    pad10 = max((size - im.shape[1]) // 2, 0)
    pad11 = max((size - im.shape[1]) - pad00, 0)
    im_pad = np.pad(im, [[pad00, pad01], [pad10, pad11], [0, 0]])
    gt_pad = np.pad(gt, [[pad00, pad01], [pad10, pad11]], constant_values=1)
    return im_pad, gt_pad


def _crop_pad(im, gt, size):
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
    im_tmp, gt_tmp = _crop(im, gt, size)
    return _pad(im_tmp, gt_tmp, size)


def _crop_scaleaug(im, gt, crop_size, max_scale=1.):
    """
    crops region of image with scale augmentation
    scale is sampled from log-uniform distribution between 1 / max_scale and max_scale
    Args:
        im: array with shape H x W x c to be cropped
        gt: array with shape H x W x 1 to be cropped in the same way as the im
        crop_size: size of the cropped region
        max_scale: max zoom in/out in the cropped region
    Returns:
        im_crop: crop_size x crop_size x c array, same type as im
        gt_crop : crop_size x crop_size x 1 array, same type as gt
    """
    min_scale = 1 / max_scale
    scale = np.exp(np.random.uniform(np.log(min_scale), np.log(max_scale)))
    s = int(np.rint(scale * crop_size))
    s = min(s, im.shape[0] - 1, im.shape[1] - 1)
    im_cropped, gt_cropped = _crop_pad(im, gt, s)

    return im_cropped, gt_cropped


def select_crop(im, gt, config, aug=False):
    """
    Randomly crops an image and gt pair
    Args:
        im: image to be cropped, ndarray with shape [H, W, c]
        gt: label to be cropped, ndarray with shape [H, W]
        config: dict object to be passed as keyword arguments to the cropping function.
                must contain key 'crop_size' with integer value and if aug==True a key 'max_scale' with float value
        aug: whether to also change the scale of the crop randomly
    Returns:
        im_crop: cropped square from image
        gt_crop: cropped square from label


    """
    valid = False
    while not valid:
        if aug:
            im_crop, gt_crop = _crop_scaleaug(im, gt, **config)
        else:
            im_crop, gt_crop = _crop_pad(im, gt, config['crop_size'])
        valid = _is_valid_crop(gt_crop)
    return im_crop, gt_crop


def select_crop_full(im, gt, config, aug=False, give_up=10):
    """
    Randomly crops an image and gt pair
    All three classes must be present in the crop and the background must be present
    above and below the other two
    Args:
        im: image to be cropped, ndarray with shape [H, W, c]
        gt: label to be cropped, ndarray with shape [H, W]
        config: dict object to be passed as keyword arguments to the cropping function.
                must contain key 'crop_size' with integer value and if aug==True a key 'max_scale' with float value
        aug: whether to also change the scale of the crop randomly
    Returns:
        im_crop: cropped square from image
        gt_crop: cropped square from label


    """
    valid = False
    tries = 0
    while not valid:
        if aug:
            im_crop, gt_crop = _crop_scaleaug(im, gt, **config)
        else:
            im_crop, gt_crop = _crop_pad(im, gt, config['crop_size'])
        valid = _is_valid_crop_full(gt_crop)
        tries += 1
        if tries > give_up:
            return select_crop(im, gt, config, aug)
    return im_crop, gt_crop


def crop_to_multiple_of(image, k):
    shape = image.shape
    crop1 = shape[0] % k
    crop2 = shape[1] % k
    return image[:shape[0] - crop1, :shape[1] - crop2, ...]
