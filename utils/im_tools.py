"""
"""
from __future__ import print_function

import os

import numpy as np
import scipy.misc as sm
import skimage.morphology as skm
from scipy.ndimage import (binary_dilation, distance_transform_cdt)
from scipy.ndimage.morphology import distance_transform_edt
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.segmentation import watershed

SE4 = [1, 3, 1]


def flip_axis(x, axis):
    """Flip a given axis.
    Borrowed from keras."""
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def s1(x):
    return flip_axis(x, 2)


def s2(x):
    return flip_axis(x, 1)


def r(x):
    return np.rot90(x, axes=(1, 2))


def r2(x):
    return np.rot90(x, axes=(2, 1))


def t(x):
    return np.transpose(x, axes=(0, 2, 1, 3))


def id(x):
    return x


def compose(op1, op2):
    """Build new function from composition of two.

    Arguments:
    op1, op2: functions taking a single argument as input.

    Returns:
    New operator op2 o op1.
    """

    def comp(x):
        return op2(op1(x))

    return comp


SYMS = (None, (r, ), (s1, ), (s2, ), (t, ), (t, s1), (s2, s1), (s2, r))
INV_SYMS = (None, (r2, ), (s1, ), (s2, ), (t, ), (s1, t), (s1, s2), (r2, s2))
AUGM_ALLER_RETOUR = (
    (id, id),
    (r, r2),
    (s1, s1),
    (s2, s2),
    (t, t),
    (compose(t, s1), compose(s1, t)),
    (compose(s2, s1), compose(s1, s2)),
    (compose(s2, r), compose(r2, s2)),
)


def random_geom_transf(x, y):
    """Applies the same random geometric transformation to two input arrays.

    This function applies the same random transformation to the two 2D input arrays.
    The random transformation is chosen among the 8 possible combinations of axis symetries,
    90 degrees rotations and transposition.

    This function is typically used to augment couples (image, segmentation). The same transformation
    is applied to both.

    Arguments:
        x, y: 2D numpy array

    Returns:
        Transformed arrays.
    """
    sym = np.random.choice(SYMS)
    if sym is None:
        return x, y
    else:
        for transf in sym:
            x = transf(x)
            y = transf(y)
        return x, y


def get_largest_cc(im_bin, connectivity=1):
    """Compute largest connected component of binary image.

    Arguments:
    im_bin: ndarray of type int, containing {0, 1}
        Binary image.
    connectivity : int, optional, {1, 2, ... ndim}
        As defined in skimage.measure.label
    """
    unique = np.unique(im_bin)
    if len(unique) == 1:
        if unique[0] == 0:
            return np.zeros(im_bin.shape, dtype="uint8")
        else:
            return np.ones(im_bin.shape, dtype="uint8")
    labels = label(im_bin, connectivity=connectivity)
    largestCC = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1
                           )  # Exclude zero
    return largestCC


def get_two_largest_cc(im_bin, connectivity=1):
    """Compute two largest connected component of binary image.
    """
    labels = label(im_bin, connectivity=connectivity)
    histo = np.bincount(labels.flat)[1:]  # Exclude zero
    max1 = np.argmax(histo)
    histo[max1] = 0
    max2 = np.argmax(histo)
    largestCC = labels == (max1 + 1)
    largestCC = np.logical_or(largestCC, labels == (max2 + 1))
    return largestCC


def imread_reshape(filename):
    im = sm.imread(filename)
    shape = im.shape
    if len(shape) == 2:
        im = im.reshape(shape[0], shape[1], 1)
    return im


def segm_contours(im_segm):
    """Compute contours of the segmentation"""
    im_dil = skm.dilation(im_segm)
    im_ero = skm.erosion(im_segm)
    return im_dil != im_ero


def top_cc(input_im):
    """Keep top CCs from input image.

    Arguments:
    input_im: numpy binary array.

    Returns:
    output_im: numpy binary array, containing top CCs.
    """
    im_mark = np.copy(input_im)
    im_mark[1:, ] = False
    out_im = skm.reconstruction(im_mark, input_im)
    return out_im


def bottom_cc(input_im):
    """Keep bottom CCs from input image.

    Arguments:
    input_im: numpy binary array.

    Returns:
    output_im: numpy binary array, containing bottom CCs.
    """
    im_mark = np.copy(input_im)
    im_mark[0:-1, ] = False
    out_im = skm.reconstruction(im_mark, input_im)
    return out_im


def top_and_bottom_cc(input_im):
    """Keep top and bottom CCs from input image.

    Arguments:
    input_im: numpy binary array.

    Returns:
    output_im: numpy binary array, containing CCs.
    """
    im_mark = np.copy(input_im)
    im_mark[1:-1, ] = False
    out_im = skm.reconstruction(im_mark, input_im)
    return out_im.astype(bool)


def top1_bot4(imbin):
    im_tmp = imbin * 255
    im_mark = np.zeros(im_tmp.shape)
    im_mark[0, :] = 1
    im_mark[-1, :] = 4
    im_mark = np.minimum(im_mark, im_tmp)
    out_im = skm.reconstruction(im_mark, im_tmp)
    return out_im.astype(np.uint8)


def rec_1_4(imin):
    im_mark = np.zeros(imin.shape)
    im_mark[0, :] = 1
    im_mark[-1, :] = 4
    im_mark = np.minimum(im_mark, imin)
    out_im = skm.reconstruction(im_mark, imin)
    return out_im.astype(np.uint8)


def keep_main_cc_and_interpolate(im_label, min_size=64000):
    """Filter labels and interpolate, for pigmented reconstructed images.

    Input image im_label is supposed to contain 3 labels. Label 1 corresponds to background,
    present at the top and bottom of the image. Labels 2 and 3 correspond to stratum corneum
    and living epidermis.

    Default method (if min_size is None): Keep background CCs touching image top and bottom, and largest CC for other labels.
    If min_size an integer value, then CCs that are larger than it are kept.
    In both cases, the resulting labels are used as markers for a watershed on a flat surface, that
    fills all the gaps between and within markers.

    Arguments:
    im_label: label image (np.uint8 or np.uint6)
    min_size: minimal size of label components to be kept. If None, then only the largest one is kept.

    Returns:
    im_out: filtered and interpolated labels."""

    labels = np.sort(np.unique(im_label))
    if not ((len(labels) == 3 and (labels == [1, 2, 3]).all()) or
            (len(labels) == 4 and (labels == [0, 1, 2, 3]).all())):
        # raise ValueError("Image should only contain 3 labels: 1, 2 and 3")
        print("Warning: image does not contain exactly labesl 1, 2 and 3")

    # First, treat background (lab==1) case
    if min_size is None:
        im_tmp = top_and_bottom_cc(im_label == 1).astype(im_label.dtype)
    else:
        im_tmp = skm.remove_small_objects(
            im_label == 1, min_size=min_size).astype(im_label.dtype)
    # Treat other labels
    for lab in [2, 3]:
        if min_size is None:
            im_tmp = im_tmp + lab * get_largest_cc(im_label == lab).astype(
                im_label.dtype)
        else:
            im_tmp = im_tmp + lab * skm.remove_small_objects(
                im_label == lab, min_size=min_size).astype(im_label.dtype)
    im_out = watershed(np.zeros(im_label.shape, dtype=np.uint8), im_tmp)
    return im_out


class KeepMainCCAndInterpolate:
    """Functor for keep_main_cc_and_interpolate."""

    def __init__(self, min_size):
        self.__min_size__ = min_size

    def __call__(self, im_label):
        return keep_main_cc_and_interpolate(im_label, self.__min_size__)


def keep_main_cc_and_interpolate_vivo(im_label, min_size=64000):
    """Filter labels and interpolate, for in vivo images.

    Input image im_label is supposed to contain 3 labels. Label 1 corresponds to coupling
    medium, present at the top of the image. Label 2 correspond to epidermis and 3 to dermis.

    For labels 1 and 3, keep CCs touching image top and bottom. For label 2, keep the
    largest CCs. Then the filtered labels are used as markers for a watershed on a flat surface, that
    fills all the gaps between and within markers.

    Arguments:
    im_label: label image (np.uint8 or np.uint6)
    min_size: minimal size of label components to be kept. If None, then only the largest one is kept.

    Returns:
    im_out: filtered and interpolated labels."""

    im_tmp = top_cc(im_label == 1).astype(im_label.dtype)
    im_tmp += 3 * bottom_cc(im_label == 3).astype(im_label.dtype)
    if min_size is None:
        im_tmp += 2 * get_largest_cc(im_label == 2).astype(im_label.dtype)
    else:
        im_tmp += 2 * skm.remove_small_objects(
            im_label == 2, min_size=min_size).astype(im_label.dtype)
    im_out = watershed(np.zeros(im_label.shape, dtype=np.uint8), im_tmp)
    return im_out


class KeepMainCCAndInterpolateVivo:
    """Functor for keep_main_cc_and_interpolate_vivo."""

    def __init__(self, min_size):
        self.__min_size__ = min_size

    def __call__(self, im_label):
        return keep_main_cc_and_interpolate_vivo(im_label, self.__min_size__)


def reg_contours_bin(im_bin, filter_radius, iterations=2):
    """Regularize contours of binary image using iterated means of opening and closing."""
    im_tmp = im_bin.astype(np.uint8) * 255
    se = skm.disk(filter_radius + 1)
    se[0, filter_radius + 1] = 0
    se[-1, filter_radius + 1] = 0
    se[filter_radius + 1, 0] = 0
    se[filter_radius + 1, -1] = 0

    for i in range(iterations):
        im_open = skm.opening(im_tmp, se)
        im_close = skm.closing(im_tmp, se)
        im_tmp = (im_open.astype(float) + im_close.astype(float)) / 2
    im_bin[:, :] = im_tmp > 127


def reg_contours(imin, filter_radius, iterations=2):
    """Regularize image using iterated means of opening and closing."""
    se = skm.disk(filter_radius)
    im_tmp = imin.astype(float)
    for i in range(iterations):
        im_open = skm.opening(im_tmp, se)
        im_close = skm.closing(im_tmp, se)
        im_tmp = (im_open.astype(float) + im_close.astype(float)) / 2
    return np.around(im_tmp).astype(imin.dtype)


def best_cc_layers(im_bin):
    """Keep connected component of im_bin that are closest to a horizontal layer"""
    im_dist = distance_transform_cdt(im_bin == 0, metric="taxicab")
    im_dist[im_dist > 255] = 255
    im_dist = 255 - im_dist
    im_dist[im_bin] = 255
    im_marker = np.zeros(im_bin.shape, dtype=np.uint8)
    im_marker[0, :] = 1
    im_marker[-1, :] = 2
    im_ws_line = watershed(im_dist, im_marker, watershed_line=True) == 0
    im_ws_line = skm.dilation(im_ws_line)
    np.logical_and(im_ws_line, im_bin, im_marker)
    im_tmp = skm.reconstruction(im_marker, im_bin)
    return np.maximum(im_tmp, im_ws_line).astype(bool)


def best_cc_layers_mask(im_bin, mask):
    """Keep connected component of im_bin that are closest to a horizontal layer"""
    im_dist = distance_transform_cdt(im_bin == 0, metric="taxicab")
    im_dist[im_dist > 255] = 255
    im_dist = 255 - im_dist
    im_dist[im_bin] = 255
    im_dist[mask] = 0
    im_marker = np.zeros(im_bin.shape, dtype=np.uint8)
    im_marker[0, :] = 1
    im_marker[-1, :] = 2
    im_ws_line = watershed(im_dist, im_marker, watershed_line=True) == 0
    im_ws_line = skm.dilation(im_ws_line)
    np.logical_and(im_ws_line, im_bin, im_marker)
    im_tmp = skm.reconstruction(im_marker, im_bin)
    return np.maximum(im_tmp, im_ws_line).astype(bool)


def two_layers_old(im_label, filter_radius=0):
    """Filter labels and interpolate, for pigmented reconstructed skin (PRP) images.

    Input image im_label is supposed to contain 3 labels. Label 1 corresponds to background,
    present at the top and bottom of the image. Labels 2 and 3 correspond to stratum corneum
    and living epidermis.

    In the case of the background, connected component touching the image bottom and top are kept.

    In the case of the stratum corneum and living epidermis, components that constitute a layer going
    from the left to the right of the image - eventually partially disconnected - are kept.

    An optional filtering to regularize contours (iterated mean of opening and closing)
    is applied in the first place.

    Arguments:
    im_label: label image (np.uint8 or np.uint6)
    filter_radius: radius of structuring element used for filtering. If zero, filtering is not applied.

    Returns:
    im_out: filtered and interpolated labels."""

    max_lab = np.max(im_label)
    # First, treat background (lab==1) case
    im_bin = im_label == 1
    im_bg_1_4 = top1_bot4(im_bin)
    im_bg_1_4[im_label == 2] = 2
    im_bg_1_4[im_label == 3] = 3
    im_bg_1_4 = watershed(np.zeros(im_label.shape, dtype=np.uint8), im_bg_1_4)
    if filter_radius > 0:
        im_bg_1_4 = reg_contours(im_bg_1_4, filter_radius)
    im_tmp = np.zeros(im_label.shape, dtype=np.uint8)
    im_tmp[im_bg_1_4 == 1] = 1
    im_tmp[im_bg_1_4 == 4] = 1
    if filter_radius > 0:
        im_tmp = top_and_bottom_cc(im_tmp)

    # Treat other labels
    for lab in range(2, max_lab + 1):
        im_bin = im_bg_1_4 == lab
        im_bin = best_cc_layers(im_bin)
        im_tmp[im_bin] = lab

    # fill the gaps using a watershed
    im_out = watershed(np.zeros(im_label.shape, dtype=np.uint8), im_tmp)
    return im_out


def two_layers(im_label, filter_radius=0):
    """Filter labels and interpolate, for pigmented reconstructed skin (PRP) images.

    Input image im_label is supposed to contain 3 labels. Label 1 corresponds to background,
    present at the top and bottom of the image. Labels 2 and 3 correspond to stratum corneum
    and living epidermis.

    In the case of the background, connected components touching the image bottom and top are kept.

    In the case of the stratum corneum and living epidermis, components that constitute a layer going
    from the left to the right of the image - eventually fragmented - are kept.

    An optional filtering to regularize contours (iterated mean of opening and closing)
    is applied in the first place.

    Arguments:
    im_label: label image (np.uint8 or np.uint6)
    filter_radius (integer): radius of structuring element used for filtering.
    If zero, filtering is not applied.

    Returns:
    im_out: filtered and interpolated labels."""

    im_tmp = np.zeros(im_label.shape, dtype=np.uint8)

    # Process non background labels
    for lab in [2, 3]:
        im_bin = im_label == lab
        if filter_radius > 0:
            reg_contours_bin(im_bin, filter_radius)
        im_bin = best_cc_layers(im_bin)
        im_label[im_bin] = 0
        im_tmp[im_bin] = lab

    # Add the background
    im_bin = im_label == 1
    mask = im_tmp == 0
    im_tmp[mask] = im_bin[mask]

    # Fill the gaps
    # pdb.set_trace()
    im_out = keep_main_cc_and_interpolate(im_tmp, min_size=None)
    return im_out


def from_channels_to_labels(im_channels, chan_to_lab_lut=None):
    """Converts multi-channel image into label image.

    Arguments:
    im_channels: numpy array, where the two first dimensions correspond to space,
        and the third to channels.
    chan_to_lab_lut: look up table giving the label for a given channel. If is None, then
        the label is equal to the channel plus one.

    Returns:
    im_out: labelled image
    """
    max_label = im_channels.shape[2]
    im_max = np.amax(im_channels, 2)
    im_out = np.zeros((im_channels.shape[0], im_channels.shape[1]), np.uint16)
    for c in range(max_label):
        if chan_to_lab_lut is None:
            label = c + 1
        else:
            label = chan_to_lab_lut[c]
        im_out[im_channels[:, :, c] == im_max[:, :]] = label

    return im_out


def top_mask(input_im, thresh=[220, 220, 220]):
    """Compute a mask indicating what pixels are connected to the image top border.

    Arguments:
    input_im: numpy array corresponding to a 2D input image, with eventually several channels.
    thresh: array of size equal to the number of channels of the image.
    It is used to compute a binary version of the image.

    Returns:
    Resulting image mask, concatenated to initial input image.
    """
    shape = input_im.shape
    if len(shape) == 2:
        input_im.reshape(shape[0], shape[1], 1)
    im_mask = input_im[:, :, 0] > thresh[0]
    for c in range(1, input_im.shape[2]):
        im_mask = np.logical_and(im_mask, input_im[:, :, c] > thresh[c])
    im_mask = im_mask.astype(np.uint8)
    im_mark = np.zeros(im_mask.shape, dtype=np.uint8)
    im_mark[0, :] = im_mask[0, :]
    out_im = skm.reconstruction(im_mark, im_mask)
    out_im = out_im.reshape(out_im.shape[0], out_im.shape[1], 1)
    out_im = np.concatenate((input_im, out_im), 2)
    return out_im


def top_bot_mask(input_im, thresh=[220, 220, 220]):
    """Compute a mask indicating what pixels are connected to the image top or bottom border.

    Arguments:
    input_im: numpy array corresponding to a 2D input image, with eventually several channels.
    thresh: array of size equal to the number of channels of the image.
    It is used to compute a binary version of the image.

    Returns:
    Resulting image mask, concatenated to initial input image.
    """
    shape = input_im.shape
    if len(shape) == 2:
        input_im.reshape(shape[0], shape[1], 1)
    im_mask = input_im[:, :, 0] > thresh[0]
    for c in range(1, input_im.shape[2]):
        im_mask = np.logical_and(im_mask, input_im[:, :, c] > thresh[c])
    im_mask = im_mask.astype(np.uint8)
    im_mark = np.zeros(im_mask.shape, dtype=np.uint8)
    im_mark[0, :] = im_mask[0, :]
    im_mark[-1, :] = im_mask[-1, :]
    out_im = skm.reconstruction(im_mark, im_mask)
    out_im = out_im.reshape(out_im.shape[0], out_im.shape[1], 1)
    out_im = np.concatenate((input_im, out_im), 2)
    return out_im


class TopMask:
    """Functor for top_mask()."""

    def __init__(self, thresh):
        """Init functor parameters.

        Arguments:
        thresh: array of size equal to the number of channels of the images
                to be processed.
        """
        self.__thresh__ = thresh

    def __call__(self, input_im):
        if input_im.shape[-1] != len(self.__thresh__):
            raise ValueError("Input image should contain %d channels" %
                             (len(self.__thresh__)))
        return top_mask(input_im, self.__thresh__)


def distance_mask(input_im, label_transfer=None):
    """Compute a mask indicating the distance from the groud truth.

    Arguments:
    input_gt: numpy array corresponding to a 2D input image, with one channel.
    It is used to transform the binary mask to signed distance function
    Meanwhile, it transfer the label when it's activated

    Returns:
    Resulting new ground truth containing the distance map and the contour/bundary.
    """
    shape = input_im.shape
    if label_transfer is not None:
        label_transfer(input_im)
    list_labels = np.unique(input_im)
    nb_labels = len(list_labels)
    out_im = np.zeros((shape[0], shape[1], nb_labels), dtype="float32")
    for i in range(nb_labels):
        inside = -distance_transform_edt(input_im == list_labels[i])
        outside = distance_transform_edt(input_im != list_labels[i])
        out_im[:, :, i] = outside + inside
    return out_im


class DistanceMask:
    """Functor for distance_mask()."""

    def __init__(self, label_transfer):
        """Init functor parameters.

        Arguments:
        label_transfer: function to transfer certain labels in ground truth.
        """
        self.__label_transfer__ = label_transfer

    def __call__(self, input_im):
        if len(np.unique(input_im)) == 1:
            raise ValueError("Input image should contain more than one label")
        return distance_mask(input_im, self.__label_transfer__)


def div_255(im, channels=None):
    im = im.astype(np.float32)
    if channels is None:
        im /= 255
    else:
        im[:, :, 0:channels] /= 255
    return im


def fill_holes_from_top_and_bottom_div_255(input_im, nb_channels):
    """Compute for the first nb_channels of input_im, a fill holes from top and bottom of the image"""
    shape = input_im.shape
    if len(shape) == 2:
        input_im.reshape(shape[0], shape[1], 1)
    for c in range(nb_channels):
        im_mark = np.copy(input_im[:, :, c])
        im_mark[1:im_mark.shape[0] - 1, :] = 0
        input_im[:, :, c] = (
            skm.reconstruction(im_mark, input_im[:, :, c], method="dilation") /
            2 + input_im[:, :, c] / 2)

    return div_255(input_im, nb_channels)


class FillHolesFromTopAndBottomDiv255:
    """Functor for fill_holes_from_top_and_bottom_div_255."""

    def __init__(self):
        pass

    def __call__(self, input_im, nb_channels):
        return fill_holes_from_top_and_bottom_div_255(input_im, nb_channels)


def fill_holes_from_top_and_bottom(input_im, nb_channels):
    """Compute for the first nb_channels of input_im, a fill holes from top and bottom of the image"""
    shape = input_im.shape
    if len(shape) == 2:
        input_im.reshape(shape[0], shape[1], 1)
    for c in range(nb_channels):
        im_mark = np.copy(input_im[:, :, c])
        im_mark[1:im_mark.shape[0] - 1, :] = 0
        input_im[:, :, c] = (
            skm.reconstruction(im_mark, input_im[:, :, c], method='dilation') /
            2 + input_im[:, :, c] / 2)

    return input_im


def makePrepro(dirIn=".", dirOut="."):
    listIn = os.listdir(dirIn)
    for entry in listIn:
        if os.path.isdir(entry):
            continue
        (base, ext) = os.path.splitext(entry)
        if ext in [".png", ".tif"]:
            print("* Handling : " + entry)
            imIn = imread(os.path.join(dirIn, entry))
            imOut = fill_holes_from_top_and_bottom(imIn, 3)
            imsave(os.path.join(dirOut, base + "_prepro2.png"), imOut)


def mean_distance_contour(ground_truth, im_pred, class1, class2):
    """
    Compute mean distance between reference and predicted contours.

    Arguments:
    ground_truth: numpy array containing a reference segmentation.
    im_pred: numpy array containing a predicted segmentation.
    class1, class2: labels defining two regions. The frontier between
        these regions is the contour of interest.

    Returns:
    Mean distance between the reference and predicted contours.
    """
    ref_contour = skm.binary_dilation(ground_truth == class2,
                                      skm.square(3)) * (ground_truth == class1)
    det_contour = skm.binary_dilation(im_pred == class2,
                                      skm.square(3)) * (im_pred == class1)
    data1 = distance_transform_edt(ref_contour != 1)[det_contour]
    data2 = distance_transform_edt(det_contour != 1)[ref_contour]
    return 0.5 * (np.mean(data1) + np.mean(data2))


def two_layers_ordered(im_label, filter_radius=0):
    """Filter labels and interpolate, for pigmented reconstructed skin (PRP) images.

    Input image im_label is supposed to contain 3 labels. Label 1 corresponds to background,
    present at the top and bottom of the image. Labels 2 and 3 correspond to stratum corneum
    and living epidermis.

    In the case of the background, connected components touching the image bottom and top are kept.

    In the case of the living epidermis, components that constitute a layer going
    from the left to the right of the image - eventually fragmented - are kept.

    The stratum corneum pixels beneath the living epidermis layer are set to background pixels,
    afterwards the same procedure used for the living epidermis is used on the remaning stratum corneum pixels.

    An optional filtering to regularize contours (iterated mean of opening and closing)
    is applied in the first place.

    Arguments:
    im_label: label image (np.uint8 or np.uint6)
    filter_radius (integer): radius of structuring element used for filtering.
    If zero, filtering is not applied.

    Returns:
    im_out: filtered and interpolated labels."""

    im_tmp = np.zeros(im_label.shape, dtype=np.uint8)
    im_label = im_label.copy()

    LE = 3
    SC = 2
    # Process living epidermis
    im_bin = im_label == LE
    if filter_radius > 0:
        reg_contours_bin(im_bin, filter_radius)
    im_bin = best_cc_layers(im_bin)
    im_label[im_bin] = 0
    im_tmp[im_bin] = LE

    # Set stratum corneum pixels below the living epidermis layer to 0
    last_le = im_label.shape[0] - np.argmax(im_tmp[::-1, :], 0)
    mask = np.tile(
        np.arange(0, im_label.shape[0])[:, np.newaxis], [1, im_label.shape[1]])
    mask = mask > last_le[np.newaxis, :]
    im_label[mask] = 1

    # Set stratum corneum connected components that do not touch living epidermis layer to 0
    conn_comps_sc = label(im_label == SC)
    for i in range(1, conn_comps_sc.max()):
        conn_comp = (conn_comps_sc == i).astype(np.uint8)
        outer_gradient = (skm.dilation(conn_comp) - conn_comp) > 0
        outer_gradient_values = im_tmp[outer_gradient]
        if LE not in outer_gradient_values:
            im_label[conn_comp.astype(bool)] = 1

    im_bin = im_label == SC
    if filter_radius > 0:
        reg_contours_bin(im_bin, filter_radius)
    im_bin = best_cc_layers(im_bin)
    im_label[im_bin] = 0
    im_tmp[im_bin] = SC

    # Add the background
    im_bin = im_label == 1
    mask = im_tmp == 0
    im_tmp[mask] = im_bin[mask]

    # Fill the gaps
    # pdb.set_trace()
    im_out = keep_main_cc_and_interpolate(im_tmp, min_size=None)
    return im_out


class TwoLayers:
    """Functor for two_layers and two_layers_ordered."""

    def __init__(self, filter_radius, ordered=False):
        """Init functor parameters.

        Arguments:
        filter_radius: radius of structuring element used for filtering.
        """
        self.__radius__ = filter_radius
        self.__ordered__ = ordered

    def __call__(self, input_im):
        if len(input_im.shape) != 2:
            raise ValueError(
                "Input image must have exactly 2 dimensions (1 channel).")
        if self.__ordered__:
            return two_layers_ordered(input_im, filter_radius=self.__radius__)
        return two_layers(input_im, filter_radius=self.__radius__)


def contour(pred, class0, class1):
    pred_class0 = pred == class0
    pred_class1 = pred == class1
    dil_class0 = binary_dilation(pred_class0, iterations=2)
    dil_class1 = binary_dilation(pred_class1, iterations=2)
    return np.minimum(dil_class0, dil_class1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # proba = imread(
    #     'sweeps_runs/unet_2023-03-03_02-27-31/predictions_efi20x-prepro_all-000/images/'
    #     'F21_07_04_PRP_FM_20220915_155143_X002_Y001_A000_Sub03_prepro_proba.png'
    # )
    # proba = imread('sweeps_runs/unet_2023-03-03_02-27-31/predictions_prepro10x_all-001/images/'
    #                'F21_14_01_PRP_FM_20220919_142659_X001_Y001_A000_Sub04_prepro_proba.png')
    # proba = imread(
    #         'sweeps_runs/multiscale_sa_unet_2023-04-10_18-09-53/'
    #         'predictions_prepro10x_104_all-000/images/'
    #         'F21_07_09_PRP_FM_20220919_141321_X002_Y001_A001_Sub04_prepro_proba.png')
    proba = imread(
            'sweeps_runs/multiscale_sa_unet_2023-04-10_18-09-53/'
            'predictions_prepro10x_104_all-000/images/'
            'F21_07_09_PRP_FM_20220919_141321_X001_Y001_A000_Sub04_prepro_proba.png')

    # proba = proba[::3, ::3, :]
    plt.imshow(proba)
    plt.show()

    pred = np.argmax(proba, -1)
    pred = np.array([2, 1, 3])[pred]
    post = two_layers_ordered(pred, 3)

    plt.subplot(121)
    plt.imshow(pred)
    plt.subplot(122)
    plt.imshow(post)
    plt.show()
