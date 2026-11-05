"""
Data generators for U-Net (TensorFlow / Keras).

Ported from zhixuhao/unet (https://github.com/zhixuhao/unet).
The original repo used standalone Keras; this version uses
`tensorflow.keras.preprocessing.image.ImageDataGenerator`.

Two key generators are exposed:

    trainGenerator(...)  -- image/mask pairs with synchronized augmentation
    testGenerator(...)   -- load N test images one at a time

Plus two helpers:

    geneTrainNpy(...)    -- read all images into RAM as NumPy arrays
    saveResult(...)      -- write predicted masks to disk
"""

from __future__ import print_function

import os
import glob

import numpy as np
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Color dict kept verbatim from the original repo. Used only by
# `labelVisualize` when `flag_multi_class=True` (multiclass segmentation).
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([
    Sky, Building, Pole, Road, Pavement,
    Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled,
])


def adjustData(img, mask, flag_multi_class, num_class):
    """Normalize image / mask pair.

    For binary segmentation (default), scale both to [0, 1] and
    threshold the mask at 0.5.

    For multiclass (`flag_multi_class=True`), build a one-hot mask
    over `num_class` channels.
    """
    if flag_multi_class:
        img = img / 255.0
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,), dtype=np.float32)
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(
            new_mask,
            (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])
        ) if flag_multi_class else np.reshape(
            new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2])
        )
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255.0
        mask = mask / 255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder,
                   aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale",
                   image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2,
                   save_to_dir=None, target_size=(256, 256), seed=1):
    """Yield (image_batch, mask_batch) with synchronized augmentation.

    The image and mask `ImageDataGenerator`s share the same `seed`,
    so every geometric transform is applied identically to both --
    a hard requirement for segmentation.

    Parameters
    ----------
    batch_size : int
        Number of image/mask pairs per yielded batch.
    train_path : str
        Root directory that contains `image_folder/` and `mask_folder/`.
    image_folder, mask_folder : str
        Sub-folder names under `train_path` (e.g. 'image', 'label').
    aug_dict : dict
        Keyword args passed to `ImageDataGenerator`, e.g.
        `rotation_range`, `width_shift_range`, `horizontal_flip`, ...
    save_to_dir : str or None
        If given, augmented samples are also written to this directory
        (handy for visualizing what the generator is producing).
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def testGenerator(test_path, num_image=30, target_size=(256, 256),
                  flag_multi_class=False, as_gray=True):
    """Yield one normalized test image at a time, as a 4-D batch of size 1."""
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255.0
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2,
                 image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    """Load the entire (augmented) training set into NumPy arrays.

    Use this when the dataset fits in RAM. Otherwise prefer `trainGenerator`.
    """
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr, mask_arr = [], []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(
            item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
            as_gray=mask_as_gray,
        )
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    """Map a class-index mask back to an RGB image using `COLOR_DICT`."""
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255.0


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    """Save a batch of predicted masks as `0_predict.png`, `1_predict.png`, ...

    Output: `<save_path>/<i>_predict.png`
    """
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) \
            if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)