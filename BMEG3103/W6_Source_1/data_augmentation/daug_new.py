# daug_new.py - Keras 3 version of daug.py
#
# Old script used ImageDataGenerator (removed in Keras 3).
# New way: build a model out of data-augmentation layers and apply it
# repeatedly to one image, saving 20 randomly transformed copies.

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence the oneDNN notice

import numpy as np
import keras
from keras import layers
from keras.utils import load_img, img_to_array, save_img

# ---------------------------------------------------------------------------
# Data augmentation pipeline (same settings as the old ImageDataGenerator)
#   rotation_range=40       -> RandomRotation(40/360)
#   width_shift_range=0.2   -> RandomTranslation(0.2, 0.2)
#   height_shift_range=0.2
#   zoom_range=0.2          -> RandomZoom(0.2)
#   horizontal_flip=True    -> RandomFlip("horizontal")
#   fill_mode='nearest'     -> fill_mode="nearest" on each layer
#   (shear_range has no built-in Keras 3 layer - omitted)
# ---------------------------------------------------------------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(40 / 360),
        layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
        layers.RandomZoom(0.2, fill_mode="nearest"),
        layers.RandomFlip("horizontal"),
    ],
    name="data_augmentation",
)

# Load the image: shape (H, W, 3), values 0-255
img = load_img("./cat.0.jpg")
x = img_to_array(img)
x = np.expand_dims(x, axis=0)  # shape (1, H, W, 3) - add batch dimension
print("Input image shape:", x.shape)

# Output folder (relative to this script)
out_dir = "./output"
os.makedirs(out_dir, exist_ok=True)

# Generate 20 randomly transformed copies
NUM_AUG = 20
for i in range(NUM_AUG):
    batch = data_augmentation(x, training=True)   # random transform each call
    aug_img = np.array(batch[0]).astype("uint8")  # back to 0-255 image
    out_path = os.path.join(out_dir, f"et3_aug_{i + 1:02d}.jpg")
    save_img(out_path, aug_img)
    print(f"saved {out_path}")

print(f"\nDone - {NUM_AUG} augmented images in {out_dir}/")
