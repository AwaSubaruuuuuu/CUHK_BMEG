"""
Train and evaluate U-Net on the membrane dataset
(port of trainUnet.ipynb).

This is the step-by-step version of `main.py`. It matches the cells of the
original notebook 1-to-1, so students can follow the same flow in their
editor or in a fresh notebook.

How to run
----------
    python trainUnet.py

It will:
    1. Train the U-Net using the on-the-fly augmentation generator.
    2. Save the best checkpoint to `unet_membrane.keras`.
    3. Run prediction on the 30 test images and write
       `data/membrane/test/<i>_predict.png`.
"""

import os

# If you have multiple GPUs, restrict to one.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.callbacks import ModelCheckpoint

from model import unet
from data import trainGenerator, geneTrainNpy, testGenerator, saveResult


# --- 1. Set up the augmentation pipeline -----------------------------------
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
)


def train_with_generator(steps_per_epoch=2000, epochs=5):
    """Train using the synchronized image/mask generator (preferred)."""
    train_gen = trainGenerator(
        batch_size=2,
        train_path='data/membrane/train',
        image_folder='image',
        mask_folder='label',
        aug_dict=data_gen_args,
        save_to_dir=None,
    )
    model = unet()
    # Keras 3.x requires the filepath to end in `.keras` (Keras native format).
    model_checkpoint = ModelCheckpoint(
        'unet_membrane.keras',
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint],
    )
    return model


def train_with_npy(epochs=10):
    """Train using NumPy arrays (only when the data fits in RAM).

    The line below is commented out in the original notebook; the generator
    pipeline is the recommended path.
    """
    # imgs_train, imgs_mask_train = geneTrainNpy(
    #     "data/membrane/train/aug/", "data/membrane/train/aug/"
    # )
    # model = unet()
    # model_checkpoint = ModelCheckpoint('unet_membrane.keras',
    #                                    monitor='loss', verbose=1,
    #                                    save_best_only=True)
    # model.fit(imgs_train, imgs_mask_train,
    #           batch_size=2, epochs=epochs, verbose=1,
    #           validation_split=0.2, shuffle=True,
    #           callbacks=[model_checkpoint])
    raise NotImplementedError(
        "Enable the geneTrainNpy + model.fit block above to use this path."
    )


def predict_and_save(weights_path='unet_membrane.keras',
                     test_dir='data/membrane/test',
                     num_images=30):
    """Run the trained model on the test set and write `_predict.png` files."""
    model = unet(pretrained_weights=weights_path)
    test_gen = testGenerator(test_dir, num_image=num_images)
    results = model.predict(test_gen, verbose=1)
    saveResult(test_dir, results)
    return results


if __name__ == '__main__':
    # Option A: train with the generator (this is what the notebook does).
    train_with_generator()

    # Option B: materialize everything as .npy and train with model.fit().
    # train_with_npy()

    # Option C: load weights from a previous run and only run inference.
    # predict_and_save()

    # The original notebook runs inference immediately after training:
    if os.path.exists('unet_membrane.keras'):
        predict_and_save()
    else:
        print("No checkpoint found at unet_membrane.keras -- skipping inference.")