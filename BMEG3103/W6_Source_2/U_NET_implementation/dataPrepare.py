"""
Data augmentation demo (port of dataPrepare.ipynb).

This script is the Python equivalent of `original_notebooks/dataPrepare.ipynb`.
It walks the student through:

    1. Building a Keras `ImageDataGenerator` with on-the-fly augmentation.
    2. Sampling augmented batches and dumping them to disk so students can
       visually verify that the image and mask transforms are synchronized.
    3. (Optional) Loading the full augmented set into NumPy arrays.

How to run
----------
    python dataPrepare.py

Outputs
-------
    data/membrane/train/aug/image_*.png   -- 3 batches x 20 = 60 augmented images
    data/membrane/train/aug/mask_*.png    -- corresponding augmented masks
"""

from data import trainGenerator, geneTrainNpy


# --- 1. Augmentation settings ----------------------------------------------
# An empty dict disables augmentation; the values below match the original repo.
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
)


def demo_visualize_augmented(num_batch=3, batch_size=20):
    """Write `num_batch * batch_size` augmented (image, mask) pairs to disk.

    Setting `save_to_dir` on the generator lets you inspect what the
    generator actually produces. Open the resulting PNGs and confirm
    that the geometric transforms are *exactly* the same for the image
    and its mask -- otherwise the model learns garbage.
    """
    myGenerator = trainGenerator(
        batch_size=batch_size,
        train_path='data/membrane/train',
        image_folder='image',
        mask_folder='label',
        aug_dict=data_gen_args,
        save_to_dir='data/membrane/train/aug',
    )
    for i, batch in enumerate(myGenerator):
        if i >= num_batch:
            break
    print(f"Wrote {num_batch * batch_size} augmented samples "
          f"to data/membrane/train/aug/")


def demo_npy(image_dir='data/membrane/train/aug/',
             mask_dir='data/membrane/train/aug/'):
    """Materialize the (augmented) training set as NumPy arrays.

    Use this only if the dataset fits in RAM; otherwise prefer the
    generator approach in `main.py` / `trainUnet.py`.
    """
    image_arr, mask_arr = geneTrainNpy(image_dir, mask_dir)
    print(f"image_arr shape: {image_arr.shape}")
    print(f"mask_arr shape:  {mask_arr.shape}")
    # np.save("data/image_arr.npy", image_arr)
    # np.save("data/mask_arr.npy", mask_arr)
    return image_arr, mask_arr


if __name__ == '__main__':
    demo_visualize_augmented()
    # demo_npy()    # uncomment to also materialize as NumPy arrays