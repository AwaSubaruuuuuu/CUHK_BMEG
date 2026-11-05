"""
Train U-Net on a 2D biomedical segmentation dataset (port of zhixuhao/unet main.py).

Two datasets are bundled under data/ (same folder convention, so the same
training code works for both):

    --dataset membrane   ISBI 2015 EM cell-membrane segmentation (30 train
                         images, 512x512 grayscale; the original zhixuhao
                         dataset -- segment the outlines between cells).
    --dataset lung       Lung CT tumor segmentation (1,832 train slices,
                         256x256; sourced from the public "Lung Tumor
                         Segmentation" set on Hugging Face. Segment the
                         tumor region inside the lung).

How to run
----------
    python main.py                              # membrane, 5 epochs x 300 steps
    python main.py --dataset lung               # lung CT tumor segmentation
    python main.py --epochs 1 --steps 50        # quick smoke test (~2 min CPU)
    python main.py --no-predict                 # train only

IMPORTANT: `--steps` (steps_per_epoch) directly controls how many
batches the model sees per epoch. With batch_size=2, `steps=5` means the
model sees only 10 images/epoch -- far too few to learn anything
(predictions will be a flat gray ~0.5 mask). Keep the total number of
*steps* in the hundreds for visible results.

Outputs
-------
    unet_membrane.keras / unet_lung.keras  -- best checkpoint (by train loss)
    data/<dataset>/test/*_predict.png      -- predicted masks
"""

import argparse
import os

# Pick GPU 0 if you have multiple; uncomment to restrict.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.callbacks import ModelCheckpoint

from model import unet
from data import trainGenerator, testGenerator, saveResult


DATASETS = {
    'membrane': dict(train_dir='data/membrane/train', test_dir='data/membrane/test',
                     test_count=30, ckpt='unet_membrane.keras'),
    'lung':     dict(train_dir='data/lung/train',     test_dir='data/lung/test',
                     test_count=98,  ckpt='unet_lung.keras'),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net on a bundled dataset.")
    p.add_argument("--dataset", choices=list(DATASETS), default="membrane",
                   help="Which dataset under data/ to train on (default: membrane).")
    p.add_argument("--epochs", type=int, default=5,
                   help="Number of training epochs (default: 5).")
    p.add_argument("--steps", type=int, default=300,
                   help="Steps per epoch (default: 300, as in the original repo). "
                        "Use e.g. 50 for a quick sanity run.")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Batch size (default: 2).")
    p.add_argument("--no-predict", action="store_true",
                   help="Skip test-set prediction after training.")
    return p.parse_args()


# --- Data augmentation settings (same recipe for both datasets) -----------
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
)


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]

    # --- Training data generator (image/label pairs, synced augmentation) -
    myGene = trainGenerator(
        batch_size=args.batch_size,
        train_path=cfg['train_dir'],
        image_folder='image',
        mask_folder='label',
        aug_dict=data_gen_args,
        save_to_dir=None,           # set to a path to inspect augmented samples
    )

    model = unet()
    # Keras 3.x requires the filepath to end in `.keras` (Keras native format).
    model_checkpoint = ModelCheckpoint(
        cfg['ckpt'],
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )

    model.fit(
        myGene,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        callbacks=[model_checkpoint],
    )

    if args.no_predict:
        return

    # --- Inference on the test set ---------------------------------------
    testGene = testGenerator(cfg['test_dir'], num_image=cfg['test_count'])
    results = model.predict(testGene, verbose=1)
    saveResult(cfg['test_dir'], results)
    print(f"\nDone. Checkpoint: {cfg['ckpt']} | Predictions: {cfg['test_dir']}/*_predict.png")


if __name__ == '__main__':
    main()