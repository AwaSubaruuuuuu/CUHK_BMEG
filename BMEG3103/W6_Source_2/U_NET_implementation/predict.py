"""
Standalone inference script -- load a trained U-Net and predict on a folder.

Use this when you already have a checkpoint (e.g. `unet_membrane.keras`)
and just want to run the model on a new directory of test images.

How to run
----------
    python predict.py --weights unet_membrane.keras \
                      --test-dir data/membrane/test \
                      --out-dir  data/membrane/test

The output filenames will be `<out-dir>/<i>_predict.png`, matching
`saveResult` from `data.py` so it is drop-in compatible with the rest
of the repo.
"""

import argparse
import os

import numpy as np

from model import unet
from data import testGenerator, saveResult


def parse_args():
    p = argparse.ArgumentParser(description="Run a trained U-Net on a folder of test images.")
    p.add_argument("--weights", default="unet_membrane.keras",
                   help="Path to the .keras / .h5 weights file.")
    p.add_argument("--test-dir", default="data/membrane/test",
                   help="Folder containing the test PNGs (named 0.png, 1.png, ...).")
    p.add_argument("--out-dir", default="data/membrane/test",
                   help="Where to write <i>_predict.png files.")
    p.add_argument("--num-images", type=int, default=30,
                   help="Number of test images to process.")
    p.add_argument("--input-size", type=int, default=256,
                   help="Spatial size the model expects (must match training).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model with weights from {args.weights} ...")
    model = unet(pretrained_weights=args.weights,
                 input_size=(args.input_size, args.input_size, 1))

    target_size = (args.input_size, args.input_size)
    test_gen = testGenerator(args.test_dir,
                             num_image=args.num_images,
                             target_size=target_size)

    print(f"Predicting on {args.num_images} images from {args.test_dir} ...")
    results = model.predict(test_gen, verbose=1)

    saveResult(args.out_dir, results)
    print(f"Done. Wrote {len(results)} predicted masks to {args.out_dir}/")


if __name__ == '__main__':
    main()