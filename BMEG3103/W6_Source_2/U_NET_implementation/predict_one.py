"""
Predict on a single arbitrary image (any filename, any size).

This is the friendlier counterpart to `predict.py` (which expects the
original repo's `0.png, 1.png, ...` naming). Use it to try the trained
model on your own microscopy / cell / medical images.

What it does
------------
    1. Loads `model.py`'s U-Net with weights from `--weights`.
    2. Loads the input image (color or grayscale, any size).
    3. Resizes it to the model's expected input (default 256x256).
    4. Runs prediction; resizes the mask back to the original image size.
    5. Saves three files into `--out-dir`:
         <name>_predict_raw.png     -- probability mask in [0,1] (grayscale)
         <name>_predict_bin.png     -- thresholded mask (binary)
         <name>_predict_overlay.png -- mask overlaid on the original

How to run
----------
    python predict_one.py my_image.png
    python predict_one.py my_image.png --weights unet_membrane.keras
    python predict_one.py my_image.jpg --threshold 0.5
"""

import argparse
import os

import numpy as np
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from model import unet


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a trained U-Net on a single image of your choice."
    )
    p.add_argument("image", help="Path to the input image (png/jpg/tif/...).")
    p.add_argument("--weights", default="unet_membrane.keras",
                   help="Trained weights file (default: unet_membrane.keras).")
    p.add_argument("--out-dir", default=".",
                   help="Where to write the outputs (default: current dir).")
    p.add_argument("--input-size", type=int, default=256,
                   help="Spatial size the model expects (default: 256).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for the binarized mask.")
    return p.parse_args()


def load_as_grayscale_array(path: str) -> np.ndarray:
    """Load image as a (H, W, 1) float32 array in [0, 1]."""
    # load_img handles png/jpg/tif/bmp and resizes on demand.
    pil = load_img(path, color_mode="grayscale",
                   target_size=None)        # keep original size
    arr = img_to_array(pil) / 255.0          # (H, W, 1) float32 in [0,1]
    return arr


def predict_single(model, img: np.ndarray, input_size: int) -> np.ndarray:
    """Run the model and return a probability mask at the model's output size."""
    h, w = img.shape[:2]
    resized = trans.resize(img, (input_size, input_size),
                           preserve_range=False)
    batch = np.expand_dims(resized, 0)       # (1, H, W, 1)
    pred = model.predict(batch, verbose=0)[0, ..., 0]   # (H, W)
    return pred


def save_outputs(out_dir: str, base_name: str,
                 original: np.ndarray, prob_mask: np.ndarray,
                 threshold: float):
    """Write raw, binarized, and overlay images to disk."""
    os.makedirs(out_dir, exist_ok=True)

    # 1) raw probability mask, resized back to original image size
    raw_full = trans.resize(prob_mask, original.shape[:2],
                            preserve_range=True)
    raw_full = np.clip(raw_full, 0.0, 1.0)
    raw_path = os.path.join(out_dir, f"{base_name}_predict_raw.png")
    io.imsave(raw_path, (raw_full * 255).astype(np.uint8))

    # 2) binarized mask
    bin_mask = (raw_full > threshold).astype(np.uint8) * 255
    bin_path = os.path.join(out_dir, f"{base_name}_predict_bin.png")
    io.imsave(bin_path, bin_mask)

    # 3) overlay: original (gray) + mask (red, alpha=0.5)
    base = original[..., 0] if original.ndim == 3 else original
    overlay = np.stack([base, base, base], axis=-1)   # (H, W, 3) grayscale-as-RGB
    overlay = (overlay * 255).astype(np.uint8) if overlay.max() <= 1.0 \
        else overlay.astype(np.uint8)
    red = (bin_mask > 0)
    overlay[red, 0] = 255      # R
    overlay[red, 1] = 0        # G
    overlay[red, 2] = 0        # B
    overlay_path = os.path.join(out_dir, f"{base_name}_predict_overlay.png")
    io.imsave(overlay_path, overlay)

    return raw_path, bin_path, overlay_path


def main():
    args = parse_args()
    if not os.path.exists(args.image):
        raise SystemExit(f"Input image not found: {args.image}")
    if not os.path.exists(args.weights):
        raise SystemExit(
            f"Weights not found: {args.weights}\n"
            f"  -> Train first with `python main.py` "
            f"or pass --weights /path/to/checkpoint."
        )

    print(f"Loading model with weights from {args.weights} ...")
    model = unet(pretrained_weights=args.weights,
                 input_size=(args.input_size, args.input_size, 1))

    print(f"Loading {args.image} ...")
    original = load_as_grayscale_array(args.image)
    print(f"  -> shape: {original.shape}")

    print(f"Running prediction at {args.input_size}x{args.input_size} ...")
    prob_mask = predict_single(model, original, args.input_size)
    print(f"  -> prob mask: min={prob_mask.min():.4f} max={prob_mask.max():.4f} "
          f"mean={prob_mask.mean():.4f}")

    # Warn when the threshold will produce an EMPTY mask -- this usually
    # means the model is undertrained (e.g. all probabilities hover below
    # the threshold) rather than "no object in the image".
    if prob_mask.max() < args.threshold:
        print(f"\n  !! WARNING: max probability {prob_mask.max():.4f} is BELOW "
              f"the threshold {args.threshold}.")
        print("     The binarized mask (and the red overlay) will be EMPTY.")
        print("     Likely causes:")
        print("       1. Model not trained enough -> train longer, e.g.")
        print("          python main.py --dataset lung --epochs 10 --steps 500")
        print("       2. Severe class imbalance (tiny objects) -> try a lower")
        print(f"          threshold, e.g. --threshold {max(0.05, round(float(np.percentile(prob_mask, 99)), 2))}")
        print("       3. Image from a different domain than the training data.")

    base = os.path.splitext(os.path.basename(args.image))[0]
    raw, bin_, overlay = save_outputs(
        args.out_dir, base, original, prob_mask, args.threshold
    )
    print(f"\nDone. Outputs:")
    print(f"  raw      : {raw}")
    print(f"  binary   : {bin_}    (threshold = {args.threshold})")
    print(f"  overlay  : {overlay}")


if __name__ == '__main__':
    main()