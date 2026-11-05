# U-Net Implementation (port of zhixuhao/unet)

A self-contained, runnable port of the well-known Keras U-Net example by
[zhixuhao](https://github.com/zhixuhao/unet) — originally written for
standalone Keras 1.x — modernized for TensorFlow 2.x so students can
read, run and modify it on a current setup.

The architecture is the symmetric 4-down / 4-up U-Net from:

> Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image
> Segmentation*, MICCAI 2015.

---

## Files in this folder

| File                     | What it is                                                                 |
|--------------------------|----------------------------------------------------------------------------|
| `model.py`               | The U-Net architecture (Keras functional API).                            |
| `data.py`                | Data generators: `trainGenerator`, `testGenerator`, `geneTrainNpy`, etc. |
| `main.py`                | End-to-end training script (the modern equivalent of the original main).  |
| `dataPrepare.py`         | Python version of `original_notebooks/dataPrepare.ipynb` (aug demo).       |
| `trainUnet.py`           | Python version of `original_notebooks/trainUnet.ipynb` (train + test).    |
| `predict.py`             | CLI inference helper: load weights and predict on a folder.               |
| `predict_one.py`         | CLI inference helper for a **single image** (any filename / size).         |
| `requirements.txt`       | Minimal pip dependencies.                                                  |
| `data/membrane/`         | ISBI 2015 membrane dataset (30 train + 30 test images + augmentation).    |
| `img/`                   | Architecture diagram and example outputs from the original repo.          |
| `original_notebooks/`    | The two Jupyter notebooks from the source repo, kept for reference.        |
| `LICENSE`                | The MIT license from the original repo.                                    |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Visualize what the data generator does

```bash
python dataPrepare.py
```

This writes 60 augmented (image, mask) pairs into `data/membrane/train/aug/`
so students can open them and verify that the geometric transforms are
**identical** for the image and its mask.

### 3. Train the model

```bash
python main.py
```

or, step-by-step (matches `trainUnet.ipynb` cell-by-cell):

```bash
python trainUnet.py
```

The script saves the best checkpoint to `unet_membrane.keras`
(by training loss) and then predicts on the 30 test images,
writing `data/membrane/test/<i>_predict.png`.

Tweak the constants at the top of `main.py` (or the function arguments
in `trainUnet.py`) to experiment with batch size, augmentation strength,
number of epochs, etc.

### 4. Predict only (skip training)

```bash
python predict.py --weights unet_membrane.keras \
                  --test-dir data/membrane/test \
                  --out-dir  data/membrane/test
```

### 5. Predict on **your own image**

```bash
python predict_one.py path/to/your_image.png
```

This is the friendly single-image entry point — it auto-handles any
filename, any size, color or grayscale. It writes three files next to
your input:

```
your_image_predict_raw.png      # probability mask in [0,1] (grayscale)
your_image_predict_bin.png      # thresholded mask (binary)
your_image_predict_overlay.png  # mask (red) on top of the original
```

Useful flags:
```bash
python predict_one.py my.jpg --weights unet_membrane.keras \
                            --input-size 256 --threshold 0.5 \
                            --out-dir results/
```

---

## What changed vs. the GitHub source

The original repo was written around 2017 for Keras 1.0 / standalone
Keras. The most important updates for a current TF/Keras student:

| Original (`zhixuhao/unet`)              | This port                                   |
|-----------------------------------------|---------------------------------------------|
| `from keras.models import *`            | `from tensorflow.keras.models import Model` |
| `from keras.layers import *`            | `from tensorflow.keras.layers import ...`   |
| `Model(input=..., output=...)`          | `Model(inputs=..., outputs=...)`            |
| `model.fit_generator(...)`              | `model.fit(...)`                            |
| `model.predict_generator(...)`          | `model.predict(...)`                        |
| `model.fit(..., nb_epoch=...)`          | `model.fit(..., epochs=...)`                |
| `keras.callbacks.ModelCheckpoint`       | `tensorflow.keras.callbacks.ModelCheckpoint`|
| `from keras.preprocessing.image import …`| `from tensorflow.keras.preprocessing.image …`|

Everything else — the architecture, the `COLOR_DICT`, the augmentation
recipe, the loss (binary cross-entropy) and the optimizer (Adam, lr=1e-4)
— is identical to the original repo, so this folder is a fair drop-in
replacement when you want to teach the same code with a modern stack.

The two original Jupyter notebooks are kept in `original_notebooks/` for
historical reference; you can open them with `jupyter notebook
original_notebooks/trainUnet.ipynb`.

---

## Dataset(s)

### 1. `data/membrane/` — ISBI 2015 EM cell-membrane segmentation (default)

The original dataset of the U-Net paper: [ISBI 2015 challenge](http://brainiac2.mit.edu/isbi_challenge/)
electron microscopy image stack, pre-extracted into per-image PNGs.

**Task:** segment the *membranes* (thin outlines between neurons).

```
data/membrane/
├── train-volume.tif      # original 30-frame EM training stack
├── train-labels.tif      # original 30-frame EM training labels
├── test-volume.tif       # original 30-frame EM test stack
├── train/
│   ├── image/            # 30 PNGs (training inputs)
│   ├── label/            # 30 PNGs (training masks)
│   └── aug/              # 120 augmented samples from dataPrepare
└── test/                 # 30 PNGs (test inputs)
```

Only 30 training images — far too small for a network this deep, which is
exactly why on-the-fly data augmentation matters here.

### 2. `data/lung/` — Lung CT tumor segmentation (second dataset)

A second, *much larger* dataset so students can compare behavior with more
data. Source: the public ["Lung Tumor Segmentation" collection on Hugging
Face](https://huggingface.co/datasets/iraqigold/lung-cancer-segmentation-vision)
(originally from Kaggle). 2D CT slices of lungs with binary tumor masks.

**Task:** segment the *tumor region* inside the lung (white = tumor).

```
data/lung/
├── train/
│   ├── image/            # 1,832 PNGs, 256x256 grayscale (CT slices)
│   └── label/            # 1,832 binary masks ({0, 255}; 255 = tumor)
└── test/                 # 98 images (0.png ... 97.png) + *_gt.png ground truth
```

Notes:
- ~16% of slices have empty masks (that CT plane misses the tumor) — a
  realistic quirk worth discussing in class.
- Uses the same `image/` + `label/` convention as `membrane/`, so the very
  same training code works on both.

### Train on either dataset

```bash
python main.py --dataset membrane    # default (30 EM images)
python main.py --dataset lung        # lung CT tumor (1,832 slices)
```

Checkpoints are written to `unet_membrane.keras` / `unet_lung.keras`
respectively, and predictions to the matching `test/` folder.

---

## Suggested student exercises

1. **Visualize the generator.** Run `dataPrepare.py` and open
   `data/membrane/train/aug/` to confirm that the augmented image and
   mask are aligned.
2. **Train.** Run `main.py` for 5 epochs, then look at
   `data/membrane/test/*_predict.png`.
3. **Tune augmentation.** Disable augmentation (`data_gen_args = dict()`),
   retrain, and compare the result.
4. **Change the architecture.** Add dropout to earlier encoder blocks,
   swap `he_normal` for another initializer, or change the optimizer.
5. **Try multiclass.** Flip `flag_multi_class=True` in `trainGenerator`
   and see what changes in `adjustData` / `saveResult`.

---

## License

This folder inherits the MIT license from the original
[zhixuhao/unet](https://github.com/zhixuhao/unet) repository
(see `LICENSE`).