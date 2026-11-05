# -*- coding: utf-8 -*-
"""
predict.py — Use the trained CNN+LSTM model to generate captions for any local image (image captioning)

Usage:
  python predict.py photo.jpg              # Single image
  python predict.py folder_with_images     # Entire folder (recursively find .jpg/.jpeg/.png)
  python predict.py a.jpg b.jpg c.jpg      # Multiple images

Requirement: Run train.py first; output/ must contain model.keras and tokenizer.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np
from PIL import Image

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "tokenizer.pkl")
IMG_SIZE = 224
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Cache the feature extractor (module level, to avoid reloading DenseNet201 repeatedly)
_feature_extractor = None


def get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        base = DenseNet201()
        _feature_extractor = Model(inputs=base.input, outputs=base.layers[-2].output)
    return _feature_extractor


def extract_feature(image_path: str) -> np.ndarray:
    """Extract DenseNet201 features for a single image, returns (1, 1920)"""
    fe = get_feature_extractor()
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return fe.predict(img, verbose=0)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, feature, tokenizer, max_length) -> str:
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption if caption else "(model produced no caption)"


def gather_images(paths):
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in names:
                    if n.lower().endswith(IMG_EXTS):
                        files.append(os.path.join(root, n))
        else:
            print(f"[warn] Not found: {p}")
    return files


def main():
    parser = argparse.ArgumentParser(description="CNN+LSTM image caption generator")
    parser.add_argument("images", nargs="+", help="Image file or folder path")
    parser.add_argument("--save", action="store_true", help="Also save results to output/predictions.txt")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Cannot find {MODEL_PATH}. Please run train.py to complete training first.")
    if not os.path.exists(TOKENIZER_PATH):
        sys.exit(f"Cannot find {TOKENIZER_PATH}. Please run train.py to complete training first.")

    print("[load] Loading model and tokenizer ...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        meta = pickle.load(f)
    tokenizer = meta["tokenizer"]
    max_length = meta["max_length"]

    files = gather_images(args.images)
    if not files:
        sys.exit("No images found")

    print(f"[predict] {len(files)} image(s) in total\n")
    lines = []
    for i, fp in enumerate(files, 1):
        try:
            feature = extract_feature(fp)
        except Exception as e:
            print(f"{i}/{len(files)}  {fp}\n  [error] Failed to read: {e}\n")
            continue
        caption = predict_caption(model, feature, tokenizer, max_length)
        print(f"{i}/{len(files)}  {fp}\n  -> {caption}\n")
        lines.append(f"{fp}\t{caption}")

    if args.save:
        out = os.path.join(OUTPUT_DIR, "predictions.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[save] Results saved to {out}")


if __name__ == "__main__":
    main()