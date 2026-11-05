# -*- coding: utf-8 -*-
"""
download_data.py — Download and organize the Flickr8k dataset

Sources (no account required, direct download):
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip   (about 1.1 GB)
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip      (about 2.4 MB)

After running, the following will be created in the same directory as the script:
  data/Images/        8091 JPG images
  data/captions.csv   Columns: image, caption (5 captions per image, equivalent to Kaggle flickr8k's captions.txt)

Usage:
  python download_data.py
  python download_data.py --keep-zip   # Keep the zip files after extraction
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

URLS = {
    "Flickr8k_Dataset.zip": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
    "Flickr8k_text.zip": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
}


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[skip] Already exists: {dest}")
        return
    print(f"[download] {url}")
    print(f"          -> {dest} (file is about 1.1 GB, please be patient)")

    def _progress(blocks, block_size, total_size):
        done = blocks * block_size
        if total_size > 0:
            pct = min(100.0, done * 100.0 / total_size)
            sys.stdout.write(f"\r  Progress: {pct:5.1f}%  ({done / 1024 / 1024:7.1f} MB)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r  Downloaded: {done / 1024 / 1024:7.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def extract(zip_path: str, dest_dir: str) -> None:
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def build_captions_csv() -> str:
    """Convert Flickr8k.token.txt (format: image.jpg#N<TAB>caption) into captions.csv"""
    import pandas as pd

    token_path = None
    for root, _, files in os.walk(DATA_DIR):
        if "Flickr8k.token.txt" in files:
            token_path = os.path.join(root, "Flickr8k.token.txt")
            break
    if token_path is None:
        raise FileNotFoundError("Cannot find Flickr8k.token.txt. Please make sure Flickr8k_text.zip has been extracted.")

    rows = []
    with open(token_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_part, caption = line.split("\t", 1)
            image = img_part.split("#")[0].strip()
            rows.append({"image": image, "caption": caption.strip()})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(DATA_DIR, "captions.csv")
    df.to_csv(csv_path, index=False)
    print(f"[captions] Wrote {csv_path} ({len(df)} captions / {df['image'].nunique()} images)")
    return csv_path


def collect_images() -> None:
    """Gather all JPGs into data/Images/ (the original folder is named Flicker8k_Dataset)"""
    images_dir = os.path.join(DATA_DIR, "Images")
    os.makedirs(images_dir, exist_ok=True)

    candidates = []
    for root, _, files in os.walk(DATA_DIR):
        if os.path.basename(root) == "Images":
            continue
        for fn in files:
            if fn.lower().endswith(".jpg"):
                candidates.append(os.path.join(root, fn))

    if not candidates:
        print("[images] No JPG found. Please make sure Flickr8k_Dataset.zip has been extracted.")
        return

    moved = 0
    for src in candidates:
        dst = os.path.join(images_dir, os.path.basename(src))
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if not os.path.exists(dst):
            shutil.move(src, dst)
            moved += 1
    print(f"[images] Images gathered in {images_dir} ({moved} moved)")

    # Remove empty original folders
    for name in ("Flicker8k_Dataset", "Flickr8k_text", "Flickr8k_Dataset/Flicker8k_Dataset"):
        p = os.path.join(DATA_DIR, name)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and organize the Flickr8k dataset")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the zip files after extraction")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    for fn, url in URLS.items():
        download(url, os.path.join(DATA_DIR, fn))
        extract(os.path.join(DATA_DIR, fn), DATA_DIR)

    build_captions_csv()
    collect_images()

    for fn in URLS:
        zp = os.path.join(DATA_DIR, fn)
        if os.path.exists(zp) and not args.keep_zip:
            os.remove(zp)
            print(f"[clean] Removed {zp}")

    print("\nDone! Dataset location:")
    print(f"  Images: {os.path.join(DATA_DIR, 'Images')}")
    print(f"  Captions: {os.path.join(DATA_DIR, 'captions.csv')}")
    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()