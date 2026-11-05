# CNN + LSTM Image Captioning (Flickr8k) — Local Execution Version

Adapted from [CHAITANYAI0/Image-Captioning-with-CNN-LSTM-and-ResNet-GRU-Flickr8k](https://github.com/CHAITANYAI0/Image-Captioning-with-CNN-LSTM-and-ResNet-GRU-Flickr8k)'s `CNN_LSTM.ipynb`,
reorganizing the original absolute paths on Mac and the notebook environment into a Python script project that can be run directly on Windows / any local machine.

## Model Architecture

- **CNN Feature Extraction**: Pretrained `DenseNet201` (with the final classification layer removed), outputting a 1920-dimensional feature vector for each image
- **Text Generation**: `Embedding → LSTM(256)`; image features go through `Dense(256)` and are reshaped, then merged with the text sequence
- **Training**: Predict the next word one at a time using `startseq ... endseq` (teacher forcing), with `categorical_crossentropy` + Adam
- **Inference**: Greedy decoding to generate the caption word by word
- **Evaluation**: BLEU-4 score

## Project Structure


```
CNNLSTM/
├── download_data.py     # 1. 下載並整理 Flickr8k 資料集（約 1.1 GB）
├── train.py             # 2. 特徵抽取 + 訓練 CNN+LSTM 模型
├── predict.py           # 3. 對任意本機圖片產生描述
├── requirements.txt
├── data/                # 自動產生
│   ├── Images/          # 8091 張圖片
│   ├── captions.csv     # 每張圖 5 句描述
│   └── features_DenseNet.pkl   # 特徵快取（第二次訓練免重抽）
└── output/              # 自動產生
    ├── model.keras      # 最佳模型
    ├── tokenizer.pkl    # tokenizer + max_length + vocab_size
    ├── loss.png         # 訓練 loss 曲線
    └── predictions.txt  # （--save 時）推論結果
```

## 執行步驟

```bat
cd D:\Training\CUHK_Sep-2026\class_DL3_TF_DataAugmentation_Keras_UNET\CNNLSTM

:: 0. Install packages (recommended to use your existing TF environment)
pip install -r requirements.txt

:: 1. Download the Flickr8k dataset (about 1.1 GB, only needs to be run once)
python download_data.py

:: 2. Train (with GPU, epochs 30~50 is recommended; on CPU, try 10~15 first)
python train.py --epochs 15 --batch-size 64

:: 3. Generate captions for your own images
python predict.py your_image.jpg
python predict.py some_folder --save
```

Main Adjustments Compared to the Original Notebook
Original Notebook	Local Version
Hardcoded /Users/chaitanya/... absolute paths	Always relative paths (script's directory)
Manual dataset download from Kaggle	download_data.py auto-downloads + converts Flickr8k.token.txt into captions.csv
Feature extraction re-run every time	Cached as features_DenseNet.pkl
model.h5 (newer Keras recommends .keras)	Saved as output/model.keras
Tokenizer only kept in memory	Saved as tokenizer.pkl, so predict.py can run inference independently
Text cleaning x.replace("[^A-Za-z]","") (literal replacement, does nothing)	Changed to real regex cleaning
plot_model requires installing graphviz separately	Removed; replaced with saving caption_model.summary()
BLEU computed after overwriting the caption to get the reference (wrong order)	Predict first, then take the reference for computation

Common Issues
Slow download: The dataset is 1.1 GB; download_data.py shows progress. You can also manually download
Flickr8k_Dataset.zip and
Flickr8k_text.zip,
place them into data/, then run python download_data.py again (it will skip existing zips).

First training takes longer: DenseNet201 features must first be extracted for 8091 images (a few minutes on GPU, about 30~60 minutes on CPU); afterwards the pkl cache is used.

Out of memory: Lower --batch-size to 32.

nltk BLEU requires a tokenizer: On the first run it will auto-download punkt; if offline, you can ignore it — training is unaffected.
