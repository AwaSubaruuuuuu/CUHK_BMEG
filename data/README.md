# 数据与模型权重说明

本仓库（`CUHK_BMEG`）只保存**源码**，不含数据集、模型权重、虚拟环境。
体积大的文件全部由 `.gitignore` 排除，以免超出 GitHub 的限制（单文件 100 MB、仓库建议 < 1 GB）。

这份清单说明：每份"缺失"的资源叫什么、多大、从哪来、应该放在哪个目录。

> 大多数情况下**不需要下载任何东西**。只想读代码、看 notebook、跑 sklearn 的小例子，`git clone` 之后直接开工即可。

---

## 一、开箱即用（无需下载）

下面这些代码自带数据或由 `keras.datasets` 首次运行时自动下载，克隆下来就能跑：

| 代码 | 数据来源 |
| --- | --- |
| `BMEG3103/CNN.py` | `keras.datasets.mnist`（自动下载，约 11 MB） |
| `BMEG3103/W2_Source_1/CNNText/` | `keras.datasets.reuters`（自动下载） |
| `BMEG3103/W6_Source_2/CNN_LSTM/` | 运行同目录 `download_data.py` 自动下载 Flickr8k（约 1.1 GB） |
| `BMEG3330/` | 主要为仿真，无外部数据依赖（需另装 `pygame`、`brainflow`） |

第一次运行会自动把数据缓存到 `~/.keras/datasets/`，所以**不需要**手动找文件。

---

## 二、小体积 CSV（几十 KB ~ 几百 KB，需要手动放）

这些文件被 `.gitignore` 中的 `*.csv` 规则排除，但都很小。缺了会报 `FileNotFoundError`。

| 文件 | 大小 | 应放在 | 需要它的脚本 |
| --- | --- | --- | --- |
| `bill_authentication.csv` | 45 KB | `W2_Source_1/SVM/`、`W2_Source_2/`、`W2_Source_3/SVM/` | SVM 示例 |
| `Wholesale customers data.csv` | 15 KB | `W2_Source_2/` | 肘部法（`elbow.py`） |
| `clustering.csv` | 26 KB | `W2_Source_2/` | KMeans |
| `Classified Data.csv` | 190 KB | `W2_Source_3/KNN/` | KNN 示例 |
| `Social_Network_Ads.csv` | 11 KB | `W2_Source_3/logistics/` | 逻辑回归 |
| `diabetes.csv` | 24 KB | `W2_Source_3/runnall/` | 综合练习 |
| `Breast_Cancer_Dataset.csv` | 122 KB | `W2_Source_4/` | 梯度下降 |
| `cancer.csv` | 122 KB | `W6_Source_1/features/` | 特征相关性分析 |
| `training.csv` | < 1 KB | `W2_Source_3/Bayes/` | 高斯贝叶斯 |

这些数据如果同学之间互相要，直接从各自电脑的 `D:\Learning\BMEG3103` 对应子目录复制即可。

---

## 三、大体积文件（需要从外部获取）

| 文件 | 大小 | 来源 | 应放在 | 说明 |
| --- | --- | --- | --- | --- |
| `vgg16_1.keras`<br>`vgg16_1.weights.h5` | 各约 1.0 GB | 自己训练产生的权重 | `BMEG3103/W6_Source_1/tf/` | ImageNet 预训练 VGG16 微调后的产物。**没有也不影响读代码**；要复现就重新训练，或向仓库作者索取 |
| `model.weights.h5` | 约 1.0 GB | 训练缓存 | `BMEG3103/W6_Source_1/tf/` | 训练中断产生的临时权重，不需要下载，删掉即可 |
| `train.csv` | 59 MB | Kaggle | `BMEG3103/W6_Source_1/features/` | 代码里是 `pd.read_csv(..., nrows=10000)`，只读前 1 万行；文件大是因为列数多 |
| `Standard Customer Data.csv` | 59 MB | Kaggle **Kaggle Customer Segmentation (Mall Customers)** 一类数据集 | `BMEG3103/W6_Source_1/VT/` | 代码只读前 4 万行 |
| `train data credit card.csv` | 14 MB | Kaggle **Credit Card Fraud Detection** | `BMEG3103/W6_Source_1/VT/` | 极不平衡二分类示例 |
| `ansur_male.csv` | 2 MB | Kaggle **ANSUR II**（美军人体测量） | `BMEG3103/W6_Source_1/VT/` | 虚拟试衣示例 |
| `lung_dataset.zip` | 77 MB | Kaggle 肺部影像分割数据集 | `BMEG3103/W6_Source_2/U_NET_implementation/data/` | 解压后得到 `lung/train`、`lung/test` |
| `final_CNN_model.h5`<br>`final_DNN_model.h5` | 5 MB / 7 MB | 自己训练产生 | `BMEG3103/` | Final Project 的训练产物 |
| `Flickr8k_Dataset.zip` | 约 1.1 GB | GitHub Releases（公开直链） | `BMEG3103/W6_Source_2/CNN_LSTM/` | **不用手动下**，`python download_data.py` 会自动下载并解压 |

> **版权提示**：上表 Kaggle 数据集遵循各自的许可协议，可能不允许二次分发。请勿把数据文件上传到本仓库或公开网盘；需要时让同学自己去 Kaggle 下载。医疗影像类数据（`lung_dataset`）尤其要注意。

---

## 四、虚拟环境

不要复制别人的 `.venv`：虚拟环境与操作系统、Python 版本绑定，Windows 上建的在 Linux 上不能用，反之亦然。

换机器时永远是这样重建：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

`BMEG3330/CCA_maze_0723_bigger.py` 与 `Virtual_control_CCA.py` 另外需要：

```bash
pip install pygame brainflow
```
