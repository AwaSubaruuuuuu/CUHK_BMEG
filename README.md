# CUHK_BMEG

CUHK 生物医学工程相关课程的**代码文件**归档仓库。

本仓库只保存可版本控制的源码与笔记（`.py` / `.ipynb` / `.m` / `.md` / Docker 配置等），
不包含模型权重、数据集、虚拟环境等大体积产物——这些由 `.gitignore` 排除。

## 快速上手

本仓库**不需要 Docker**，用本地 Python 虚拟环境即可。推荐 Python **3.10 – 3.12**。

```bash
git clone https://github.com/AwaSubaruuuuuu/CUHK_BMEG.git
cd CUHK_BMEG

python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
jupyter lab                     # 打开对应课程的 .ipynb
```

之后更新代码只需要：

```bash
git pull
pip install -r requirements.txt   # 仅当依赖有变化时
```

- 所有依赖见 [`requirements.txt`](requirements.txt)；`pygame`、`brainflow`、`MindSpore` 属于可选项，按需单独安装。
- 数据集与模型权重不进仓库，需要的文件及其放置位置见 [`data/README.md`](data/README.md)。
- 大多数示例（MNIST、Reuters、Flickr8k）首次运行会自动下载数据，无需手动准备。
- `BMEG3103/Dockerfile`、`compose.yaml` 是课程最初的环境容器配置，**可选**，不安装 Docker 也能正常使用。

## 目录结构

```
CUHK_BMEG/
├── BMEG3103/       # 机器学习 / 深度学习课程代码
├── BMEG3330/       # 循证医学相关：CCA 迷宫、控制与 MATLAB 实验
├── data/README.md  # 数据集与模型权重的来源、体积、放置位置
└── requirements.txt
```

### BMEG3103

| 目录 | 内容 |
| --- | --- |
| `10.13_Class_source/` | LSTM 示例（`lstm1.py`, `lstm2.py`, `Test.ipynb`） |
| `W2_Source_1/CNNText/` | 一维 CNN 文本分类 |
| `W2_Source_1/SVM/` | SVM 示例 |
| `W2_Source_2/` | KMeans、肘部法、随机森林 |
| `W2_Source_3/` | 高斯贝叶斯、KNN、逻辑回归、SVM、随机森林 |
| `W2_Source_4/` | 梯度下降 |
| `W6_Source_1/` | TensorFlow：特征工程、数据增强、函数式 API、迁移学习 |
| `W6_Source_2/CNN_LSTM/` | CNN-LSTM 完整实现（数据下载 / 训练 / 预测） |
| `W6_Source_2/U_NET_implementation/` | U-Net 图像分割完整实现 |
| `Dockerfile`, `compose.yaml` | 课程环境容器配置（可选，本地 venv 即可替代） |

### BMEG3330

| 文件 | 内容 |
| --- | --- |
| `CCA_maze_0723_bigger.py` | CCA 迷宫仿真主程序 |
| `Virtual_control_CCA.py` | 虚拟控制实验 |
| `MATLAB/` | EMD、滤波器设计、回归演示脚本 |

## 使用约定

- 提交前 `git status` 检查，确认没有误加 `.h5`、`.venv`、数据文件
- 不要手工复制 `xxx - 副本.py`；用分支或提交记录代替
- 提交信息建议格式：`BMEG3103: 简述改动`

## 说明

本仓库是从 `D:\Learning\BMEG3103`、`D:\Learning\BMEG3330` 中抽取代码建立的分类归档，
原始目录中的数据集、模型权重、虚拟环境仍保留在原处。
