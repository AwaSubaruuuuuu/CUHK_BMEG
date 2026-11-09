# 数据与模型权重

本仓库只保存 **project 代码**，不含数据集、模型权重、虚拟环境，也不含课程学习资料。
体积大的文件全部由 `.gitignore` 排除，以免超出 GitHub 的限制（单文件 100 MB、仓库建议 < 1 GB）。

实际上，两个 project 的**代码本身不依赖任何需要手动下载的数据**：

- **BMEG3103 Final Project**：注意力模型要的是实时脑波流，由 Arduino + MindWave/TGAM 现场采集，没有固定数据集。
- **BMEG3330 EEG Project**：要的是 OpenBCI Ganglion 的实时 EEG 流，也没有固定数据集。

---

## 一、训练产物（本地生成，不进仓库）

| 文件 | 大小 | 产生方式 | 位置 |
| --- | --- | --- | --- |
| `*.h5` / `*.keras` / `*.ckpt` | 不定 | 自己训练脑波模型时产生 | 项目目录下 |
| `model.json` / `model_int8.h` | ~80 KB | `python NeuroFocus/main.py --train` 生成 | `NeuroFocus/models/` |

这些都是可以随时重新生成的产物，已经全部被 `.gitignore` 排除，**不要提交**。

## 二、采集到的原始 EEG（本地保存，不进仓库）

| 文件 | 说明 | 建议位置 |
| --- | --- | --- |
| `*.csv` / `*.txt` 脑波原始数据 | 从串口录制下来的 TGAM 数据帧或 Ganglion 数据 | `BMEG3103-FinalProject/recordings/` |
| `*.set` / `*.fdt` / `*.mat` | EEGLAB / MATLAB 格式的采集记录 | `BMEG3330-EEG-Project/recordings/` |

`*.csv`、`*.mat` 已在 `.gitignore` 中，`recordings/` 目录按需自己建即可。

## 三、课程资料（版权归课程与老师，不要上传）

Project brief、讲义 PDF、实验说明书、演示视频、待机软件 `.exe` 等全部不放进仓库——
一方面 `.gitignore` 已经排除 `*.pdf` / `*.docx` / `*.ppt*` / `*.exe` / `*.jpg` / `*.mp4`，
另一方面这些材料的版权不属于我们，公开仓库里分发可能有问题。

需要时从课程平台（Blackboard / Moodle）或同学处获取，放在本地对应目录即可。

---

## 虚拟环境

不要复制别人的 `.venv`：虚拟环境与操作系统、Python 版本绑定，Windows 上建的在 Linux 上不能用，反之亦然。

换机器时永远是这样重建：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt                         # 公共依赖
pip install -r BMEG3330-EEG-Project/requirements.txt    # 只做 3330 时
```
