# CUHK_BMEG

CUHK 生物医学工程课程 **project 代码仓库**。

本仓库只放两个课程的 project 实现和运行环境（代码 + 依赖说明），**不放课程学习资料、数据集、模型权重与虚拟环境**——这些由 `.gitignore` 排除，保存在本地。

| 课程 | Project | 目录 |
| --- | --- | --- |
| BMEG3103 Big Data in HealthCare | Neuro-AI 脑波交互系统（单通道 EEG 注意力检测） | [`BMEG3103-FinalProject/`](BMEG3103-FinalProject/) |
| BMEG3330 Neuroengineering | SSVEP BCI 赛车 / 迷宫控制 | [`BMEG3330-EEG-Project/`](BMEG3330-EEG-Project/) |

> BMEG3330 仍在准备阶段；BMEG3103 已交付可运行的程序 [`BMEG3103-FinalProject/NeuroFocus/`](BMEG3103-FinalProject/NeuroFocus/)。

## 目录结构

```
CUHK_BMEG/
├── BMEG3103-FinalProject/        # Neuro-AI 脑波交互系统
│   ├── arduino/                  # Arduino 固件（MindWave / TGAM 串口读取、UART 转发）
│   ├── tools/                    # 串口调试配置（UartAssist.cfg、configure.dat）
│   ├── NeuroFocus/               # 最终交付：单文件 GUI（采集 / 数据集 / 1-D CNN / 实时推演）
│   ├── requirements.txt
│   └── README.md
├── BMEG3330-EEG-Project/         # SSVEP BCI 赛车竞赛
│   ├── cca_maze.py               # 迷宫版主程序（含运行时调参）
│   ├── virtual_control.py        # 无迷宫的虚格动画版
│   ├── requirements.txt
│   └── README.md
├── data/README.md                # 本地数据与模型权重的放置约定（内容不进 Git）
├── requirements.txt              # 仓库级公共依赖
└── .gitignore
```

## 环境搭建

不需要 Docker，用本地 Python 虚拟环境即可，推荐 **Python 3.10 – 3.12**。

```bash
git clone https://github.com/AwaSubaruuuuuu/CUHK_BMEG.git
cd CUHK_BMEG

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt                        # 公共依赖
pip install -r BMEG3330-EEG-Project/requirements.txt   # 只做 3330 时
```

更新代码只需要 `git pull`；依赖没变就不用重装。

## BMEG3103 — Neuro-AI 脑波交互系统

用单通道 EEG 做**实时注意力检测**，注意力分数越高说明越专注；分数低时不动作，持续专注时触发动作。

- 注意力分数范围 **0 – 100**
- 基线用 **3 秒移动平均**，分数**连续 2 秒 > 75** 才判定为触发
- 模型：**1D CNN，3061 参数**，输入最近 **8 秒**的脑波数据，留出测试集准确率目标 **≥ 85%**
- 硬件链路：MindWave / TGAM 脑波模块 → Arduino（UART）→ 笔记本 / 树莓派 / Jetson Nano

`arduino/` 下是采集固件：`Arduino.ino` 读取 TGAM 串口并转发，`mindwave_uart.ino` 是最小可用的 UART 读取示例。`tools/` 是配套的串口调试工具配置。

> 模型与触发逻辑已实现：`NeuroFocus/main.py` 一个文件里包含串口解析、采集与标注、数据集构建、3061 参数 1-D CNN 训练与实时推演界面（只用 numpy + pyserial）。
>
> 真实数据（1 位被试、2 次会话、474 个窗口，按会话留一）**accuracy 0.890 ± 0.018、F1 0.883、AUC 0.967**，
> 已超过 brief 的 85% 要求；同一验证集上 brief 的 Option A 基线只有 0.616。详见
> [`BMEG3103-FinalProject/README.md`](BMEG3103-FinalProject/README.md) 与 `docs/报告.md` 8.3 节。
>
> 交付版就在 `main`：12 维特征、真实采集数据。仓库另有 `two-stage-14f` 分支，保留了另一条
> 实现路线——14 维特征 + 公开数据预训练 / 留一会话交叉验证 / 四种训练模式，指标来自合成数据，
> 用来做消融与对照。两条路线的取舍见 [`PATCH_NOTES.md`](PATCH_NOTES.md)。

## BMEG3330 — SSVEP BCI 赛车竞赛

用 **SSVEP**（稳态视觉诱发电位）把「盯着哪个闪烁方块」解码成「往哪个方向走」。

- 4 个目标频率 **8 / 11 / 13 / 15 Hz**，分别映射 **上 / 下 / 左 / 右**
- 解码：**FBCCA**（滤波器组典型相关分析），三个子带权重 `[1.0, 0.42, 0.25]`
  - 带通 `[6–45]` / `[14–45]` / `[22–45]` Hz，4 阶 Butterworth
- 采样率 `FS = 200` Hz（须与 OpenBCI Ganglion 一致），窗口 `WIN_TIME = 3.0` s，滑动步长 `SLIDE_STEP = 0.2` s
- 判定阈值 `THRESHOLD = 0.135`，连续 `BUFFER_SIZE = 2` 帧一致才真正移动
- 采集板 OpenBCI Ganglion，`SERIAL_PORT = "COM6"`（按自己电脑改）

| 文件 | 说明 |
| --- | --- |
| `cca_maze.py` | 迷宫版。运行时可用 `[` / `]` 调窗口长度（0.5–5.0 s，步长 0.5），`-` / `=` 调阈值（步长 0.005），现场调参不用改代码 |
| `virtual_control.py` | 简化版，没有迷宫，只有四方向闪烁方块和移动方块，适合先验证解码链路 |

两个脚本都需要真实硬件（OpenBCI Ganglion + `brainflow` + `pygame`），没有设备时无法运行。

## 使用约定

- 提交前 `git status` 检查，确认没有误加 `.h5` / `.csv` / `.venv` / 课程 PDF
- 不要手工复制 `xxx - 副本.py`，用分支或提交记录代替
- 提交信息建议格式：`BMEG3103: 简述改动` / `BMEG3330: 简述改动`
- 课程讲义、数据集版权归课程与老师，不放进仓库
