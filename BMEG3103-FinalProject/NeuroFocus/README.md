# NeuroFocus · 单通道脑电专注度解码系统

BMEG3103 Final Project  ——  用 **1 个通道（FP1）的脑电** 判断"此刻是否在专注"，
并把判断结果做成一个可实时演示的桌面程序。

```
原始 EEG（TGAM @512Hz 小包 + 1Hz 大包）
        │  TGAMParser：帧同步 / 校验和 / 断帧恢复
        ▼
  1 Hz 特征行（14 维：attention、meditation、signal、8 段功率、raw_std、raw_max…）
        │  滑动窗口 = 最近 8 秒
        ▼
  8 × 14 的"小图"  ──►  3 层 1-D CNN（3 157 参数）──►  P(专注) ∈ [0,1]
        │
        ▼
  GUI 实时曲线 / 混淆矩阵 / 边缘部署文件
```

> 采集时被试**全程闭眼**：静息段放松，心算段默默"从 1000 连续减 7"，
> 不看屏幕、不敲键盘、不出声 —— 眼动与手部肌电因此不会混进标签。
> 操作者只负责在段末问一句"算到几了"，把报数记进 `*_tasks.csv`（仅作行为证据，不参与训练）。

## 1. 快速开始

```bash
# 依赖：Python 3.10+，numpy、pyserial（tkinter 为 Python 自带）
pip install -r requirements.txt

python main.py            # 打开图形界面（默认连接 COM7）
```

没有硬件也能把整条链路跑通（数据为**合成**，见第 4 节）：

```bash
python main.py --selftest      # 协议解析 + 数值梯度 + 端到端小训练
python main.py --make-demo     # 生成 4 份模拟会话 CSV
python main.py --train         # 训练并保存 models/model.json
```

有硬件时（设备已开机、接收器插在电脑上）：

```bash
python main.py --record --port COM7 --subject S01 --blocks 6 --block-sec 60
```

程序会用提示音引导"静息 60 s / 心算 60 s"交替采集：**两种状态都闭眼**，
心算段被试默算"从 1000 连续减 7"，界面只给操作者看，被试全程不看屏幕、不碰键盘。
结果写入 `data/raw/`，每段报数写入同名 `*_tasks.csv`；采集一结束还会自动做一次体检，
把两类状态的幅度与 alpha 对比结果写进 `*_quality.txt`——眨眼/咬牙这类眼手伪迹会被点名，
免得把"睁着眼睛打字"当成"专注"训练进模型。
推荐直接用 `python main.py` 的「② 采集与标注」页采集；命令行 `--record` 同样只做文字与
提示音引导，并自动写入每段的心算记录。详细佩戴与采集说明见 [`docs/采集协议.md`](docs/采集协议.md)。

## 2. 界面（6 个页签）

| 页签 | 作用 |
| --- | --- |
| ① 设备与信号 | 选串口或模拟器；看 512 Hz 原始波形、Signal 质量、Attention/Meditation 曲线 |
| ② 采集与标注 | 按协议引导采集（全程闭眼，心算段默算 1000−7），自动打标签并写 CSV（这是**生成数据集**的地方） |
| ③ 数据集 | 选文件 → 构建 8×14 窗口 → 导出 `data/dataset.csv` 与 HTML 报告 |
| ④ 训练与评估 | 训练 3 层 1-D CNN，看 loss 曲线、混淆矩阵、Accuracy/F1/AUC，保存模型、导出部署文件 |
| ⑤ 实时推演 | 加载模型，实时显示 P(专注) 仪表与概率曲线（可调阈值） |
| ⑥ 说明 / AI 使用 | 项目说明与 AI 使用声明 |

## 3. 目录结构

```
NeuroFocus/
├── main.py                  ← 唯一的程序文件（协议/GUI/模型/训练全部在里面）
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                 ← 原始会话 CSV（每行 1 秒），文件名 = 被试_时间
│   │   ├── *_tasks.csv      ← 每段心算的报数（段号/题面文字/报数），仅作行为校验，不参与训练
│   │   └── *_quality.txt    ← 采集结束时的自动体检（两类幅度/alpha 对比 + 结论），不参与训练
│   ├── pilot/               ← 旧协议（屏幕出题）采到的会话，只作对照，不参与训练
│   ├── dataset.csv          ← 展开后的窗口数据集（每行一个 8×14 样本）
│   └── dataset_report.html  ← 数据集质量报告（每份文件的收/弃样本数）
├── models/
│   ├── model.json           ← 训练好的权重 + 标准化参数 + 指标
│   └── model_int8.h         ← int8 量化权重的 C 头文件（MCU 侧可直接用）
└── docs/
    ├── 报告.md              ← 项目报告
    ├── 一页说明.md          ← 单页说明（示意图 / 模型论证 / 挑战）
    ├── 采集协议.md          ← 给操作者和受试者看的采集流程
    └── AI使用说明.md        ← AI 使用声明
```

## 4. 数据说明（重要）

`data/raw/*_demo_sim.csv` 与当前 `models/model.json` 是由 **合成数据** 训练出来的，
只用于证明"采集 → 数据集 → 训练 → 实时推演"这条链路可以跑通，
**不代表真实脑电的性能**。报告中的真实指标必须用你自己采集的会话重新训练后得到。

真实采集 → 重新训练：

```bash
python main.py            # ② 采集与标注页：全程闭眼，心算段默算 1000 − 7
#   至少 3 位被试、每人 ≥3 段静息 + 3 段心算
python main.py --train --epochs 300
```

> `data/pilot/` 里那份会话是**旧协议（屏幕出题 + 敲键盘）**采的，只作对照：
> 它单特征 AUC（`raw_max` 0.936）高得可疑，说明眼动/手部肌电混进了标签，
> 所以没有放进 `data/raw/`，也不会被 `--train` 读到。

## 5. 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--selftest` | 无硬件自检：协议解析、梯度数值校验、端到端小训练 |
| `--make-demo` | 生成 4 份合成会话 CSV 到 `data/raw/` |
| `--train [CSV…]` | 训练（不给文件则用 `data/raw/*.csv`），保存模型、导出数据集与 HTML 报告 |
| `--record` | 按协议采集真实数据（`--port --baud --subject --blocks --block-sec`） |
| `--export-tflite` | 把模型转成 int8 TFLite（需要额外安装 tensorflow） |

`tests/` 下是无需硬件的自动化检查（全部用模拟器/合成数据）：

| 脚本 | 检查内容 |
| --- | --- |
| `tests/gui_smoke.py` | 采集 → 构建数据集 → 训练 → 实时推演 的端到端联通 |
| `tests/capture_smoke.py` | 采集页闭眼默算协议、`*_tasks.csv`、采集体检 `quality_lines` |
| `tests/train_paths.py` | ④ 页两种入口（已构建的数据集 / 现场构建）都能训练 |
| `tests/capture_layout.py` | 采集页在 1202×820 下不溢出，报告各控件请求宽度 |
| `tests/serial_probe.py` | 真机串口探测（需要设备与 COM 口） |

## 6. AI 使用声明

本项目在代码骨架、GUI 布局、文档起草与文字润色上使用了 AI 辅助（DeepSeek），
实验设计、数据采集、指标复现与结论判断由作者本人完成并负责。
逐项说明见 [`docs/AI使用说明.md`](docs/AI使用说明.md)。
