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

> **两个候选模型，只差初始化方式**：`models/model.json` 用公开数据预训练后再用主数据微调，
> `models/model_scratch.json` 只用主数据从零训练。两者结构、参数量、输入完全相同，
> 差别只在"起点"。交付哪个由留一会话交叉验证按预先声明的规则决定
> （本批合成数据上选了 scratch），⑤ 页可随时切换对比。见 §4 与 `docs/报告.md` 8.4。

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
python main.py --train         # 快速训练（留出 20% 会话），保存 models/model.json
```

有硬件时（设备已开机、接收器插在电脑上）：

```bash
python main.py --record --port COM7 --subject S01 --blocks 8 --block-sec 60
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
| ③ 数据集 | 选文件（自采 / 外部 / 模拟分别标注）→ 构建 8×14 窗口 → 导出 `data/dataset.csv` 与 HTML 报告；可导入公开数据集 |
| ④ 训练与评估 | **四种模式**：快速训练 / 留一会话交叉验证 / 最终模型 / 外部预训练；看 loss、混淆矩阵、CV 明细表 |
| ⑤ 实时推演 | 选择模型文件（`models/*.json`）→ 加载 → 实时 P(专注) 仪表与概率曲线（可调阈值） |
| ⑥ 说明 / AI 使用 | 项目说明与 AI 使用声明 |

## 3. 目录结构

```
NeuroFocus/
├── main.py                  ← 唯一的程序文件（协议/GUI/模型/训练全部在里面）
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                 ← 自采会话 CSV（每行 1 秒），文件名 = 被试_时间
│   │   ├── *_tasks.csv      ← 每段心算的报数（段号/题面文字/报数），仅作行为校验，不参与训练
│   │   └── *_quality.txt    ← 采集结束时的自动体检（两类幅度/alpha 对比 + 结论），不参与训练
│   ├── external/            ← 公开数据集导入后的标准会话（ext_性别_标签_部位_序号.csv）
│   ├── pilot/               ← 旧协议（屏幕出题）采到的会话，只作对照，不参与训练
│   ├── dataset.csv          ← 展开后的窗口数据集（每行一个 8×14 样本）
│   ├── dataset_report.html  ← 数据集质量报告（每份文件的收/弃样本数）
│   └── cv_scratch.json      ← 交叉验证明细（运行时生成）
├── models/
│   ├── model.json           ← 候选：外部预训练 → 主数据微调
│   ├── model_scratch.json   ← 候选：只用主数据从零训练
│   ├── pretrained_external.json ← 中间产物：只用公开数据训练的权重
│   └── model_int8.h         ← int8 量化权重的 C 头文件（MCU 侧可直接用）
└── docs/
    ├── 报告.md              ← 项目报告
    ├── 一页说明.md          ← 单页说明（示意图 / 模型论证 / 挑战）
    ├── 采集协议.md          ← 给操作者和受试者看的采集流程
    └── AI使用说明.md        ← AI 使用声明
```

## 4. 数据与训练协议（重要）

### 4.1 一个会话不够

`② 采集与标注` 点一次"开始"→"结束"产生一个会话文件（8 段 × 60 s = 8 分钟，约 344 个窗口）。
相邻窗口共享 7/8 秒信号，把同一会话切成训练/验证等于**泄漏**：这样得到的
accuracy 会接近 1.000，但换一个会话就崩（合成数据实测 0.710，真实数据更差）。
所以程序强制：

- 训练/验证按**会话文件**分组（`split_dataset(..., by="session")`）；
- 只有 1 个会话时，③ 页会明确提示「[警告] 只有 1 个分组……指标会明显偏乐观」，
  ④ 页的交叉验证会直接拒绝（`ValueError`），不会偷偷随机切分；
- 报告里的性能数字只能来自**留一会话交叉验证**（4 折 = 4 个会话，8 折 = 8 个会话）。

### 4.2 主数据集：1 被试 × 8 会话

| 项目 | 取值 |
| --- | --- |
| 被试 | 1 人（本人） |
| 会话数 | 8（每次重新贴电极、重新开始，间隔 ≥ 10 分钟） |
| 每会话 | 8 段 × 60 s（静息/心算交替，共 4+4 段） |
| 总时长 | 64 分钟，约 2 752 个窗口 |
| 评估 | 留一会话 8 折，报告 mean ± std |

每次采集结束看 `*_quality.txt`：结论为"告警"的会话**重采**，不进训练集。
报告只能声称"个人化、跨会话"，不能声称"跨被试"——外部数据也不能替你补这个缺口。

### 4.3 外部数据只做预训练

`③ 数据集 → 导入外部数据…` 可以把公开的 NeuroSky 数据集（160 个 10 s 试次，含注意力标签）
转成标准会话 CSV，落在 `data/external/`。它**不参与主数据训练**，只用于：

```bash
python main.py --import-external "path/to/dataset.zip"
python main.py --pretrain --epochs 200      # → models/pretrained_external.json
```

外部数据缺 `raw_std` / `raw_max` 两列，导入时补 0，训练时用训练集均值填充；
所以它的作用只是"给卷积核一个不坏的起点"，而不是替代主数据。

### 4.4 两阶段与两个模型

```bash
# 1) 主数据交叉验证（只出指标，不留权重）
python main.py --cv 8 --init scratch                 # → data/cv_scratch.json
python main.py --cv 8 --init external                # → data/cv_external.json（需先 --pretrain）

# 2) 用全部会话重训最终模型（轮数取 CV 最佳轮中位数，至少 3 轮）
python main.py --fit-final --init scratch            # → models/model_scratch.json
python main.py --fit-final --init external           # → models/model.json
```

选择规则在跑之前就定好：**按交叉验证均值选；两者差值 < 1 个标准差时选 scratch**
（奥卡姆剃刀，避免为 0.5 个点牺牲可解释性）。规则写进报告，选完不再改。
本批合成数据：scratch 0.798 ± 0.121、external 0.759 ± 0.164，差值 0.039 < 1σ，
按规则选 scratch（外部预训练在这批数据上没有收益，原因见 `docs/报告.md` 8.4）。

`data/raw/*_demo_sim.csv` 与当前 `models/*.json` 是 **合成数据** 训练的，
只用于证明链路可跑通，**不代表真实脑电的性能**。

> `data/pilot/` 里那份会话是**旧协议（屏幕出题 + 敲键盘）**采的，只作对照：
> 它单特征 AUC（`raw_max` 0.936）高得可疑，说明眼动/手部肌电混进了标签，
> 所以没有放进 `data/raw/`，也不会被 `--train` 读到。

## 5. 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--selftest` | 无硬件自检：协议解析、梯度数值校验、端到端小训练 |
| `--make-demo` | 生成 4 份合成会话 CSV 到 `data/raw/` |
| `--import-external ZIP_OR_DIR` | 导入公开数据集到 `data/external/` |
| `--pretrain` | 只用外部数据训练 → `models/pretrained_external.json`（`--epochs --folds`） |
| `--train [CSV…]` | 快速训练（留出 20% 会话），保存模型、导出数据集与 HTML 报告 |
| `--cv K` | 留一会话交叉验证 K 折，指标写 `data/cv_<init>.json` |
| `--fit-final` | 用全部会话训练最终模型（`--init scratch/external`） |
| `--by {session,subject}` | 分组方式：按会话（默认）或按被试 |
| `--init {scratch,external}` | 随机初始化或外部预训练初始化 |
| `--epochs --lr --folds` | 最大轮数 / 学习率 / 折数 |
| `--record` | 按协议采集真实数据（`--port --baud --subject --blocks --block-sec`） |
| `--export-tflite` | 把模型转成 int8 TFLite（需要额外安装 tensorflow） |

`tests/` 下是无需硬件的自动化检查（全部用模拟器/合成数据）：

| 脚本 | 检查内容 |
| --- | --- |
| `tests/gui_smoke.py` | 采集 → 构建数据集 → 训练 → 实时推演 的端到端联通 |
| `tests/capture_smoke.py` | 采集页闭眼默算协议、`*_tasks.csv`、采集体检 `quality_lines` |
| `tests/train_paths.py` | ④ 页两种入口（已构建的数据集 / 现场构建）都能训练 |
| `tests/train_modes.py` | ④ 页四种模式（快速/交叉验证/最终/预训练）都能跑通 |
| `tests/multisession.py` | 按会话划分不泄漏、单会话护栏、两阶段初始化、外部数据识别 |
| `tests/capture_layout.py` | 采集页在 1202×820 下不溢出，报告各控件请求宽度 |
| `tests/serial_probe.py` | 真机串口探测（需要设备与 COM 口） |

## 6. AI 使用声明

本项目在代码骨架、GUI 布局、文档起草与文字润色上使用了 AI 辅助（DeepSeek），
实验设计、数据采集、指标复现与结论判断由作者本人完成并负责。
逐项说明见 [`docs/AI使用说明.md`](docs/AI使用说明.md)。
