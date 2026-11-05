# BMEG3103 Final Project — Neuro-AI 脑波交互系统

BMEG3103 *Big Data in HealthCare* 的期末项目。

用**单通道 EEG** 实时估计「注意力」，注意力高到一定程度就自动触发一个动作，注意力掉下来就停止——把脑波直接变成交互信号。

## 系统设计

```
MindWave / TGAM 脑波模块
        │  UART（默认 57600 8N1）
        ▼
   Arduino UNO  ──────────►  笔记本 / 树莓派 / Jetson Nano
  （arduino/）                 ├─ 预处理：重采样、带通、滑窗
                               ├─ 1D CNN 推理（约 3000 参数，输入最近 8 s）
                               └─ 决策：分数 0–100，基线 3 s 移动平均，
                                          连续 2 s > 75 → 触发动作
```

- **输入**：最近 8 秒的脑波数据（滑动窗口，重叠切分）
- **输出**：0 – 100 的注意力分数
- **触发规则**：以 3 秒移动平均为基线做归一化；分数**连续 2 秒 > 75** 才触发，避免单帧噪声误触
- **模型**：1D CNN，参数量约 3K，目标是能跑在嵌入式板子上
- **验收指标**：留出测试集（hold-out）准确率 **≥ 85%**

## 目录内容

| 路径 | 说明 |
| --- | --- |
| `arduino/Arduino.ino` | Arduino 端采集固件：从 TGAM 读数据帧、校验和、打包转发到串口 |
| `arduino/mindwave_uart.ino` | 最小可用的 MindWave UART 读取示例，调试接线时先用它 |
| `tools/UartAssist.cfg` | UartAssist 串口助手配置（波特率、显示格式等） |
| `tools/configure.dat` | 配套调试软件配置 |
| `attention_cnn.py` | **模型脚手架**，见下方说明 |
| `requirements.txt` | 本项目依赖 |

> 串口调试工具本身是 `.exe`，体积大且不适合放仓库（`.gitignore` 已排除），需要时从课程资料里取。

## `attention_cnn.py` 说明

这个文件目前是 **MNIST 手写数字分类的教学示例**（先训练一个全连接 DNN，再训练一个 CNN，保存成 `final_DNN_model.h5` / `final_CNN_model.h5`），**不是**本项目的注意力模型。保留它是为了照它的结构改成 1D 卷积的脑波模型：

- 把 `Conv2D + (28,28)` 输入换成 `Conv1D + (time_steps, channels)`
- 输出层从 10 类 softmax 换成单神经元 + sigmoid（或回归到 0–100）
- 脚本里用到的 `predict_classes()` 在新版 TensorFlow 已被移除，要改成：

  ```python
  pred = np.argmax(model.predict(x), axis=1)
  ```

## 运行环境

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
```

想直接试跑脚手架：

```bash
python attention_cnn.py           # 首次运行会自动下载 MNIST，并生成两个 .h5
```

训练产物（`*.h5`）已在 `.gitignore` 中，不会误提交。

## 硬件接线要点

1. TGAM 模块的 `VCC` 接 Arduino 3.3V/5V（按模块规格），`GND` 共地
2. `TXD` → Arduino 的软串口 RX
3. 波特率 **57600**，8 数据位、无校验、1 停止位
4. 数据包以 `0xAA 0xAA` 开头，倒数第二个字节是校验和：`~(payload 各字节求和) & 0xFF`

接好先烧 `mindwave_uart.ino`、开串口助手确认能收到 `0xAA 0xAA` 帧，再烧 `Arduino.ino` 走完整链路。

## 进度

项目处于**准备阶段**：硬件链路和脚手架已就位，注意力模型与触发逻辑尚未实现。

## 评分构成（参考）

AI 模型 19%、演示 6%、单页说明 5%。课程整体：Assignment 10% + Group Project & Presentation 40% + Final Exam 50%；项目展示 2026-12-01，期末考 2026-12-03。
