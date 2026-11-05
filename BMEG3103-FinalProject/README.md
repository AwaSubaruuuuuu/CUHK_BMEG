# BMEG3103 Final Project — Neuro-AI 脑波交互系统

BMEG3103 *Big Data in HealthCare* 的期末项目。

用**单通道 EEG** 实时估计「注意力」，注意力高到一定程度就自动触发一个动作，注意力掉下来就停止——把脑波直接变成交互信号。

## 系统设计

```
MindWave / TGAM 脑波模块（蓝牙，57600 8N1）
        │  ~512 raw/秒 + 每秒约 1 个大包
        ▼
   Arduino UNO（arduino/Arduino.ino）
        │  只做一行一行转发，不在板子上解析
        ▼
   笔记本 / 树莓派 / Jetson Nano
        ├─ 解析：小包取原始波，大包取信号质量 / 注意力 / 放松度 / 8 段功率
        ├─ 预处理：按 ~512 Hz 插值到 200 Hz、带通、滑窗
        ├─ 1D CNN 推理（约 3000 参数，输入最近 8 s）
        └─ 决策：分数 0–100，基线 3 s 移动平均，
                   连续 2 s > 75 → 触发动作
```

- **输入**：最近 8 秒的脑波数据（滑动窗口，重叠切分）
- **输出**：0 – 100 的注意力分数
- **触发规则**：以 3 秒移动平均为基线做归一化；分数**连续 2 秒 > 75** 才触发，避免单帧噪声误触
- **模型**：1D CNN，参数量约 3K，目标是能跑在嵌入式板子上
- **验收指标**：留出测试集（hold-out）准确率 **≥ 85%**

## 采到的数据长什么样（已用真实包验证）

TGAM 的串口上有两种包，**每秒约 512 个小包 + 1 个大包**：

| 包 | 字节数 | 结构 | 内容 |
| --- | --- | --- | --- |
| 小包 | 9 | `AA AA 04 80 02 <hi> <lo> <chk>` | 一个原始波采样点，约 512 Hz |
| 大包 | 38 | `AA AA 20 <32 字节 payload> <chk>` | 原始波 + 信号质量 + 注意力 + 放松度 + 8 段功率 |

大包 payload 各字节含义（已用 `tools/UartAssist.cfg` 里抓到的真实包逐字节核对过）：

| 偏移 | 字节 | 含义 | 实测样例 |
| --- | --- | --- | --- |
| `[0]` | 1 | raw 低字节 | `0x02` |
| `[1]` | 1 | raw 高字节 | `0x00` |
| `[2]` | 1 | 未知/保留（三次都是 `0x83`） | `0x83` |
| `[3]` | 1 | **信号质量** | `0x18` = 24 |
| `[4..27]` | 24 | **8 段功率**，每 3 字节大端，顺序 delta / theta / low-alpha / high-alpha / low-beta / high-beta / low-gamma / mid-gamma | 见下 |
| `[28]` | 1 | 未知/保留（三次都是 `0x04`） | `0x04` |
| `[29]` | 1 | **注意力 Attention** 0–100 | `0x5D` = 93 |
| `[30]` | 1 | **放松度 Meditation** 0–100 | `0x05` |
| `[31]` | 1 | 未知/保留 | `0x40` |

实测一帧的 8 段功率（从 `payload[4]` 起、每 3 字节大端读，最后一组只用了 2 字节）：

```
delta       629342      low-beta     15387
theta        81168      high-beta     9755
low-alpha     6427      low-gamma    11623
high-alpha   34768      mid-gamma     1671
```

校验：`chk == (~sum(payload)) & 0xFF`，小包则是 `(~sum(payload)) & 0xFF`（payload = 04 之后的 4 字节，即 `80 02 hi lo`）。丢包率 10% 以下不影响结果。

> ⚠️ **`arduino/Arduino.ino` 里有一处偏移错误要修**：第 111 行 `feature[8] = payloadData[29]` 取的是**注意力**值，但代码注释称它为信号质量；真正的信号质量在 `payloadData[3]`。`payloadData[29..31]` 其实是 `attention, meditation, ?`。正因如此，那一段从 `j=4; j<28` 循环出来的「8 段功率」只有 7 段是对的，最后一格是错的。`mindwave_uart.ino` 则是对的（`[1]`→信号质量、`[29]`→注意力、`[31]`→放松度），可以直接照它改。

## 目录内容

| 路径 | 说明 |
| --- | --- |
| `arduino/Arduino.ino` | Arduino 端采集固件：从 TGAM 读数据帧、校验和、打包转发到串口（**偏移有误，见上**） |
| `arduino/mindwave_uart.ino` | 最小可用的 MindWave UART 读取示例，偏移正确，调试接线时先用它 |
| `tools/UartAssist.cfg` | UartAssist 串口助手配置（含三帧真实抓包，可用于离线校验解析器） |
| `tools/configure.dat` | 配套调试软件配置 |
| `requirements.txt` | 本项目依赖 |

> 串口调试工具本身是 `.exe`，体积大且不适合放仓库（`.gitignore` 已排除），需要时从课程资料里取。

## 运行环境

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
```

## 硬件接线要点

1. TGAM 模块的 `VCC` 接 Arduino 3.3V/5V（按模块规格），`GND` 共地
2. `TXD` → Arduino 的软串口 RX
3. 波特率 **57600**，8 数据位、无校验、1 停止位
4. 数据包以 `0xAA 0xAA` 开头，第三个字节是 payload 长度（小包 `0x04`，大包 `0x20`）
5. 校验和：`(~sum(payload)) & 0xFF`
6. 佩戴：耳夹夹在**左耳垂根部肉的位置**，头带金属电极贴在**左额头眉毛上方**；先用湿纸巾擦一下去油脂。信号质量要等它降到接近 0 才可信

接好先烧 `mindwave_uart.ino`、开串口助手确认能收到 `AA AA 20` 帧，再烧 `Arduino.ino` 走完整链路。

## 进度

项目处于**准备阶段**：硬件链路已通、数据格式已核对清楚，注意力模型与触发逻辑尚未实现。

## 评分构成（参考）

AI 模型 19%、演示 6%、单页说明 5%。课程整体：Assignment 10% + Group Project & Presentation 40% + Final Exam 50%；项目展示 2026-12-01，期末考 2026-12-03。
