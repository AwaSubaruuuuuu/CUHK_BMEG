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

校验：`chk == (~sum(payload)) & 0xFF`；小包同理，payload 是 `0x04` 之后的 4 个字节（`80 02 hi lo`）。丢包率 10% 以下不影响结果。

## 已修正的解析错误

原始课程资料里的两个 `.ino` 都有偏移错误，**已于 2026-11-05 修正并离线验证**（用上面三帧真实抓包逐字节比对）。

### `Arduino.ino`

| 问题 | 修正前 | 修正后 |
| --- | --- | --- |
| 信号质量取错字节 | `feature[8] = payloadData[29];`（实为**注意力**，注释却写「信号质量」） | `signalquality = payloadData[3];` |
| 放松度缺失 | 从未读取 | `meditation = payloadData[30];` |
| 输出把注意力重复了两遍 | 最后一格 = 注意力，注意力本身没单独输出 | 每行输出 `signal, attention, meditation, 6 段功率` |
| 8 段功率只有 7 段对 | 循环 `j = 4..27` 拼出的第 8 段实际混入了 `payloadData[28]`（保留位，实测恒为 `0x04`） | 8 段按 `4 + 3k` 单独取，转发前 6 段；第 8 段需上位机配合时再放开 |

### `mindwave_uart.ino`

| 问题 | 修正前 | 修正后 |
| --- | --- | --- |
| 信号质量取错字节 | `signalquality = payloadData[1];`（raw 采样点的高字节） | `signalquality = payloadData[3];` |
| 放松度取错字节 | `meditation = payloadData[31];`（保留位） | `meditation = payloadData[30];` |

### 离线验证结果

三帧真实包喂进修正后的索引，校验和全部通过，字段值与抓包完全一致：

| 包 | signal | attention | meditation | delta | theta | low-alpha | high-alpha | low-beta | high-beta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 24 | 93 | 5 | 629342 | 81168 | 6427 | 34768 | 15387 | 9755 |
| 2 | 24 | 91 | 5 | 2462430 | 243910 | 10200 | 140098 | 63812 | 24485 |
| 3 | 24 | 81 | 5 | 1057014 | 614704 | 49578 | 128516 | 60637 | 100842 |

> 这三帧的 `signal` 恒为 24、`meditation` 恒为 5，说明当时电极接触一般，抓包时并没有真正在做注意力实验。**要拿到能训练的数据，必须重新采集。**

## 目录内容

| 路径 | 说明 |
| --- | --- |
| `arduino/Arduino.ino` | 主固件：解析 TGAM 大包，按 `signal,attention,meditation,6 段功率` 每行输出 |
| `arduino/mindwave_uart.ino` | 最小 UART 读取示例，只打印 SignalQuality / Attention / Meditation，先用它确认链路通 |
| `tools/UartAssist.cfg` | UartAssist 串口助手配置（含三帧真实抓包，可用于离线校验解析器） |
| `tools/configure.dat` | 配套调试软件配置 |
| `requirements.txt` | 本项目依赖 |

> 串口调试工具本身是 `.exe`，体积大且不适合放仓库（`.gitignore` 已排除），需要时从课程资料里取。

## 烧录与接线注意事项

1. **烧录前必须拔掉 TGAM 的 TXD。** Arduino 的硬件串口 `Serial` 在这个固件里既用来读 TGAM、又用来往电脑输出，TGAM 送来的 `AA AA` 会打断 avrdude 的上传流程，典型报错：`avrdude: stk500_recv(): programmer is not responding`。
2. 更稳妥的接法是把 TGAM 接到**软串口**（如 `SoftwareSerial` 的 D2/D3），硬件串口只对电脑，两者互不干扰——代价是要额外占两个引脚，且软串口跑 57600 在部分板子上不稳。
3. 波特率固定 **57600**，8N1。
4. 输出是**逗号分隔**的纯数字行，直接 `Serial` 抓下来存成 `.csv` 就能进 pandas；`mid_gamma` 未输出，需要时改 `passFeature(..., 6)` 为 `8`。


## 运行环境

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
```

## 硬件接线要点

1. TGAM 模块的 `VCC` 接 Arduino 3.3V/5V（按模块规格），`GND` 共地
2. `TXD` → Arduino 的 RX（走硬件串口则见上方「烧录与接线注意事项」，烧录时先拔掉）
3. 波特率 **57600**，8 数据位、无校验、1 停止位
4. 数据包以 `0xAA 0xAA` 开头，第三个字节是 payload 长度（小包 `0x04`，大包 `0x20`）
5. 校验和：`(~sum(payload)) & 0xFF`
6. 佩戴：耳夹夹在**左耳垂根部肉的位置**，头带金属电极贴在**左额头眉毛上方**；先用湿纸巾擦一下去油脂。信号质量要等它降到接近 0 才可信
7. 蓝牙接收器（`HR-` 开头）插 USB 免驱；win7 需装 COM 驱动

接好先烧 `mindwave_uart.ino`、开串口助手确认能收到 `AA AA 20` 帧，再烧 `Arduino.ino` 走完整链路。


## 进度

项目处于**准备阶段**：硬件链路已通、数据格式已核对清楚，注意力模型与触发逻辑尚未实现。

## 评分构成（参考）

AI 模型 19%、演示 6%、单页说明 5%。课程整体：Assignment 10% + Group Project & Presentation 40% + Final Exam 50%；项目展示 2026-12-01，期末考 2026-12-03。
