# BMEG3330 EEG Project — SSVEP BCI 赛车竞赛

BMEG3330 *Neuroengineering* 的课程项目（BCI 赛车竞赛）。

用 **SSVEP**（稳态视觉诱发电位）：屏幕上四个方块分别以不同频率闪烁，用户盯着哪个方块，枕区 EEG 就会出现对应频率的节律响应；解码出「在看哪个」就等于拿到「上 / 下 / 左 / 右」指令。

## 解码管线

```
OpenBCI Ganglion（FS = 200 Hz，COM6）
        │  brainflow 实时取数
        ▼
滑动缓冲（WIN_TIME = 3.0 s，SLIDE_STEP = 0.2 s）
        │
        ▼
滤波器组：4 阶 Butterworth 带通
  子带1 [6–45] Hz   子带2 [14–45] Hz   子带3 [22–45] Hz
        │
        ▼
FBCCA：各子带对 4 个目标频率分别算 CCA 相关系数，
       权重 [1.0, 0.42, 0.25] 加权求和
        │
        ▼
取最大者作为这一帧的判定；相关系数 < THRESHOLD = 0.135 视为无效帧
        │
        ▼
BUFFER_SIZE = 2：连续 2 帧判定一致才真正发出移动指令
        │
        ▼
Pygame 渲染（迷宫版 / 虚格动画版）
```

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 目标频率 | 8 / 11 / 13 / 15 Hz | 分别映射 上 / 下 / 左 / 右 |
| `FS` | 200 Hz | 必须与 Ganglion 采样率一致，不要改 |
| `WIN_TIME` | 3.0 s | 相关系数窗口长度 |
| `SLIDE_STEP` | 0.2 s | 多久更新一次判定 |
| `THRESHOLD` | 0.135 | 低于此值不出指令 |
| `BUFFER_SIZE` | 2 | 连续同帧数，抑制抖动 |
| `FB_WEIGHTS` | [1.0, 0.42, 0.25] | 三个滤波器组的权重 |
| `SERIAL_PORT` | `"COM6"` | Windows 串口号，按实际改（设备管理器里看） |

## 两个程序

### `cca_maze.py` — 迷宫版

带迷宫地图的主程序，小车在网格里按 SSVEP 指令移动。额外提供**运行时调参**，现场试不用改代码重跑：

| 按键 | 作用 |
| --- | --- |
| `[` / `]` | 缩短 / 延长窗口长度 `WIN_TIME`（0.5 – 5.0 s，步长 0.5） |
| `-` / `=` | 降低 / 提高判定阈值 `THRESHOLD`（步长 0.005） |
| `Esc` | 退出 |

调参线程和主线程用 `settings_lock` 保护，改参数时不会读到半更新的状态。

### `virtual_control.py` — 简化版

没有迷宫，只有四个方向的闪烁方块和中间一个方块：判定对了方块就往那个方向动。**建议先用它调通硬件和解码链路**，确认能稳定控制后再换迷宫版。

## 运行环境

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
python virtual_control.py         # 先用简化版验证
python cca_maze.py                # 再玩迷宫版
```

依赖只有 `numpy`、`scipy`、`scikit-learn`、`pygame`、`brainflow`。**没有 OpenBCI Ganglion 设备时程序无法运行**（一启动就会去连板子）。

## 硬件注意

- OpenBCI Ganglion 装好驱动后，在设备管理器确认串口号，改 `SERIAL_PORT`
- Ganglion 采样率固定 200 Hz，与 `FS` 一致；若换用其他板子，`FS` 必须同步修改，否则频率映射会全部错位
- 电极位置建议：枕区（O1 / O2），参考和地线按 Ganglion 默认贴法
- 闪烁频率不能随便改：8 / 11 / 13 / 15 Hz 是经过验证、和屏幕刷新率配合良好的组合，改成别的值要重新做 CCA 参考信号

## 进度

项目处于**准备阶段**：解码与渲染框架已就位，竞赛策略和最终提交尚未完成。

## 评分构成（参考）

项目整体占 40%：Proposal 3%、Mid Report 7%、Mid Report 之后的 Progress 5%、Project Demo 10%、Final Report 10%、Peer review 5%。Final Report 截止 2026-12-06，竞赛日 2026-12-03。
