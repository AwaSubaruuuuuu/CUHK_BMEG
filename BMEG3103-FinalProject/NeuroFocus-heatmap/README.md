# NeuroFocus 注意力热力图（Chrome 插件 + 本地推理服务）

看 B 站视频时，把「每秒的专注概率」按时间顺序排成一张点阵热力图：**集中 = 渐变的绿**、
**不集中 = 渐变的橙**。点任一格子就跳回那一段时间重看，并把那一段重新测一遍、刷新颜色。

> 这是一个**独立程序**，不改动 `NeuroFocus-main` 的任何文件：`server.py` 只是在运行时用
> `importlib` 把它的 `main.py` 当库加载，复用 `TGAMParser / SerialSource /
> frame_features / zscore / FocusCNN / rest_threshold`；模型则从本目录 `models/` 里挑。

---

## 目录结构

```
NeuroFocus-heatmap/
├─ run.bat                双击启动：补装依赖 + 起服务
├─ requirements.txt       numpy / pyserial
├─ server.py              本地推理服务（127.0.0.1:8765，SSE 推送分数）
├─ models/                可选模型：一个 json 一个模型，面板下拉里挑
├─ state/                 阈值落盘（每个模型一份；运行时生成，不入库）
├─ arduino/
│  └─ nfh_bridge/nfh_bridge.ino  Arduino 透传桥固件（原样转发 TGAM + 附加状态帧）
├─ extension/             Chrome 扩展（Manifest V3）
│  ├─ manifest.json
│  ├─ background.js       打开侧边栏 / 转发悬浮按钮的点击
│  ├─ content.js          B 站页面里的悬浮按钮 + 读取/跳转 video.currentTime
│  ├─ panel.html/.css/.js 侧边栏面板：概率、矩阵热力图、点击重测
├─ tools/                 开发期自检脚本（可删）
│  ├─ probe_sim.py        连上服务，打印 80 秒分数的分布（看是否真的在 0.1~0.9 之间动）
│  ├─ probe_msim.py       离线跑模拟源，网格搜索模拟参数
│  ├─ bridge_selftest.py  桥协议自检：混流下桥帧能摘出来、TGAMParser 不受干扰
│  └─ e2e_bridge.py       起临时服务，验证 /bridge 与 SSE bridge 事件
└─ README.md
```

> **放在仓库里的位置**：本目录位于 `BMEG3103-FinalProject/NeuroFocus-heatmap/`，与
> `BMEG3103-FinalProject/NeuroFocus/`（main 分支那份程序）并列。
> 服务启动时会自动按 `../NeuroFocus` → `../NeuroFocus-main` → `NF_DIR` 环境变量的顺序
> 去找 `main.py`（解析 TGAM、算特征、模型结构都在里面）；目录名不一样就用 `--nf-dir` 指过去。
>
> **模型本身不再从那边拿**：本目录 `models/` 里的 json 才是可选列表，启动时默认加载第一个能用的。
> ⚠️ 只认 **main 分支**那份（12 维 / 3061 参数）。14 维 two-stage 版的模型（`features=0`、
> 也没有 `zscore`）会在下拉里列出来但标成「不可用」，点不了。

## 为什么不是「纯插件」
扩展拿不到 TGAM 的蓝牙串口，也无法加载 `model.json` 跑那套分类流程，所以拆成两件：

- **本地服务** `server.py`：连头环（或回放 CSV / 模拟数据）→ 8 秒窗 → 模型 → 每秒一个概率，
  用 SSE 推给插件；
- **Chrome 插件**：只负责 UI（侧边栏面板）、B 站播放器控制、矩阵渲染、点击跳转与存储。

插件用 SSE（`EventSource`）而不是 WebSocket：内容脚本注入的代码会受页面 CSP 限制，而侧边栏
页面是扩展自己的页面，不受 B 站 CSP 影响，连接稳定、断线自动重连。

---

## 一、准备（只做一次）

1. 装依赖：`py -3 -m pip install -r requirements.txt`（numpy、pyserial）；或直接双击 `run.bat`，它会先补装再起服务。
2. 把模型 JSON 放进 `models/`（文件名就是模型身份，见后面「换模型」）。
3. 在 Chrome 里加载扩展，见「三、加载扩展」。

## 二、每次使用

双击 `run.bat`（会补装依赖并起服务），或者手动在窗口里跑：

```bat
cd "...\NeuroFocus-heatmap"

:: 接真头环
py -3 server.py --source serial --serial-port COM7

:: 不接硬件，用模拟数据把整条链路跑通（数据是编的，只用于演示）
py -3 server.py --source sim

:: 回放以前录的会话 CSV（可加速）
py -3 server.py --source replay --replay "..\NeuroFocus-main\data\raw\S01_20261110_160753.csv" --speed 2
```

服务起来以后：打开 B 站视频 → 右下角方格 → 侧边栏 → 面板里点「开始测量」。

浏览器里打开 `http://127.0.0.1:8765/health` 能返回 JSON 即正常。

`server.py` 默认按 `../NeuroFocus` → `../NeuroFocus-main` → `NF_DIR` 环境变量的顺序去找 `main.py`；
目录名不一样就用 `--nf-dir` 指过去。

常用参数：`--source serial|sim|replay|none`、`--serial-port COM7`、`--http-port 8765`、
`--models-dir models`（模型目录）、`--model <model.json>`（指定默认加载哪个）、
`--nf-dir <NeuroFocus>`、`--no-autostart`（启动时不自动开始采集）、
`--check-models`（只校验 `models/` 里每个文件然后退出，不启动服务）。

### 接硬件（TGAM / MindWave）：A 蓝牙、B 直连、C Arduino 三种都行

这个服务只做一件事：从**一条 COM 口**上读原始的 TGAM 字节流（`0xAA 0xAA …`，57600 8N1），
自己解析出 attention、8 个频段和 512 Hz 的 raw，再算特征送模型。所以硬件侧只需要
「把 TGAM 的 UART 变成 Windows 认的 COM 口」，转接方式三种都行：

| 方案 | 需要的件 | 接线 | 说明 |
| --- | --- | --- | --- |
| A 蓝牙（NeuroFocus 原方案） | MindWave 头环 + 蓝牙接收器（SPP） | 无 | 接收器插 USB → 设备管理器出现 `COMx` → 选它即可 |
| B USB-TTL（裸 TGAM 模块） | CH340 / CP2102 / FT232 转接板（**3.3V 电平**） | TGAM `VCC`→3.3V、`GND`→GND、`TXD`→转接板 `RXD` | 最便宜的直连方案，只收不发，`RXD` 可不接 |
| C Arduino 桥（本项目自带固件） | Uno/Nano + USB 线 | TGAM `TXD`→`D2`、`3.3V`→TGAM `VCC`、`GND`→`GND`；按键 `D4`→`GND`；LED 用板载 `D9`/`D13` | 字节**原样透传**，额外多出按键标记、伪迹、断链看门狗、片上 LED 反馈，见下一节 |

要点：

- **必须是完整字节流**：模型要 attention + 8 频段 + `raw_std`/`raw_max`，只给 eSense（attention/meditation）
  的两字段固件不够——raw（512 Hz，`0x02` 包）也得一起过来。
- 侧边栏里「数据源」选 **TGAM 串口**，下方会多出串口输入框：点 **刷新串口** 扫描，
  下拉里选 `COMx` 或直接手输，再点 **开始测量**。命令行走 `--source serial --serial-port COMx` 也一样。
- `GET http://127.0.0.1:8765/ports` 就是扫描结果（JSON），排错时先看它。
- 打不开端口时错误会原文回显（例如 `could not open port 'COM1': FileNotFoundError`）——
  通常是口选错、被 NeuroFocus 的 `main.py` 占着（同一 COM 口不能两进程同时用）、或接收器没插好。
- MindWave Mobile 2 只走 BLE，Windows 不会自动给普通 COM 口；要么用 SPP 版/老款 MindWave，
  要么自己加一个 BLE→虚拟串口 的桥。
- 电平：TGAM 是 3.3V；用 5V 的 Arduino/转接板时，`TXD` 方向（TGAM→板）一般能读，
  `RXD` 方向（板→TGAM）需要分压，**但本服务不需要往 TGAM 写任何东西**，别接就好。

### 方案 C：用 Arduino 做桥，比纯蓝牙多什么

固件在 `arduino/nfh_bridge/nfh_bridge.ino`（单文件，Uno/Nano/Leonardo/Mega/ESP32 都能编）。

**它做的事**：把 TGAM 的每个字节**原样**写到 USB 串口（所以 B 站插件照旧能收 raw + 大包），
同时每秒在流里插一帧自己的状态帧 `AA AA 0C …`（`0x0C` 是 NeuroFocus 的 `TGAMParser`
不认识的包长，会被安全丢掉，不会污染原解析）。

**蓝牙替代不了的四件事**：

1. **片上 LED 反馈**：模型判不专注时 Arduino 自己把 LED 调暗/闪，不经过电脑和浏览器。
   串口拥塞、浏览器卡顿、你切到别的标签页时，反馈还在。
2. **物理按键标记（毫秒级）**：手指按一下 `D4`，记一个 MARK。走蓝牙时你只能靠「手动在
   面板上点一下」，中间隔了整个系统链路（几百毫秒到几秒）；按键是硬件中断级的。
3. **断链看门狗**：TGAM 停发超过 2.5 秒，Arduino 立刻把 `link=0` 报上来，比「模型输出变得
   很奇怪」早几十秒——电极脱落、电池没电、蓝牙掉线都能第一时间显示。
4. **接触/运动伪迹标记**：按 raw 的标准差/峰值标出「这一秒别信」（`artifact`），
   面板会把那一格描一圈橙边，导出 CSV 里有 `noisy` 列。这些统计本来就能在主机算，
   但放在 Arduino 上等于多了一个**不依赖主机**的旁路校验。

顺带还有：裸 TGAM 模块的 3.3V 供电与电平隔离、以及留给 IMU/SD 卡的扩展余量。

**烧写**：

1. Arduino IDE 打开 `arduino/nfh_bridge/nfh_bridge.ino`，板子选 Uno/Nano，端口选上，
   直接上传（Mega/Leonardo/ESP32 把文件顶部 `USE_SERIAL1` 改成 `1`）。
2. 烧完再接线：TGAM `TXD`→`D2`，`3.3V`→TGAM `VCC`，`GND` 共地；按键一端 `D4`、一端 `GND`。
3. 回到面板：「数据源」选 **TGAM 串口**，刷新串口选 Arduino 那个 `COMx`，开始测量。
   面板里会多出一条 **Arduino 桥** 状态（TGAM link / signal / raw_std / 坏帧），
   按 `D4` 会在矩阵上方留下一个 `⚑ 12:34` 标签，点它能跳回那一段重测；
   `LED` 按钮是给固件发 `#LED:1` / `#LED:0`，`PING` 发 `#PING` 验通。

**命令**（电脑→Arduino，`\n` 结尾，必须以 `#` 开头）：`#PING`、`#LED:1`、`#LED:0`、
`#THR:0..100`、`#BLINK`。阈值变化时服务会自动下发 `#THR:`，固件据此决定 LED 什么时候亮。

**风险**：Uno 的软串口跑 57600 会丢字节，状态帧里的 `坏帧` 计数一直涨就说明撑不住——
换 Mega/Leonardo（硬件 `Serial1`，把 `USE_SERIAL1` 设 1）或 ESP32。
另外 **同一时刻只有一个程序能占这个 COM 口**：NeuroFocus 的 `main.py` 和本服务不能同时开。

## 三、加载扩展

1. Chrome 地址栏打开 `chrome://extensions`，右上角打开「开发者模式」。
2. 「加载已解压的扩展程序」→ 选 `extension` 目录。
3. 打开任意视频页 —— **B 站** `https://www.bilibili.com/video/BV...` 或
   **YouTube** `https://www.youtube.com/watch?v=...`（Shorts 不支持，见「已知限制」），
   右下角会出现一个绿色方格悬浮按钮。
4. 点悬浮按钮 → 侧边栏打开「NeuroFocus 注意力热力图」。

> 需要 Chrome 116+（`chrome.sidePanel`）。改了 `content.js` / `manifest.json` 后要回
> `chrome://extensions` 点一次刷新，并 **F5 重新加载页面**；只改侧边栏文件则关掉侧边栏再打开即可。

## 四、使用

1. 视频正常播放，侧边栏「数据源」选好后点 **开始测量**。
2. 前 8 秒是在攒窗口，还没有分数；之后每秒出一个概率，矩阵里对应当前播放位置的那一格会亮起蓝框。
3. **校准（重要）**：坐下、闭眼、别动，点 **校准**，面板上会出现 30 秒倒计时和进度条。
   结束后阈值从该模型自带的默认值换成你自己的静息分位值，绿/橙的分界才准。结果**按模型存盘**
   （换模型不会串、重启还在）。阈值只影响颜色与「专注/静息」判定，不改模型。
4. **看热力图**：一格 = 一段时间（默认 10 秒，可在「格宽」里切 5/15/30 秒，**改格宽不会丢数据**，
   因为原始分数是按「秒」存的）。绿越亮越集中，橙越亮越涣散，深灰 = 还没测过。
5. **点格子重看**：点任意一格 → 视频跳到该格开头（+0.2 s）并自动播放 → 这一轮新测的分数覆盖
   同一格 → 颜色刷新；测过一次的格子上会有个小圆点，hover 显示「第 N 轮（重测过）」和旧数据。
6. 往回拖动进度条超过 3 秒，会自动算作「新一轮」，避免上一轮的分数污染这一段。
7. **按键标记（Arduino 桥）**：听到没看懂的地方，按一下 Arduino 上的 `D4` 键 → 矩阵上方出现
   `⚑ 12:34` 标签 → 点它跳回那一段，顺便把那一格清空重测。
8. **导出 CSV**（列：`video_id, bin_start_s, bin_end_s, mean_p, samples, pass, noisy`；
   `noisy` 是这一格里被判为伪迹的秒数），可直接进 Excel 画图。

### 延迟补偿

从「你看到画面」到「模型出分」之间有 1~3 秒的滞后（窗口 8 秒 + 蓝牙 + 推理）。默认把分数记在
`当前播放时间 - 2.0s`；如果颜色整体偏晚（该绿的格子亮在更后面），把滑块往大调；偏早就调小。

### 换模型（换人 / 重训之后）

训练不在插件里做。用 `NeuroFocus/main.py` 的 `--record` 采数据、`--train` 出新的 `model.json`，然后：

1. 把文件复制进 `heatmap/models/`，取名 **`subjectNN-YYYY-MM-DD.json`**。下拉里显示的就是
   「文件名 + json 里的 `trained_at`」，同名会覆盖（视为同一模型、旧的校准值继续沿用）。
2. 侧边栏「模型与校准」里点 **刷新** → 在下拉里挑 → 立刻生效。测量中下拉是灰的，要先停止。
3. 第一次用某个模型：角标显示「未校准 · 默认 x.xx」，这个 x.xx 是**那个模型自己** `model.json`
   里的 `threshold.default`，不是全局 0.5。坐下闭眼点一次校准。
4. 换走再换回：回到那个模型自己存过的校准值；它没校过就用它自己的默认值。

一句话规则：**当前模型的校准值 > 该模型的 `threshold.default`**，两者都没有才退到 0.5。
落盘在 `state/thresholds.json`，重启还在。

命令行也能查：`py -3 server.py --check-models` 逐个校验 `models/`，不可用的会写出原因
（维数不对、JSON 坏了……）。

### 数据存在哪

`chrome.storage.local`，key 是 `nfh:v2:<模型名>:<bvid>`：**同一个视频在不同模型下是两份数据**，
互不覆盖；旧的 `nfh:v1:<bvid>` 保留但不再读。标记（`⚑`）也存在这一份里，所以换模型时标记
跟着换。点「清空本视频」只清「当前模型 + 当前视频」。

---

## 五、已知限制

- **模拟数据不是测量**：`--source sim` 造的是落在模型训练分布内的合成样本（由 `DemoSource` 生成），
  只为演示链路与配色；任何结论都必须用 `serial` 或 `replay` 的真实数据。
- **单通道、被试相关**：模型用的是 FP1 单导 + 8 秒窗，跨被试/跨天漂移明显。谁测就用谁的模型
  （见「换模型」），同一人也建议隔天重新校准；必要时用 `NeuroFocus/main.py --train` 重训。
- **判定不是医学结论**：这是课程项目里的「专注度代理量」，不是注意力诊断。
- B 站前端改版可能让 `content.js` 里的选择器失效；目前依次尝试
  `#bilibili-player video` / `.bpx-player-container video` / `.bpx-player-video-wrap video` / `video`。
- YouTube 只在普通 `/watch?v=` 页面注入：**Shorts（`/shorts/`）不支持**（按需求排除），
  首页、频道页、嵌入播放器也不注入。播放器选择器依次尝试
  `#movie_player video.html5-main-video` / `#movie_player video` / `video.html5-main-video` / `video`。
- YouTube 贴片广告期间 `video.currentTime` 走的是广告时间轴，这一段的分数会被记到视频进度上；
  看到黄色广告条时最好按一下「停止测量」。YouTube 还有「暂停即存档」的行为，暂停时不记分（设计如此）。
- 只在桌面版 Chrome 测过；Safari/Firefox 不支持 `sidePanel`。
- 端口 8765 被占用时用 `--http-port` 换端口，并把 `extension/panel.js` 顶部的 `API` 一起改掉。
- `--source sim` 时面板上的「Arduino 桥」状态是**合成的**（标了「模拟」），按键/伪迹事件由代码造出来
  只为让界面可调；真桥需要烧 `nfh_bridge.ino` 并选串口源。
- Arduino 桥的 LED 反馈是**本机 1 秒级**的近似（状态帧 1 Hz），不是逐窗实时；按键 MARK 本身是毫秒级的。

## 六、排错

| 现象 | 原因 / 处理 |
| --- | --- |
| 侧边栏一直「本地服务未连接」 | 服务没起：双击 `run.bat`，或前台 `py -3 server.py`；浏览器打开 `http://127.0.0.1:8765/health` 看有没有 JSON |
| 面板一直「等待开始」但设备管理器里有 COMx | 那个口没被认出来（不是蓝牙 SPP 口）：把接收器重新配对；或手动 `py -3 server.py --source serial --serial-port COMx` 看报什么错 |
| 点了「开始测量」很久没有分数 | 口开了但对端没发数据：头环没开机 / 电量低 / 波特率不是 57600 |
| 测量中突然停了 | 断连保护：串口不再来数据就自动停。检查线/蓝牙，然后重新点「开始测量」 |
| 悬浮按钮不出现 | 页面没刷新（扩展装好后要 F5）；或不在 `/video/`（B 站）或 `/watch?v=`（YouTube）页面；Shorts 本来就不注入 |
| 侧边栏显示「未找到视频播放器」 | 页面上没有 `<video>`（首次加载慢、或 YouTube 是嵌入页）；等两秒会自动重试 |
| 概率一直是 100% 或一直 0% | 数据源不对（`sim` 曾因量纲问题恒为 1.0，已修）；真机则是信号质量差，看提示里的 signal 码 |
| 提示「信号质量差，这一秒不计入」 | TGAM 的 signal 码异常（接触不良/电量低），这一秒会跳过而不是记 0 |
| 点格子没反应 | 视频页需刷新；或播放器在广告/付费内容上不可跳转 |
| `ModuleNotFoundError: numpy` | `py -3 -m pip install numpy` |
| 想换模型 | 把 json 放进 `models/` → 面板点「刷新」→ 下拉里选（见「换模型」）；命令行用 `--check-models` 先验证 |
| 下拉里某个模型是灰的 | 它和当前 `main.py` 不匹配（多半是 14 维/坏 JSON），鼠标悬停看原因 |

## 七、开发期自检

```bat
py -3 -m py_compile server.py
node --check extension\panel.js

:: 只校验 models/ 里每个模型能不能用（不启动服务、不占串口）
py -3 server.py --nf-dir "..\NeuroFocus" --check-models

:: 服务用 --source sim 起来后：看 80 秒分数的分布是否真的在 0.1~0.9 之间
py -3 tools\probe_sim.py
:: 离线网格搜索模拟参数（不启动服务）
py -3 tools\probe_msim.py
:: Arduino 桥协议自检（不需要板子：混流下桥帧/坏校验/TGAMParser 互不干扰）
py -3 tools\bridge_selftest.py
:: 端到端：临时起一个 sim 服务，验证 /bridge 与 SSE bridge 事件
py -3 tools\e2e_bridge.py
```
