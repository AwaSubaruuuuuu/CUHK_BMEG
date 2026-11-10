# -*- coding: utf-8 -*-
"""
================================================================================
 NeuroFocus —— 单通道 FP1 脑电专注度解码系统
 BMEG3103 Final Project
================================================================================

 硬件   : TGAM 单通道干电极脑电套件（采集电极贴左前额 FP1，参考/接地夹左耳垂）
 连接   : 蓝牙接收器枚举为串口（默认 COM7），57600 8N1
 数据   : TGAM 大包 1 Hz（Signal / Attention / Meditation / 8 段 EEG Power）
          + 小包 512 Hz（Raw 原始波形，用于伪迹检测与波形显示）
 模型   : 1-D CNN（3 个卷积层，参数量 ≈3k），输出 0~1 专注概率
 标签   : 由采集协议给定的任务段（静息=0 / 心算=1），并做行为校验

--------------------------------------------------------------------------------
 AI 使用声明（必读）
--------------------------------------------------------------------------------
 本项目在以下环节使用了 AI 辅助（DeepSeek 编码助手）：
   1. 代码骨架、GUI 布局与本文档的起草；
   2. 数据清洗规则与特征工程的初步方案；
   3. 报告与说明文档的行文润色。
 下列内容由作者本人负责并已逐项核对 / 需本人完成：
   - 实验协议的制定与实际数据采集（受试者、佩戴、心算任务）；
   - 所有指标的复现实验与结论判断；
   - 现场演示与答辩。
 详见 docs/AI使用说明.md。

 运行:  python main.py
 依赖:  numpy, pyserial（若未安装，程序会自动尝试同目录 vendor/）
================================================================================
"""

import os
import sys
import csv
import json
import math
import time
import queue
import random
import glob
import shutil
import argparse
import importlib
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict

# ---- 依赖自举：优先使用仓库内 vendor/ 目录 -------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# 允许把依赖放在 main.py 同级 vendor/ 或上一级 vendor/（离线安装时使用）
# vendor/cp314 这类"版本专属"子目录优先，避免混用别的 CPython 版本的 .pyd
_PYTAG = "cp%d%d" % sys.version_info[:2]
_VENDOR_DIRS = []
for _root in (_HERE, os.path.dirname(_HERE)):
    for _sub in (os.path.join("vendor", _PYTAG), "vendor"):
        _cand = os.path.join(_root, _sub)
        if os.path.isdir(_cand) and _cand not in _VENDOR_DIRS:
            _VENDOR_DIRS.append(_cand)
for _cand in _VENDOR_DIRS:
    if _cand not in sys.path:
        sys.path.append(_cand)


def _import_dep(name):
    """按顺序尝试各 vendor 目录；numpy 的 .pyd 与 CPython 版本绑死，
    只有真能加载成功的那份才算数（否则静默回退到"无 numpy"）。"""
    try:
        return importlib.import_module(name)
    except Exception:
        pass
    for d in _VENDOR_DIRS:
        if not os.path.isdir(os.path.join(d, name)):
            continue
        for k in [k for k in list(sys.modules) if k == name or k.startswith(name + ".")]:
            del sys.modules[k]
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


np = _import_dep("numpy")
serial = _import_dep("serial")
try:
    from serial.tools import list_ports
except Exception:  # pragma: no cover
    list_ports = None

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


APP_TITLE = "NeuroFocus · 单通道脑电专注度解码系统"
APP_VERSION = "1.0"

DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 57600

BANDS = ["delta", "theta", "lowAlpha", "highAlpha", "lowBeta", "highBeta", "lowGamma", "midGamma"]
BAND_LABELS = {
    "delta": "Delta 1-4Hz", "theta": "Theta 4-8Hz", "lowAlpha": "LowAlpha 8-10Hz",
    "highAlpha": "HighAlpha 10-13Hz", "lowBeta": "LowBeta 13-17Hz", "highBeta": "HighBeta 18-30Hz",
    "lowGamma": "LowGamma 30-40Hz", "midGamma": "MidGamma 40-50Hz",
}

DATA_DIR = os.path.join(_HERE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODEL_DIR = os.path.join(_HERE, "models")
DOC_DIR = os.path.join(_HERE, "docs")
for _d in (DATA_DIR, RAW_DIR, MODEL_DIR, DOC_DIR):
    os.makedirs(_d, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model.json")

# 采集与建模参数
WINDOW = 8              # 时间窗长度（秒）—— 与 Project Brief 的 "last 8 attention values" 一致
SETTLE_SECONDS = 10     # 每段开始丢弃的佩戴稳定期
CALIB_SECONDS = 50      # 每个会话开头用于「本人基线校准」的静息秒数（部署时同一协议）
CALIB_LIVE_SECONDS = 30 # 实时推演时请求被试保持静息的校准秒数

# 模型输入特征（12 维）。
# 刻意排除 signal 与 meditation，两者仍保留在原始 CSV 里供质检/说明，但不进模型：
#   * signal 是接触质量标记而不是脑电：S01 全程恒 0 → 该列 sd 落到 1e-6 下界，
#     换到 S02（8 行 signal=26）时标准化 z 值会冲到 1e5，直接把网络打爆；
#   * meditation 是厂商派生量、与 attention 高度共线，且跨会话漂移不稳（单特征 AUC 0.42）。
# 实测（见 docs/报告.md 消融表）：去掉这两维后跨会话准确率提升约 1–5 个百分点。
FEATURE_NAMES = (["attention"]
                 + ["log_" + b for b in BANDS]
                 + ["attention_diff", "raw_std", "raw_max"])
FEATURE_INDEX = {n: i for i, n in enumerate(FEATURE_NAMES)}
BAND_FEATURES = ["log_" + b for b in BANDS]

# TGAM 数据格式说明中列出的「耳机没戴好」码：signal 落在这些值上都代表接触不良，必须剔除。
BAD_SIGNAL_CODES = frozenset({29, 54, 55, 56, 80, 81, 82, 107, 200})

# 标准化后的截断上限。静息段某些列几乎不动，除以很小的 sd 会产生数十倍 z 值；
# 截断到 ±8 既不改变排序（AUC 不受影响），又能避免把网络打爆。
Z_CLIP = 8.0


# =============================================================================
# 1. TGAM 协议解析
# =============================================================================
@dataclass
class Sample:
    """一秒钟的脑电摘要（由一个大包 + 期间的全部小包统计得到）。"""
    t: float                 # 本地时间戳 (epoch seconds)
    signal: int = 0          # 信号质量：0 最好，>100 接触不良
    attention: int = 0       # 设备注意力 0-100
    meditation: int = 0      # 设备放松度 0-100
    bands: dict = field(default_factory=dict)
    raw_n: int = 0           # 本秒内收到的 raw 采样点数
    raw_std: float = 0.0     # raw 标准差（伪迹指标）
    raw_max: float = 0.0     # raw 最大绝对值（伪迹指标）
    label: int = -1          # 段标签：0 静息 / 1 心算 / -1 未标注

    def row(self):
        d = {"t": round(self.t, 3), "label": self.label, "signal": self.signal,
             "attention": self.attention, "meditation": self.meditation,
             "raw_std": round(self.raw_std, 2), "raw_max": round(self.raw_max, 2)}
        for b in BANDS:
            d[b] = int(self.bands.get(b, 0))
        return d


class TGAMParser:
    """TGAM 字节流状态机。

    帧格式: 0xAA 0xAA <len> <payload...> <checksum>
    checksum = (~sum(payload)) & 0xFF
      len = 0x20 大包: 02 sig | 83 18 + 24B 八频段 | 04 attention | 05 meditation
      len = 0x04 小包: 80 02 hi lo  -> 16 位有符号 raw 采样 (512 Hz)
    """

    def __init__(self):
        self.buf = bytearray()
        self.bad = 0
        self.n_big = 0
        self.n_raw = 0
        self._raw = []
        self.raw_deque = deque(maxlen=4096)   # 供界面实时绘制原始波形

    def feed(self, data: bytes):
        self.buf += data
        out = []
        buf = self.buf
        while True:
            i = buf.find(b"\xaa\xaa")
            if i < 0:
                if len(buf) > 1024:
                    del buf[:-32]
                break
            if i > 0:
                del buf[:i]
            if len(buf) < 4:
                break
            plen = buf[2]
            if len(buf) < plen + 4:
                break
            payload = bytes(buf[3:3 + plen])
            cs = buf[3 + plen]
            if ((~sum(payload)) & 0xFF) != cs:
                self.bad += 1
                del buf[0]
                continue
            del buf[:plen + 4]
            if plen == 0x20:
                s = self._parse_big(payload)
                if s is not None:
                    out.append(s)
            elif plen == 0x04 and len(payload) == 4 and payload[0] == 0x80 and payload[1] == 0x02:
                v = (payload[2] << 8) | payload[3]
                if v > 32768:
                    v -= 65536
                self._raw.append(v)
                self.raw_deque.append(v)
                self.n_raw += 1
        self.buf = buf
        return out

    def _parse_big(self, p: bytes):
        if len(p) < 32:
            return None
        self.n_big += 1
        s = Sample(t=time.time(), signal=p[1], attention=p[29], meditation=p[31])
        if p[2] == 0x83 and p[3] == 0x18:
            for k, name in enumerate(BANDS):
                o = 4 + 3 * k
                s.bands[name] = (p[o] << 16) | (p[o + 1] << 8) | p[o + 2]
        raw = self._raw
        if raw:
            arr = np.asarray(raw, dtype=np.float64) if np is not None else None
            s.raw_n = len(raw)
            if arr is not None:
                s.raw_std = float(arr.std())
                s.raw_max = float(np.abs(arr).max())
            else:  # 无 numpy 时的退化路径（仍可运行）
                m = sum(raw) / len(raw)
                s.raw_std = math.sqrt(sum((v - m) ** 2 for v in raw) / len(raw))
                s.raw_max = float(max(abs(v) for v in raw))
        self._raw = []
        return s


# =============================================================================
# 2. 数据源：串口 / 回放 / 模拟
# =============================================================================
class BaseSource(threading.Thread):
    """所有数据源都是后台线程，把 Sample 推进同一个队列。"""

    def __init__(self, out_queue: "queue.Queue"):
        super().__init__(daemon=True)
        self.q = out_queue
        self._stop = threading.Event()
        self.ok = False
        self.message = "未启动"
        self.parser = TGAMParser()
        self.label = -1
        self.raw_callback = None

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def _emit(self, samples):
        for s in samples:
            if self.label >= 0:      # 回放/模拟源自带标签时不覆盖
                s.label = self.label
            self.q.put(s)

    def close(self):
        pass


class SerialSource(BaseSource):
    def __init__(self, out_queue, port, baud=DEFAULT_BAUD):
        super().__init__(out_queue)
        self.port, self.baud = port, baud
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            self.ser.reset_input_buffer()
            self.ok = True
            self.message = f"{self.port} @ {self.baud} 已连接"
        except Exception as e:
            self.ok = False
            self.message = f"打开 {self.port} 失败: {e}"
            return
        last_ok = time.time()
        while not self.stopped():
            try:
                chunk = self.ser.read(256)
                if not chunk:
                    if time.time() - last_ok > 5:
                        self.message = "串口 5 秒无数据（设备是否开机/佩戴？）"
                    continue
                last_ok = time.time()
                if self.raw_callback is not None:
                    self.raw_callback(chunk)
                samples = self.parser.feed(chunk)
                if samples:
                    self.message = f"{self.port} 已连接 · 已收 {self.parser.n_big} 包 · 坏包 {self.parser.bad}"
                self._emit(samples)
            except Exception as e:
                self.ok = False
                self.message = f"串口读取异常: {e}"
                time.sleep(0.5)
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass

    def close(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass


class ReplaySource(BaseSource):
    """按原始时间间隔回放 CSV，用于无设备时的完整流程演示与测试。"""

    def __init__(self, out_queue, path, speed=1.0, loop=False):
        super().__init__(out_queue)
        self.path, self.speed, self.loop = path, max(0.05, speed), loop

    def run(self):
        while not self.stopped():
            try:
                rows = list(csv.DictReader(open(self.path, encoding="utf-8")))
            except Exception as e:
                self.message = f"回放失败: {e}"
                return
            if not rows:
                self.message = "回放文件为空"
                return
            self.ok = True
            self.message = f"回放 {os.path.basename(self.path)} · {len(rows)} 秒"
            prev_t = None
            for r in rows:
                if self.stopped():
                    return
                t = float(r["t"])
                if prev_t is not None:
                    delay = min(2.0, max(0.0, (t - prev_t) / self.speed))
                    time.sleep(delay)
                prev_t = t
                self._emit([row_to_sample(r)])
            if not self.loop:
                self.message = "回放结束"
                return
            self.message = "回放循环中…"


class SimulatorSource(BaseSource):
    """生理启发式模拟器：让没有硬件时也能完整走通"采集→建模→演示"。

    生成规律（仅用于流程演示，不可作为真实实验结论）：
      静息段：alpha 增强、theta 降低、Attention 降低但波动小
      心算段：beta/gamma 增强、alpha 降低、Attention 抬高且方差增大
    """

    def __init__(self, out_queue, state_provider=None):
        super().__init__(out_queue)
        self.state_provider = state_provider or (lambda: "rest")
        self._t0 = time.time()

    def run(self):
        self.ok = True
        self.message = "模拟数据源运行中（非真实测量）"
        base = {b: random.uniform(3000, 12000) for b in BANDS}
        att = 45.0
        med = 55.0
        while not self.stopped():
            time.sleep(1.0)
            st = self.state_provider()
            focus = (st == "focus")
            drift = 0.25 if focus else 0.12
            for b in BANDS:
                target = base[b] * (1.6 if (focus and b in ("lowBeta", "highBeta", "lowGamma", "midGamma")) else
                                    0.7 if (focus and b in ("lowAlpha", "highAlpha")) else
                                    1.5 if ((not focus) and b in ("lowAlpha", "highAlpha")) else 1.0)
                base[b] = base[b] * 0.85 + target * 0.15 * random.uniform(0.9, 1.1)
            att += (14 if focus else -10) * 0.15 + random.gauss(0, 4.5)
            att = min(99, max(1, att * 0.85 + (68 if focus else 38) * 0.15))
            med += (-6 if focus else 6) * 0.15 + random.gauss(0, 3.0)
            med = min(99, max(1, med))
            s = Sample(t=time.time(), signal=0, attention=int(att), meditation=int(med),
                       bands={b: int(v) for b, v in base.items()})
            s.raw_n = 512
            s.raw_std = 900 if focus else 700
            s.raw_max = 2600 if focus else 1900
            self._emit([s])


def row_to_sample(r: dict) -> Sample:
    s = Sample(t=float(r["t"]), signal=int(float(r.get("signal") or 0)),
               attention=int(float(r.get("attention") or 0)),
               meditation=int(float(r.get("meditation") or 0)),
               label=int(float(r.get("label") or -1)))
    s.raw_std = float(r.get("raw_std") or 0)
    s.raw_max = float(r.get("raw_max") or 0)
    s.bands = {b: int(float(r.get(b) or 0)) for b in BANDS}
    return s


# =============================================================================
# 3. 录制器
# =============================================================================
class Recorder:
    """把每秒 Sample 写成 UTF-8 CSV，文件头记录协议元信息。"""

    def __init__(self, path, subject="S01", note=""):
        self.path = path
        self.rows = []
        self.subject = subject
        self.note = note
        self.started = time.time()
        self._fh = None
        self._w = None

    def open(self):
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=list(Sample(t=0.0).row().keys()))
        self._w.writeheader()

    @property
    def n(self):
        return len(self.rows)

    def add(self, s: Sample):
        if self._w is None:
            self.open()
        self._w.writerow(s.row())
        self.rows.append(s)
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def summary(self):
        n0 = sum(1 for s in self.rows if s.label == 0)
        n1 = sum(1 for s in self.rows if s.label == 1)
        bad = sum(1 for s in self.rows if s.signal > 100)
        return {"seconds": len(self.rows), "rest": n0, "focus": n1, "bad_signal": bad}




# =============================================================================
# 4. 特征工程与数据集构建
# =============================================================================
def frame_features(samples):
    """把一段连续 Sample 转成 (T, C) 特征矩阵（原始量纲，未标准化）。

    设计说明（报告里会解释理由）：
      * Attention 是设备厂商算法的输出，只作为**特征**，绝不作为标签；
      * 8 个频段取 log1p 压缩动态范围（原始值可达 1e6，跨被试差异极大）；
      * attention_diff 捕捉趋势，raw_std / raw_max 把肌电/运动信息显式喂给网络；
      * signal / meditation 不进特征，见 FEATURE_NAMES 处的说明。
    """
    T = len(samples)
    C = len(FEATURE_NAMES)
    X = np.zeros((T, C), dtype=np.float32)
    i_att = FEATURE_INDEX["attention"]
    i_diff = FEATURE_INDEX["attention_diff"]
    i_std = FEATURE_INDEX["raw_std"]
    i_max = FEATURE_INDEX["raw_max"]
    band_idx = [FEATURE_INDEX[b] for b in BAND_FEATURES]
    prev_att = None
    for i, s in enumerate(samples):
        att = s.attention / 100.0
        X[i, i_att] = att
        for k, b in enumerate(BANDS):
            X[i, band_idx[k]] = math.log1p(max(0, s.bands.get(b, 0))) / 20.0
        X[i, i_diff] = 0.0 if prev_att is None else (att - prev_att)
        X[i, i_std] = min(3.0, s.raw_std / 2000.0)
        X[i, i_max] = min(3.0, s.raw_max / 6000.0)
        prev_att = att
    return X


def load_csv_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quality_lines(samples, settle=SETTLE_SECONDS):
    """采集结束后的一分钟体检：两类状态的幅度与 alpha 是否对齐。

    单通道 FP1 最容易出的事故不是"信号坏了"，而是"两类状态的眼/手动作不一样"：
    比如静息闭眼放松、心算时睁眼皱眉，那模型学到的是眨眼而不是专注。
    这里只做描述与提示，不改数据、不参与训练。

    返回 (lines, verdict)，verdict 为 ok / warn / unknown。
    """
    runs = []                      # 连续同 label 的片段
    for s in samples:
        if runs and runs[-1][0] == s.label:
            runs[-1][1].append(s)
        else:
            runs.append((s.label, [s]))
    agg = {}
    for lab, seg in runs:
        seg = seg[settle:] if len(seg) > settle else seg
        if not seg:
            continue
        d = agg.setdefault(lab, {"n": 0, "std": 0.0, "alpha": 0.0, "delta": 0.0})
        d["n"] += len(seg)
        d["std"] += sum(x.raw_std for x in seg)
        d["alpha"] += sum(x.bands.get("highAlpha", 0) for x in seg)
        d["delta"] += sum(x.bands.get("delta", 0) for x in seg)
    if 0 not in agg or 1 not in agg or agg[0]["n"] < 10 or agg[1]["n"] < 10:
        return ["体检：两类有效数据不足，无法比较（至少各需要 10 秒）。"], "unknown"
    for d in agg.values():
        for k in ("std", "alpha", "delta"):
            d[k] /= d["n"]
    r, m = agg[0], agg[1]
    r_std = m["std"] / max(1e-6, r["std"])
    r_alpha = m["alpha"] / max(1e-6, r["alpha"])
    r_delta = m["delta"] / max(1e-6, r["delta"])
    lines = [
        "静息：幅度 %.1f，highAlpha %.0f，delta %.0f" % (r["std"], r["alpha"], r["delta"]),
        "心算：幅度 %.1f（静息的 %.2f 倍），highAlpha %.0f（%.2f 倍），delta %.0f（%.2f 倍）"
        % (m["std"], r_std, m["alpha"], r_alpha, m["delta"], r_delta),
    ]
    if r_alpha < 0.70 and r_std > 1.50:
        lines.append("结论：心算段 alpha 明显下降、幅度明显上升，疑似睁眼/眨眼/眼动。"
                     "请确认全程闭眼（可加眼罩），这一段不建议入库。")
        return lines, "warn"
    if r_std > 1.50 or r_std < 0.67:
        lines.append("结论：两类幅度相差较大，可能有咬牙、点头、说话或手部动作。"
                     "请让被试两种状态保持同样的姿势与口部放松，这一段建议重采。")
        return lines, "warn"
    lines.append("结论：两类幅度与 alpha 基本对齐，标签较干净（幅度比 %.2f）。" % r_std)
    return lines, "ok"


def is_raw_csv(path):
    """是否为可入库的原始采集文件。

    *_tasks.csv 是心算段的行为记录（t,qid,question,answer,result），列结构不同，
    绝不能混进数据集，否则 build_dataset 会把文字当成 EEG 行。
    """
    if not path.lower().endswith(".csv") or path.lower().endswith("_tasks.csv"):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            head = f.readline()
    except OSError:
        return False
    return ("signal" in head) and ("attention" in head)


def _segments(rows):
    """把连续、同标签、间隔≈1s 的行切成段，返回 [(start_idx, end_idx, label)]。"""
    segs, start = [], None
    prev_t, prev_lab = None, None
    for i, r in enumerate(rows):
        try:
            t = float(r["t"])
        except Exception:
            continue
        lab = int(float(r.get("label") or -1))
        if start is None:
            start, prev_t, prev_lab = i, t, lab
            continue
        gap = t - prev_t if prev_t is not None else 1.0
        if lab != prev_lab or gap > 2.5 or lab not in (0, 1):
            segs.append((start, i, prev_lab))
            start = i
        prev_t, prev_lab = t, lab
    if start is not None:
        segs.append((start, len(rows), prev_lab))
    return [s for s in segs if s[2] in (0, 1)]


def build_dataset(paths, window=WINDOW, stride=1, settle=SETTLE_SECONDS,
                  max_signal=100, artifact_z=4.0, calib_seconds=CALIB_SECONDS,
                  verbose=False):
    """原始流 → 切段 → 丢稳定期 → 质量/伪迹过滤 → 滑窗 → 标签 → 按会话分组。

    与早期版本相比有两处关键修正（详见 docs/报告.md 的「清洗与剔除」一节）：
      1) 伪迹阈值不再有 2500 的死地板。旧阈值 max(med+4·1.4826·MAD, 2500) 远高于真实
         raw_std 上界（实测最大 424），导致「伪迹」恒为 0、清洗形同虚设。现在只保留数据
         自适应的 MAD 判据，并以中位数的 1.5 倍为下限（避免 MAD=0 时误杀全部样本）。
      2) signal 用 TGAM 文档列出的「耳机没戴好」码集合判断，而不是只比 >100。

    另外，每个会话的第一个静息段会被抽出来计算「本人基线」（mu/sd），供
    standardize_by_group() 使用 —— 这就是报告里承诺的「每人单独校准」。
    """
    if np is None:
        raise RuntimeError(
            "当前 Python（%s）里没有可用的 numpy，无法构建数据集。\n"
            "请用 `pip install numpy` 装到同一个 Python，"
            "或把与该 Python 版本匹配的 numpy 解压到程序目录的 vendor/ 下。"
            % sys.version.split()[0])
    window, stride, settle = int(window), int(stride), int(settle)   # GUI 传进来的可能是 float
    calib_seconds = int(calib_seconds)
    paths = [p for p in paths if is_raw_csv(p)]
    Xs, ys, groups, per_file = [], [], [], []
    reject = {"短段": 0, "信号差": 0, "伪迹": 0, "窗内不完整": 0}

    all_std = []
    for p in paths:
        for r in load_csv_rows(p)[::5]:
            try:
                all_std.append(float(r.get("raw_std") or 0))
            except Exception:
                pass
    if all_std:
        arr = np.asarray(all_std)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        robust = med + artifact_z * 1.4826 * mad      # 经典稳健 4σ 界
        # 实测 raw_std 是明显双峰的（S01 静息 p50≈36、p75≈151），稳健界会砍掉 1/3 的秒数，
        # 而 raw_std 本身与标签强相关（越大越偏静息），按它剔除等于删除有效信号。
        # 因此伪迹阈值取「稳健界」与「全库 99.5 分位」中的较松者，只捉真正的极端幅度。
        std_thr = max(robust, float(np.quantile(arr, 0.995)), med * 4.0)
    else:
        std_thr = float("inf")          # 没有 raw_std 列时不做伪迹过滤

    calib_rows = {}                     # 会话名 -> (T, C) 第一个静息段的原始特征
    for p in paths:
        rows = load_csv_rows(p)
        name = os.path.basename(p)
        got = 0
        have_calib = name in calib_rows
        for (a, b, lab) in _segments(rows):
            seg_rows = rows[a:b]
            if len(seg_rows) <= settle + window:
                reject["短段"] += 1
                continue
            seg_rows = seg_rows[settle:]
            ok = []
            for r in seg_rows:
                try:
                    sig = float(r.get("signal") or 0)
                    rs = float(r.get("raw_std") or 0)
                    att = float(r.get("attention") or 0)
                except Exception:
                    continue
                bad = (sig > max_signal) or (int(sig) in BAD_SIGNAL_CODES) or (att <= 0)
                art = rs > std_thr
                if bad:
                    reject["信号差"] += 1
                if art:
                    reject["伪迹"] += 1
                ok.append((r, (not bad) and (not art)))
            if lab == 0 and not have_calib:
                # 只用「本次会话的第一个静息段」做本人基线：与部署协议一致，且不引入任何标签信息
                feats = [frame_features([row_to_sample(r)])[0] for r, keep in ok if keep]
                if feats:
                    calib_rows[name] = np.stack(feats[:calib_seconds if calib_seconds > 0 else None])
                    have_calib = True
            for i in range(0, len(ok) - window + 1, stride):
                chunk = ok[i:i + window]
                if not all(v for _, v in chunk):
                    reject["窗内不完整"] += 1
                    continue
                Xs.append(frame_features([row_to_sample(r) for r, _ in chunk]))
                ys.append(lab)
                groups.append(name)
                got += 1
        per_file.append({"file": name, "rows": len(rows), "windows": got})
        if verbose:
            print(f"[dataset] {name}: rows={len(rows)} windows={got}")

    if not Xs:
        raise RuntimeError("没有生成任何窗口：请检查标签是否有效、信号质量是否合格。"
                           f"（伪迹阈值 raw_std>{std_thr:.0f}，拒绝统计 {reject}）")
    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    garr = np.asarray(groups)
    calib, mu_global, sd_global = fit_calibration(X, garr, calib_rows)
    return {"X": X, "y": y, "groups": garr,
            "per_file": per_file, "reject": reject, "std_thr": std_thr,
            "calib": calib, "mu_global": mu_global, "sd_global": sd_global}


def safe_sd(F, ref=None, floor_ratio=0.05, ref_ratio=0.25):
    """列标准差的稳健下界。

    早期版本用 sd+1e-6 做下限：只要某列在训练会话里是常量，换到别的会话时 z 值就会被
    放大到十万量级（signal 列的实际事故）。这里给两道下限：
      1) 该表典型 sd 的 5%；
      2) 若有参考 sd（全库同一列的 sd），不低于它的 25%。
    静息段里某一列常常几乎不动，只用局部 sd 会把一个 0.1 的抖动放大成 z=100；
    参考 sd 反映「这一列真实的动态范围」，可以避免。
    """
    F = np.asarray(F, dtype=np.float32)
    sd = F.std(axis=0).astype(np.float32)
    pos = sd[sd > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    floor = np.maximum(float(max(1e-3, floor_ratio * med)), 1e-6)
    if ref is not None:
        floor = np.maximum(floor, ref_ratio * np.asarray(ref, dtype=np.float32))
    return np.maximum(sd, floor).astype(np.float32)


def zscore(x, mu, sd, clip=Z_CLIP):
    """标准化并截断到 ±clip，实时推演与离线评估共用同一套。"""
    z = (np.asarray(x, dtype=np.float32) - mu) / sd
    return np.clip(z, -clip, clip).astype(np.float32)


def fit_calibration(X, groups, calib_rows, min_rows=5):
    """为每个会话估计本人静息基线，返回 (calib, mu_global, sd_global)。

    有静息段的会话用自己的（mu, sd）；缺失时回退到「所有会话静息段的池化统计」，
    并在 calib[g]["from"] 里记下来源，便于在报告里如实说明。
    """
    uniq = sorted(set(np.asarray(groups).tolist()))
    pX = np.asarray(X, dtype=np.float32)
    ref_sd = pX.reshape(-1, pX.shape[-1]).std(axis=0).astype(np.float32)
    pool = (np.concatenate(list(calib_rows.values()), axis=0) if calib_rows
            else pX.reshape(-1, pX.shape[-1]))
    mu_g = pool.mean(axis=0).astype(np.float32)
    sd_g = safe_sd(pool, ref=ref_sd)
    calib = {}
    for g in uniq:
        f = calib_rows.get(g)
        if f is not None and len(f) >= min_rows:
            calib[g] = {"mu": f.mean(axis=0).astype(np.float32),
                        "sd": safe_sd(f, ref=ref_sd),
                        "n": int(len(f)), "from": "rest"}
        else:
            calib[g] = {"mu": mu_g, "sd": sd_g, "n": 0, "from": "global"}
    return calib, mu_g, sd_g


def standardize_by_group(ds):
    """按会话（= 每次佩戴）各自的静息基线做 z-score。

    训练、验证、实时推演共用同一套变换，保证「线上看到的就是线下评的」。
    """
    X = np.asarray(ds["X"], dtype=np.float32)
    out = np.empty_like(X)
    calib = ds.get("calib") or {}
    mu_g, sd_g = ds.get("mu_global"), ds.get("sd_global")
    for i, g in enumerate(ds["groups"]):
        c = calib.get(g)
        mu = c["mu"] if c is not None else mu_g
        sd = c["sd"] if c is not None else sd_g
        out[i] = zscore(X[i], mu, sd)
    return out


def split_dataset(ds, val_ratio=0.2, seed=42, by_group=True):
    """按"会话文件"分组划分，避免同一段连续数据同时进训练/验证集（防泄漏）。

    注意：按会话分组时 val_ratio 只能取整到「整数个会话」。只有 2 个会话时
    n_val = max(1, round(2*0.2)) = 1，即实际留出 50%，而不是 20%；
    train_from_dataset 会把实际比例写进 meta，并在界面/报告里如实提示。
    """
    rng = np.random.default_rng(seed)
    groups = ds["groups"]
    uniq = sorted(set(groups.tolist()))
    if by_group and len(uniq) >= 2:
        rng.shuffle(uniq)
        n_val = max(1, int(round(len(uniq) * val_ratio)))
        val_g = set(uniq[:n_val])
        val_mask = np.array([g in val_g for g in groups])
    else:
        idx = rng.permutation(len(ds["y"]))
        val_mask = np.zeros(len(idx), dtype=bool)
        val_mask[idx[:max(1, int(len(idx) * val_ratio))]] = True
    return ~val_mask, val_mask


def standardize(Xtr, Xva):
    """全局 z-score（仅用于自检/对照；正式训练走 standardize_by_group）。"""
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    sd = safe_sd(Xtr.reshape(-1, Xtr.shape[-1]))
    return zscore(Xtr, mu, sd), zscore(Xva, mu, sd), mu, sd


# =============================================================================
# 5. 1-D CNN（纯 NumPy；3 个卷积层，参数量 ≈3k）
# =============================================================================
class Conv1D:
    def __init__(self, cin, cout, k=3, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = (rng.standard_normal((cout, cin, k)) * math.sqrt(2.0 / (cin * k))).astype(np.float32)
        self.b = np.zeros(cout, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache = None

    def forward(self, x):                                  # (B, Cin, T)
        B, C, T = x.shape
        K = self.W.shape[2]
        xp = np.pad(x, ((0, 0), (0, 0), (K // 2, K // 2)))
        cols = np.lib.stride_tricks.sliding_window_view(xp, K, axis=2)   # (B,C,T,K)
        self.cache = (cols, B, C, T)
        return np.einsum("bctk,ock->bot", cols, self.W, optimize=True) + self.b[None, :, None]

    def backward(self, dout):
        cols, B, C, T = self.cache
        K = self.W.shape[2]
        self.dW = np.einsum("bot,bctk->ock", dout, cols, optimize=True).astype(np.float32)
        self.db = dout.sum(axis=(0, 2)).astype(np.float32)
        dcols = np.einsum("bot,ock->bctk", dout, self.W, optimize=True)
        dxp = np.zeros((B, C, T + 2 * (K // 2)), dtype=np.float32)
        for k in range(K):
            dxp[:, :, k:k + T] += dcols[:, :, :, k]
        return dxp[:, :, K // 2: K // 2 + T]


class Dense:
    def __init__(self, cin, cout, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = (rng.standard_normal((cin, cout)) * math.sqrt(2.0 / cin)).astype(np.float32)
        self.b = np.zeros(cout, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self.cache
        self.dW = (x.T @ dout).astype(np.float32)
        self.db = dout.sum(axis=0).astype(np.float32)
        return dout @ self.W.T


class FocusCNN:
    """Conv1D(12→16,k3) → Conv1D(16→20,k3) → Conv1D(20→24,k3) → GAP → Dense(24→1)

    参数量 3061（12 维输入），与 Project Brief「1-D CNN with 3 layers ≈3k params」对齐。
    """

    def __init__(self, in_ch=len(FEATURE_NAMES), seed=0):
        rng = np.random.default_rng(seed)
        self.c1 = Conv1D(in_ch, 16, 3, rng)
        self.c2 = Conv1D(16, 20, 3, rng)
        self.c3 = Conv1D(20, 24, 3, rng)
        self.fc = Dense(24, 1, rng)
        self.acts = {}

    def params(self):
        return [self.c1.W, self.c1.b, self.c2.W, self.c2.b,
                self.c3.W, self.c3.b, self.fc.W, self.fc.b]

    def n_params(self):
        return int(sum(p.size for p in self.params()))

    def forward(self, x, cache=True):
        h = x.transpose(0, 2, 1).astype(np.float32)          # (B,T,C) -> (B,C,T)
        a1 = self.c1.forward(h); h1 = np.maximum(a1, 0)
        a2 = self.c2.forward(h1); h2 = np.maximum(a2, 0)
        a3 = self.c3.forward(h2); h3 = np.maximum(a3, 0)
        g = h3.mean(axis=2)
        z = self.fc.forward(g)
        if cache:
            self.acts = {"a1": a1, "a2": a2, "a3": a3, "h3": h3, "g": g}
        return z[:, 0]

    def backward(self, dz):
        B = dz.shape[0]
        T = self.acts["h3"].shape[2]
        dg = self.fc.backward(dz[:, None].astype(np.float32))
        dh3 = np.repeat(dg[:, :, None], T, axis=2) / T
        da3 = dh3 * (self.acts["a3"] > 0)
        dh2 = self.c3.backward(da3)
        da2 = dh2 * (self.acts["a2"] > 0)
        dh1 = self.c2.backward(da2)
        da1 = dh1 * (self.acts["a1"] > 0)
        self.c1.backward(da1)                                # 只为累加 dW/db

    def grads(self):
        return [self.c1.dW, self.c1.db, self.c2.dW, self.c2.db,
                self.c3.dW, self.c3.db, self.fc.dW, self.fc.db]

    def predict_proba(self, x, batch=256):
        out = []
        for i in range(0, len(x), batch):
            z = self.forward(x[i:i + batch], cache=False)
            out.append(sigmoid(z))
        return np.concatenate(out) if out else np.zeros(0)

    def state(self):
        return {"c1.W": self.c1.W, "c1.b": self.c1.b, "c2.W": self.c2.W, "c2.b": self.c2.b,
                "c3.W": self.c3.W, "c3.b": self.c3.b, "fc.W": self.fc.W, "fc.b": self.fc.b}

    def load_state(self, st):
        self.c1.W = np.asarray(st["c1.W"], dtype=np.float32); self.c1.b = np.asarray(st["c1.b"], dtype=np.float32)
        self.c2.W = np.asarray(st["c2.W"], dtype=np.float32); self.c2.b = np.asarray(st["c2.b"], dtype=np.float32)
        self.c3.W = np.asarray(st["c3.W"], dtype=np.float32); self.c3.b = np.asarray(st["c3.b"], dtype=np.float32)
        self.fc.W = np.asarray(st["fc.W"], dtype=np.float32); self.fc.b = np.asarray(st["fc.b"], dtype=np.float32)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-4):
        self.params, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = g + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(p.dtype)


def sigmoid(z):
    """数值稳定的 logistic。|z|>60 时饱和，避免 exp 溢出（旧版 `1/(1+exp(-z))` 会刷 RuntimeWarning）。"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def bce_with_logits(z, y):
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(loss.mean())


# ---- 指标 -------------------------------------------------------------------
def confusion_matrix_bin(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def f1_score_bin(y, pred):
    c = confusion_matrix_bin(y, pred)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def auc_score(y, score):
    y = np.asarray(y).astype(int); s = np.asarray(score, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for u in np.where(cnt > 1)[0]:
        m = inv == u
        ranks[m] = ranks[m].mean()
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(y, prob, thr=0.5):
    pred = (np.asarray(prob) >= thr).astype(int)
    c = confusion_matrix_bin(y, pred)
    acc = (c["tp"] + c["tn"]) / max(1, len(y))
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return {"acc": acc, "precision": prec, "recall": rec,
            "f1": f1_score_bin(y, pred), "auc": auc_score(y, prob),
            "confusion": c, "thr": thr, "n": int(len(y))}


def cm_matrix(m):
    """把 evaluate() 的 confusion 字典转成 2×2 列表 [[tn, fp], [fn, tp]]。"""
    c = m["confusion"]
    return [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]


def best_threshold(y, prob, lo=0.05, hi=0.95, n=91):
    """扫描阈值，返回使平衡准确率 (TPR+TNR)/2 最大的那个。

    注意：在验证集上挑阈值再用同一批数据报指标是乐观估计，报告里必须标注；
    真正可直接部署的是 rest_threshold()（只用静息段，不需要标签）。
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2 or len(prob) == 0:
        return 0.5
    best_t, best_v = 0.5, -1.0
    for t in np.unique(np.concatenate([[0.5], np.linspace(lo, hi, n)])):
        pred = (prob >= t).astype(int)
        c = confusion_matrix_bin(y, pred)
        tpr = c["tp"] / max(1, c["tp"] + c["fn"])
        tnr = c["tn"] / max(1, c["tn"] + c["fp"])
        v = 0.5 * (tpr + tnr)
        if v > best_v + 1e-9:
            best_t, best_v = float(t), v
    return best_t


def rest_threshold(prob_rest, q=0.90, lo=0.2, hi=0.8):
    """无标签阈值校准：把「本人静息段」被打分的 q 分位当作判定阈值。

    部署时不需要任何标签——让被试静息 50 秒，取模型对自己静息窗打分的 90 分位作为门限，
    等价于把误报率控制在 10% 左右。

    为什么是 90 而不是 95：50 个窗口只有约 6 个独立块（lag-1 自相关 ≈0.9），
    95 分位恰好落在极少数尾部窗口上，噪声很大（实测两折一个 0.95 撞上界、另一个 0.52）。
    90 分位更稳，两折均值也更高（0.869 vs 0.862）。
    """
    p = np.asarray(prob_rest, dtype=float)
    if p.size == 0:
        return 0.5
    return float(np.clip(np.quantile(p, q), lo, hi))


# ---- 训练 -------------------------------------------------------------------
def train_cnn(Xtr, ytr, Xva, yva, epochs=300, batch=32, lr=1e-3, seed=0,
              progress=None, stop_flag=None, augment=True, select="auc", patience=60):
    """训练 1-D CNN，并按 select 指定的验证指标保存最佳权重。

    select:
      "auc" —— 按验证 AUC 选点（默认）。AUC 只看排序，不受会话之间概率整体平移的影响，
               在「只有两个会话、窗与窗之间高度相关」的场景下比 val_loss 稳得多。
      "loss" —— 旧行为（按 val_loss）。保留只为复现历史结果：实测它会选中第 1 轮
               （验证会话的概率校准在第一轮之后就开始漂移，而 AUC 其实还在变好），
               导致报出去的准确率是"基本没训练"的模型的成绩。
    """
    net = FocusCNN(Xtr.shape[-1], seed=seed)
    opt = Adam(net.params(), lr=lr, wd=1e-4)
    rng = np.random.default_rng(seed)
    T = Xtr.shape[1]
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
            "val_f1": [], "val_auc": []}
    best = {"state": None, "score": -1e9, "val_loss": 1e9, "epoch": -1, "select": select}
    bad = 0
    n = len(ytr)

    for ep in range(1, epochs + 1):
        if stop_flag is not None and stop_flag():
            break
        idx = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            xb = Xtr[b].copy()
            if augment:
                # 数据增强：窗内 ±2 帧平移（边缘复制，不用 np.roll：旧版会首尾回绕，
                # 制造一个物理上不存在的跳变）+ 每样本 ±3% 增益抖动 + 小高斯噪声。
                # 在 lag-1 自相关 ≈0.89 的序列上，纯平移几乎不改变样本，所以必须补增益抖动。
                for j in range(len(b)):
                    sh = int(rng.integers(-2, 3))
                    if sh:
                        pad = np.pad(xb[j], ((2, 2), (0, 0)), mode="edge")
                        xb[j] = pad[2 - sh:2 - sh + T]
                xb *= (1.0 + rng.normal(0.0, 0.03, (len(b), 1, 1))).astype(np.float32)
                xb += rng.normal(0.0, 0.02, xb.shape).astype(np.float32)
            z = net.forward(xb)
            tot += bce_with_logits(z, ytr[b]) * len(b)
            net.backward(((sigmoid(z) - ytr[b]) / len(b)).astype(np.float32))
            opt.step(net.grads())
        tr_loss = tot / n

        zv = net.forward(Xva, cache=False)
        vl = bce_with_logits(zv, yva)
        pv = sigmoid(zv)
        acc = float(((pv >= 0.5) == yva).mean())
        f1 = f1_score_bin(yva, (pv >= 0.5).astype(int))
        va = auc_score(yva, pv)

        for k, v in zip(["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_auc"],
                        [ep, tr_loss, vl, acc, f1, va]):
            hist[k].append(v)

        if select == "auc":
            improved = va > best["score"] + 1e-6
        else:
            improved = vl < best["val_loss"] - 1e-5
        if improved:
            best.update(score=(va if select == "auc" else -vl), val_loss=vl, epoch=ep,
                        state={k: v.copy() for k, v in net.state().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if progress:
                    progress(ep, epochs, tr_loss, vl, acc, f1, va, "early-stop")
                break
        if progress and (ep == 1 or ep % 5 == 0):
            progress(ep, epochs, tr_loss, vl, acc, f1, va, "")
    if best["state"] is not None:
        net.load_state(best["state"])
    return net, hist, best


def threshold_baseline(Xraw, y, win=3, thr=75.0):
    """Project Brief 的 Option A：3 秒滑动平均 + 阈值，作为 CNN 的对照。

    必须与 CNN 在**同一批窗口**上比较。旧版传的是全量数据（train+val），
    而 CNN 的指标只算验证集，于是基线被虚高（0.783 vs 0.686 就是这个 bug 的产物）。
    """
    i_att = FEATURE_INDEX["attention"]
    att = np.asarray(Xraw)[:, :, i_att] * 100.0
    ma = np.apply_along_axis(lambda r: np.convolve(r, np.ones(win) / win, mode="valid")[-1], 1, att)
    pred = (ma > thr).astype(int)
    m = evaluate(y, ma / 100.0, thr=thr / 100.0)
    return m, float((pred == np.asarray(y).astype(int)).mean())


# ---- 模型存取 ----------------------------------------------------------------
def save_model(path, net, mu, sd, meta):
    payload = {"format": "neurofocus-cnn-v1", "arch": "Conv16-20-24-GAP-Dense1",
               "features": FEATURE_NAMES, "window": WINDOW,
               "params": int(net.n_params()),
               "mu": mu.tolist(), "sd": sd.tolist(),
               "weights": {k: v.tolist() for k, v in net.state().items()},
               "meta": meta}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def load_model(path):
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    net = FocusCNN(len(p["features"]))
    net.load_state({k: np.asarray(v, dtype=np.float32) for k, v in p["weights"].items()})
    return net, np.asarray(p["mu"], dtype=np.float32), np.asarray(p["sd"], dtype=np.float32), p["meta"]




# =============================================================================
# 6. 演示数据生成（无硬件时也能跑通全流程；文件中标注 simulated=1）
# =============================================================================
def simulate_session(path, subject="S00", blocks=6, block_sec=60, seed=1, gain=1.0, mix=0.35):
    """生成一条"看起来像真实采集"的原始 CSV（1 Hz 行，含 raw_std/raw_max）。

    刻意保留类间重叠（mix 比例的"走神/突然用力"）与慢漂移，让演示指标接近真实场景，
    而不是一个虚假的 100%。仅用于流程演示与回归测试；真实结论必须用真实采集数据。
    """
    rng = np.random.default_rng(seed)
    rest_band = {b: rng.uniform(3000, 11000) for b in BANDS}
    focus_gain = {("lowBeta", "highBeta", "lowGamma", "midGamma"): 1.45,
                  ("lowAlpha", "highAlpha"): 0.72}
    prof_rest = {b: rest_band[b] for b in BANDS}
    prof_focus = {}
    for b in BANDS:
        g = 1.05
        for ks, v in focus_gain.items():
            if b in ks:
                g = v
        prof_focus[b] = rest_band[b] * g

    order = []
    for i in range(blocks):
        order += [0, 1] if i % 2 == 0 else [1, 0]
    order = order[:blocks]

    rows, t = [], time.time() - blocks * block_sec
    att = med = None
    for lab in order:
        seg_bias = rng.normal(0, 0.12)          # 段级基线漂移（跨会话差异）
        for _ in range(block_sec):
            state = lab
            if rng.random() < mix:              # 走神 / 短暂发力
                state = 1 - lab
            prof = prof_focus if state == 1 else prof_rest
            new = {}
            for b in BANDS:
                target = prof[b] * (1 + seg_bias) * rng.uniform(0.9, 1.1)
                prev = new.get(b, rows[-1][b] if rows else target)
                new[b] = max(200.0, 0.75 * prev + 0.25 * target)
            if att is None:
                att = 60.0 if state else 45.0
            t_att = (62.0 if state == 1 else 43.0) * (1 + 0.6 * seg_bias)
            att = 0.75 * att + 0.25 * (t_att + rng.normal(0, 7.0))
            t_med = 45.0 if state else 58.0
            med = 55.0 if med is None else 0.8 * med + 0.2 * (t_med + rng.normal(0, 6.0))
            art = rng.random() < 0.02
            rows.append({
                "t": round(t, 3), "label": lab,
                "signal": int(rng.integers(0, 3) if not art else rng.integers(120, 200)),
                "attention": int(np.clip(att, 1, 99)), "meditation": int(np.clip(med, 1, 99)),
                "raw_std": round(float(abs(rng.normal(900 if state else 760, 130)) * (5.5 if art else 1.0)), 2),
                "raw_max": round(float(abs(rng.normal(2600 if state else 2000, 320)) * (3.5 if art else 1.0)), 2),
                **{b: int(max(0, v * gain)) for b, v in new.items()},
            })
            t += 1.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def make_demo_set(outdir, subjects=("S01", "S02", "S03", "S04"), blocks=6, block_sec=60, mix=0.35):
    paths = []
    for i, s in enumerate(subjects):
        p = os.path.join(outdir, f"{s}_demo_sim.csv")
        simulate_session(p, s, blocks=blocks, block_sec=block_sec, seed=100 + i,
                        gain=0.8 + 0.1 * i, mix=mix)
        paths.append(p)
    return paths


# =============================================================================
# 7. 训练流程编排（GUI 与命令行共用）
# =============================================================================
def train_from_dataset(ds, epochs=300, lr=1e-3, val_ratio=0.2, seed=42,
                       progress=None, stop_flag=None, select="auc",
                       standardize="train"):
    """在已经构建好的数据集上训练（GUI 的「开始训练」优先走这里，避免重复构建）。

    评估协议（报告里要如实写）：
      * 划分：按会话分组，留出整个会话做验证，绝不把同一段连续信号劈成两半；
      * 标准化：默认 ``standardize="train"`` —— 只用**训练会话**的均值/标准差，
        模型训练完 mu/sd 就是模型的固定参数，部署时直接套用，不需要佩戴者先静息；
        ``standardize="session"`` 改为每个会话用自己的静息段 z-score（换人部署的可选路径）；
      * 阈值：主表用 0.5（无标签、无需校准）；另报「本人静息打分 90 分位」和
        「验证集最优」两档作参考，并在注脚说明后者是乐观估计。

    本机 2 段真实数据上的实测（stride 1、474 窗、逐会话交叉验证）：
      train 训练集标准化 + 0.5 阈值 → acc 0.867 / 0.893，均值 0.880；
      session 逐会话静息校准 → acc 0.887 / 0.838，均值 0.863（并不更好，且部署更麻烦）。
    """
    tr, va = split_dataset(ds, val_ratio=val_ratio, seed=seed, by_group=True)
    if standardize == "session":
        Xz = standardize_by_group(ds)             # 每个会话各自的静息基线
        mu, sd = ds["mu_global"], ds["sd_global"]
    else:
        Ftr = ds["X"][tr].reshape(-1, ds["X"].shape[-1])
        mu = Ftr.mean(axis=0).astype(np.float32)
        sd = safe_sd(Ftr)                          # 只有训练会话参与，部署即固定
        Xz = zscore(ds["X"], mu, sd)
    Xtr, Xva = Xz[tr], Xz[va]
    ytr, yva = ds["y"][tr], ds["y"][va]
    net, hist, best = train_cnn(Xtr, ytr, Xva, yva, epochs=epochs, lr=lr, seed=seed,
                                progress=progress, stop_flag=stop_flag, select=select)
    pva = net.predict_proba(Xva)

    m = evaluate(yva, pva, thr=0.5)               # 主指标：不需要任何校准
    thr_val = best_threshold(yva, pva)
    m_val = evaluate(yva, pva, thr=thr_val)       # 乐观估计（阈值也用了验证标签）
    # 参考档：换人部署时用「本人静息打分 90 分位」当阈值（无标签）
    n_cal = max(10, min(int(CALIB_SECONDS), int((yva == 0).sum()) // 2))
    p_rest = pva[yva == 0][:n_cal]
    thr_rest = rest_threshold(p_rest)
    m_rest = evaluate(yva, pva, thr=thr_rest)

    base_m, base_acc = threshold_baseline(ds["X"][va], yva, win=3, thr=75.0)

    uniq = sorted(set(ds["groups"].tolist()))
    # 注意：split_dataset 返回布尔掩码，索引方式也要兼容，所以直接数实际取到的行数
    n_tr = int(ds["X"][tr].shape[0])
    n_va = int(ds["X"][va].shape[0])
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": n_tr, "n_val": n_va,
        "n_groups": len(uniq),
        "val_ratio_requested": float(val_ratio),
        "val_ratio_actual": n_va / max(1, len(ds["y"])),
        "pos_rate": float(ds["y"].mean()),
        "metrics": m,                          # 主指标 = 固定模型 + 0.5 阈值
        "metrics_thr_default": m,
        "metrics_thr_rest": m_rest,
        "metrics_thr_val_best": m_val,
        "threshold": {"rest": thr_rest, "val_best": thr_val, "default": 0.5,
                      "calib_windows": int(len(p_rest))},
        "baseline": base_m, "baseline_acc": base_acc,
        "standardization": ("per-session rest block (own mean/sd)" if standardize == "session"
                            else "train sessions only (frozen with the model)"),
        "standardize_mode": standardize,
        "calibration": {g: {"from": c["from"], "n": c["n"]} for g, c in ds["calib"].items()},
        "select_criterion": select,
        "best_epoch": best["epoch"], "epochs_run": len(hist["epoch"]),
        "std_thr": ds["std_thr"], "reject": ds["reject"],
        "per_file": ds["per_file"],
        "val_groups": sorted(set(ds["groups"][va].tolist())),
        "train_groups": sorted(set(ds["groups"][tr].tolist())),
        "n_params": int(net.n_params()),
        "features": list(FEATURE_NAMES),
    }
    return {"net": net, "mu": mu, "sd": sd,
            "hist": hist, "metrics": m, "metrics_default": m,
            "metrics_rest": m_rest,
            "metrics_val_best": m_val, "threshold": meta["threshold"],
            "yva": yva, "pva": pva,
            "baseline": base_m, "baseline_acc": base_acc, "ds": ds, "meta": meta}


def run_training(raw_paths, epochs=300, lr=1e-3, val_ratio=0.2, seed=42,
                 progress=None, stop_flag=None, settle=SETTLE_SECONDS):
    """从原始会话 CSV 走完整流程：构建数据集 → 训练 → 评估。"""
    ds = build_dataset(raw_paths, settle=settle)
    return train_from_dataset(ds, epochs=epochs, lr=lr, val_ratio=val_ratio,
                              seed=seed, progress=progress, stop_flag=stop_flag)


def export_dataset_csv(ds, path):
    cols = [f"{n}" for n in FEATURE_NAMES]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "label"] + [f"{c}_t{t}" for t in range(ds["X"].shape[1]) for c in cols])
        for i in range(len(ds["y"])):
            flat = ds["X"][i].reshape(-1)
            w.writerow([ds["groups"][i], int(ds["y"][i])] + [f"{v:.5f}" for v in flat])
    return path


def dataset_report_html(ds, meta, path):
    y = ds["y"]
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    rows = "".join(
        f"<tr><td>{r['file']}</td><td>{r['rows']}</td><td>{r['windows']}</td></tr>"
        for r in ds["per_file"])
    rej = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in ds["reject"].items())
    m = meta.get("metrics", {})
    m0 = meta.get("metrics_thr_default", {})
    mr = meta.get("metrics_thr_rest", {})
    mv = meta.get("metrics_thr_val_best", {})
    b = meta.get("baseline", {})
    th = meta.get("threshold", {})
    cal = meta.get("calibration", {})
    calib_rows = "".join(
        f"<tr><td>{g}</td><td>{c['from']}</td><td>{c['n']}</td></tr>" for g, c in cal.items())
    n_rej = sum(ds["reject"].values())
    feats = "、".join(meta.get("features") or FEATURE_NAMES)
    html = f"""<!doctype html><meta charset="utf-8">
<title>NeuroFocus 数据集报告</title>
<style>body{{font-family:system-ui,'Microsoft YaHei';max-width:900px;margin:32px auto;line-height:1.7}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#f2f5f9}}h2{{margin-top:28px;border-left:4px solid #3b82f6;padding-left:10px}}
.warn{{color:#b45309}}</style>
<h1>NeuroFocus 数据集报告</h1>
<p>生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>1. 概况</h2>
<table><tr><th>窗口总数</th><th>静息(0)</th><th>心算(1)</th><th>正类比例</th><th>特征维度</th><th>窗口长度</th></tr>
<tr><td>{len(y)}</td><td>{n0}</td><td>{n1}</td><td>{y.mean():.3f}</td>
<td>{ds['X'].shape[-1]}</td><td>{ds['X'].shape[1]} s</td></tr></table>
<p>特征：{feats}</p>
<h2>2. 按会话统计</h2>
<table><tr><th>文件</th><th>原始秒数</th><th>生成窗口</th></tr>{rows}</table>
<h2>3. 清洗与剔除</h2>
<p>伪迹阈值（raw_std &gt; {ds['std_thr']:.1f}；取稳健 4σ 界、全库 99.5 分位、4×中位数 三者较松者）；
接触不良判定用 TGAM 文档的「没戴好」码集合 {sorted(BAD_SIGNAL_CODES)}。</p>
<table><tr><th>原因</th><th>计数</th></tr>{rej}</table>
<p class="{'warn' if n_rej == 0 else ''}">合计剔除 {n_rej} 个窗口。
{'本次没有任何窗口被剔除——说明接触质量很好，属于合理结果，但报告中不应把「清洗」写成有实质作用的一步。'
 if n_rej == 0 else '剔除比例 %.1f%%，在协议允许的范围内。' % (100.0 * n_rej / max(1, n_rej + len(y)))}</p>
<h2>4. 划分与本人基线校准</h2>
<p>训练组：{', '.join(meta.get('train_groups', [])) or '—'}</p>
<p>验证组：{', '.join(meta.get('val_groups', [])) or '—'}</p>
<p>会话数 {meta.get('n_groups', len(cal))}；请求留出 {meta.get('val_ratio_requested', 0.2):.0%}，
实际留出 {meta.get('val_ratio_actual', float('nan')):.0%}
{'（会话太少，按会话分组只能整段留出，比例无法精确控制）' if meta.get('n_groups', 0) <= 2 else ''}。</p>
<table><tr><th>会话</th><th>基线来源</th><th>校准秒数</th></tr>{calib_rows}</table>
<p>标准化方式：{meta.get('standardization', '—')}。
{'默认只用训练会话的均值/标准差，训练完就固化进模型，部署时直接套用，佩戴者不需要先静息。'
 if meta.get('standardize_mode', 'train') == 'train'
 else '每个会话用自己开头的静息段估计均值/标准差，等价于「换人先静息 50 秒」。'}
下表仍列出各会话的静息段长度，供「换人部署时先做一次静息校准」参考。</p>
<h2>5. 模型与阈值</h2>
<p>参数量：{meta.get('n_params', '—')}；按 <b>{meta.get('select_criterion', 'auc')}</b> 选点，
最佳轮次 {meta.get('best_epoch', '—')}／共 {meta.get('epochs_run', '—')} 轮。</p>
<table><tr><th>判定阈值</th><th>阈值</th><th>accuracy</th><th>F1</th><th>AUC</th><th>性质</th></tr>
<tr><td>默认 0.50</td><td>0.50</td><td><b>{m.get('acc', float('nan')):.3f}</b></td>
<td><b>{m.get('f1', float('nan')):.3f}</b></td><td>{m.get('auc', float('nan')):.3f}</td>
<td><b>报告主表采用（固定模型，无需校准）</b></td></tr>
<tr><td>本人静息 90 分位</td><td>{th.get('rest', float('nan')):.2f}</td>
<td>{mr.get('acc', float('nan')):.3f}</td><td>{mr.get('f1', float('nan')):.3f}</td>
<td>{mr.get('auc', float('nan')):.3f}</td><td>换人部署时的可选校准（无标签）</td></tr>
<tr><td>验证集最优</td><td>{th.get('val_best', float('nan')):.2f}</td>
<td>{mv.get('acc', float('nan')):.3f}</td><td>{mv.get('f1', float('nan')):.3f}</td>
<td>{mv.get('auc', float('nan')):.3f}</td><td>乐观估计（阈值也用了验证标签）</td></tr>
</table>
<p>对照（brief Option A：3 秒滑动平均 + 阈值 75，<b>同一验证集</b>）：
accuracy {b.get('acc', float('nan')):.3f}、F1 {b.get('f1', float('nan')):.3f}、AUC {b.get('auc', float('nan')):.3f}。</p>
<p class="warn"><b>提示：</b>若本次使用的是 <code>*_demo_sim.csv</code> 模拟数据，
上述指标仅用于验证流程可运行，不能作为真实脑电的结论。</p>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# =============================================================================
# 8. 边缘部署导出
# =============================================================================
def export_tflite(net, mu, sd, path, window=WINDOW):
    """导出 TensorFlow Lite（int8 全整数量化）。需要本机已安装 tensorflow。"""
    import tensorflow as tf  # noqa
    K = tf.keras
    inp = K.Input(shape=(window, len(FEATURE_NAMES)), name="eeg_window")
    x = K.layers.Permute((2, 1))(inp)
    for i, (cin, cout) in enumerate([(net.c1.W.shape[1], net.c1.W.shape[0]),
                                     (net.c2.W.shape[1], net.c2.W.shape[0]),
                                     (net.c3.W.shape[1], net.c3.W.shape[0])]):
        layer = K.layers.Conv1D(cout, 3, padding="same", activation="relu", name=f"conv{i+1}")
        x = layer(x)
    x = K.layers.GlobalAveragePooling1D()(x)
    x = K.layers.Dense(1, activation="sigmoid", name="focus")(x)
    model = K.Model(inp, x)
    for layer, conv in zip([l for l in model.layers if isinstance(l, K.layers.Conv1D)],
                           [net.c1, net.c2, net.c3]):
        layer.set_weights([conv.W.transpose(2, 1, 0).astype("float32"), conv.b.astype("float32")])
    model.get_layer("focus").set_weights([net.fc.W.astype("float32"), net.fc.b.astype("float32")])

    def rep():
        rng = np.random.default_rng(0)
        for _ in range(100):
            yield [rng.standard_normal((1, window, len(FEATURE_NAMES))).astype("float32")]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    data = conv.convert()
    with open(path, "wb") as f:
        f.write(data)
    return path, len(data)


def export_edge_header(net, mu, sd, path, name="neurofocus_model"):
    """导出 int8 量化权重的 C 头文件，用于无 TF 环境下的 MCU 侧集成。"""
    lines = [f"/* Auto-generated by NeuroFocus. 3-layer 1-D CNN, {net.n_params()} params. */",
             "#ifndef NEUROFOCUS_MODEL_H", "#define NEUROFOCUS_MODEL_H", "#include <stdint.h>",
             f"#define NF_WINDOW {WINDOW}", f"#define NF_FEATURES {len(FEATURE_NAMES)}", ""]
    lines.append("static const float NF_MEAN[] = {" + ",".join(f"{v:.6f}f" for v in mu) + "};")
    lines.append("static const float NF_STD[]  = {" + ",".join(f"{v:.6f}f" for v in sd) + "};")
    for key, arr in net.state().items():
        flat = arr.reshape(-1)
        scale = float(np.abs(flat).max()) / 127.0 + 1e-12
        q = np.clip(np.round(flat / scale), -127, 127).astype(int)
        cname = key.replace(".", "_")
        lines.append(f"static const float {cname}_scale = {scale:.9f}f;")
        lines.append(f"static const int8_t {cname}[] = {{" + ",".join(str(v) for v in q) + "};")
    lines += ["#endif", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path




# =============================================================================
# 9. 图形控件（纯 tkinter Canvas 绘制，不依赖 matplotlib）
# =============================================================================
C_BG = "#0b1220"
C_GRID = "#1e293b"
C_TXT = "#94a3b8"
C_MAIN = "#38bdf8"
C_WARN = "#f59e0b"
C_OK = "#22c55e"
C_BAD = "#ef4444"
UI_FONT = ("Microsoft YaHei UI", 10)
UI_FONT_S = ("Microsoft YaHei UI", 9)
UI_FONT_B = ("Microsoft YaHei UI", 11, "bold")


class LinePlot(tk.Canvas):
    """多序列折线图：render([(名称, 颜色, 数值列表), ...], 标题)。"""

    def __init__(self, master, height=150, **kw):
        super().__init__(master, height=height, bg=C_BG, highlightthickness=0, **kw)

    def render(self, series, title="", ymin=None, ymax=None, ylabel=""):
        self.delete("all")
        w, h = max(self.winfo_width(), 200), max(self.winfo_height(), 80)
        pad_l, pad_r, pad_t, pad_b = 46, 10, 18, 16
        self.create_text(pad_l, 8, text=title, fill=C_TXT, anchor="w", font=UI_FONT_S)
        if ylabel:
            self.create_text(w - pad_r, 8, text=ylabel, fill=C_TXT, anchor="e", font=UI_FONT_S)
        data = [v for _, _, vs in series for v in vs if v == v]
        if not data:
            self.create_text(w / 2, h / 2, text="等待数据…", fill="#334155", font=UI_FONT_S)
            return
        lo = min(data) if ymin is None else ymin
        hi = max(data) if ymax is None else ymax
        if hi - lo < 1e-9:
            hi = lo + 1.0
        span = hi - lo
        lo, hi = lo - span * 0.08, hi + span * 0.08
        for i in range(5):
            y = pad_t + (h - pad_t - pad_b) * i / 4
            self.create_line(pad_l, y, w - pad_r, y, fill=C_GRID)
            self.create_text(pad_l - 6, y, text=f"{hi - (hi - lo) * i / 4:.0f}",
                             fill=C_TXT, anchor="e", font=("Consolas", 8))
        n_max = max((len(vs) for _, _, vs in series), default=0)
        for name, color, vs in series:
            if len(vs) < 2:
                continue
            pts = []
            for i, v in enumerate(vs):
                x = pad_l + (w - pad_l - pad_r) * (i / max(1, n_max - 1))
                y = pad_t + (h - pad_t - pad_b) * (1 - (v - lo) / (hi - lo))
                pts += [x, y]
            self.create_line(*pts, fill=color, width=2, smooth=False)
            self.create_text(w - pad_r, h - 6, text=name, fill=color, anchor="se", font=UI_FONT_S)


class BarGauge(tk.Canvas):
    """0–100 水平条形仪表。"""

    def __init__(self, master, height=64, label="", color=C_MAIN, **kw):
        super().__init__(master, height=height, bg=C_BG, highlightthickness=0, **kw)
        self.label, self.color = label, color
        self.value = None
        self.unit = ""

    def render(self, value, sub=""):
        self.delete("all")
        w, h = max(self.winfo_width(), 160), max(self.winfo_height(), 50)
        self.value = value
        self.create_text(8, 12, text=self.label, fill=C_TXT, anchor="w", font=UI_FONT_S)
        txt = "—" if value is None else f"{value:.0f}"
        self.create_text(w - 8, 12, text=txt + self.unit, fill=self.color, anchor="e",
                         font=("Consolas", 15, "bold"))
        y0, y1 = 32, h - 18
        self.create_rectangle(8, y0, w - 8, y1, outline=C_GRID, fill="#111c2e")
        if value is not None:
            frac = max(0.0, min(1.0, value / 100.0))
            if frac > 0:
                self.create_rectangle(8, y0, 8 + (w - 16) * frac, y1, outline="", fill=self.color)
        if sub:
            self.create_text(8, h - 7, text=sub, fill=C_TXT, anchor="w", font=("Consolas", 8))


class Heatmap(tk.Canvas):
    """2×2 混淆矩阵。"""

    def __init__(self, master, size=190, **kw):
        super().__init__(master, width=size, height=size, bg=C_BG, highlightthickness=0, **kw)

    def render(self, cm):
        self.delete("all")
        s = min(self.winfo_width(), self.winfo_height()) or 190
        pad, cell = 44, (min(self.winfo_width(), self.winfo_height()) - 44) / 2
        cell = max(40, (s - pad) / 2)
        names = ["静息 0", "专注 1"]
        total = max(1, sum(sum(r) for r in cm))
        for i in range(2):
            for j in range(2):
                v = cm[i][j]
                frac = v / total
                col = f"#{int(30 + 90 * frac):02x}{int(60 + 120 * frac):02x}{int(110 + 130 * frac):02x}"
                x0, y0 = pad + j * cell, pad - 26 + i * cell
                self.create_rectangle(x0, y0, x0 + cell - 4, y0 + cell - 4, fill=col, outline=C_GRID)
                self.create_text(x0 + cell / 2, y0 + cell / 2, text=str(v), fill="#e2e8f0",
                                 font=("Consolas", 16, "bold"))
        for j, n in enumerate(names):
            self.create_text(pad + j * cell + cell / 2, pad - 36, text=n, fill=C_TXT, font=UI_FONT_S)
        for i, n in enumerate(names):
            self.create_text(pad - 8, pad - 26 + i * cell + cell / 2, text=n, fill=C_TXT,
                             anchor="e", font=UI_FONT_S)
        self.create_text(pad - 8, pad - 48, text="预测→", fill="#475569", anchor="e", font=("Consolas", 8))
        self.create_text(6, pad - 26 + cell, text="真实", fill="#475569", angle=90, font=("Consolas", 8))


# =============================================================================
# 10. 应用主体
# =============================================================================
class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.source = None
        self.recorder = None
        self.recent = deque(maxlen=1200)      # 最近样本（秒）
        self.model, self.mu, self.sd, self.meta = None, None, None, None
        self.ds = None
        self.infer_on = False
        self.infer_buf = deque(maxlen=WINDOW)
        self.infer_probs = deque(maxlen=180)
        self.infer_true = deque(maxlen=180)
        self.infer_labels = deque(maxlen=180)
        # 实时「本人静息校准」：收集 CALIB_LIVE_SECONDS 秒静息特征，估计 mu/sd 与判定阈值
        self.calib_on = False
        self.calib_rows = []
        self.calib_probs = []
        self.thr_cal = 0.5
        self.calib_buf = deque(maxlen=WINDOW)
        self.train_thread = None
        self.train_msgs = queue.Queue()
        self.stop_flag = {"stop": False}
        self.raw_files = []
        root.title("NeuroFocus · 单通道 EEG 专注度分析（BMEG3103 Final Project）")
        root.geometry("1180x820")
        root.minsize(980, 640)
        self._style()
        self.build_ui()
        self.refresh_raw_list()
        self._fit_window()
        self.root.after(150, self.tick)

    # ---------- 基础 ----------
    def _fit_window(self):
        """按屏幕可用区域和实际内容需求收放窗口。

        不同 Windows 机器的分辨率/缩放差别很大：1180x820 在高 DPI 笔记本上可能
        超出屏幕，底部采集日志就被截掉。这里同时考虑"控件真实需要的尺寸"和
        "屏幕能给的尺寸"，取较小者并居中。
        """
        root = self.root
        root.update_idletasks()
        need_w, need_h = root.winfo_reqwidth(), root.winfo_reqheight()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = min(max(1180, need_w), max(960, sw - 80))
        h = min(max(820, need_h), max(600, sh - 120))
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.minsize(min(1040, w), min(700, h))
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TNotebook.Tab", font=UI_FONT, padding=(14, 7))
        st.configure("TButton", font=UI_FONT, padding=(8, 4))
        st.configure("TLabel", font=UI_FONT)
        st.configure("TEntry", font=UI_FONT)
        st.configure("Header.TLabel", font=UI_FONT_B)

    def build_ui(self):
        head = ttk.Frame(self.root, padding=(12, 8, 12, 2))
        head.pack(fill="x")
        ttk.Label(head, text="NeuroFocus", font=("Microsoft YaHei UI", 15, "bold")).pack(side="left")
        ttk.Label(head, text="  单通道 FP1 EEG · 1-D CNN(3 层) · 专注度识别",
                  foreground="#475569").pack(side="left")
        self.lbl_src = ttk.Label(head, text="未连接", foreground="#b91c1c")
        self.lbl_src.pack(side="right")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=6)
        self.tab_device = DeviceTab(self.nb, self)
        self.tab_capture = CaptureTab(self.nb, self)
        self.tab_dataset = DatasetTab(self.nb, self)
        self.tab_train = TrainTab(self.nb, self)
        self.tab_live = LiveTab(self.nb, self)
        self.tab_about = AboutTab(self.nb, self)
        for t, name in [(self.tab_device, "① 设备与信号"), (self.tab_capture, "② 采集与标注"),
                        (self.tab_dataset, "③ 数据集"), (self.tab_train, "④ 训练与评估"),
                        (self.tab_live, "⑤ 实时推演"), (self.tab_about, "⑥ 说明/AI 使用")]:
            self.nb.add(t, text=name)

        self.status = tk.StringVar(value="就绪。默认串口 COM7，57600 波特。")
        if np is None:
            self.status.set("⚠ 未找到 numpy（%s）：构建数据集与训练不可用，"
                            "请安装匹配版本的 numpy 或放入 vendor/。"
                            % sys.version.split()[0])
        bar = ttk.Frame(self.root, padding=(12, 2, 12, 6))
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self.status, foreground="#334155").pack(side="left")

    def say(self, msg):
        self.status.set(f"{time.strftime('%H:%M:%S')}  {msg}")

    def attach_source(self, src, name):
        self.detach_source()
        self.source = src
        src.start()
        self.lbl_src.config(text=f"● {name}", foreground="#15803d")
        self.say(f"已连接：{name}")

    def detach_source(self):
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
        self.source = None
        self.lbl_src.config(text="未连接", foreground="#b91c1c")

    def start_recording(self, path):
        self.recorder = Recorder(path)
        return path

    def stop_recording(self):
        if self.recorder:
            n = self.recorder.n
            self.recorder.close()
            self.recorder = None
            return n
        return 0

    def refresh_raw_list(self):
        self.raw_files = sorted(p for p in glob.glob(os.path.join(RAW_DIR, "*.csv")) if is_raw_csv(p))
        if hasattr(self, "tab_dataset"):
            self.tab_dataset.reload(self.raw_files)

    # ---------- 主循环 ----------
    def tick(self):
        try:
            self._tick_once()
        except Exception as e:
            traceback.print_exc()
            self.say(f"运行时异常：{e}")
        finally:
            self.root.after(120, self.tick)

    def start_calibration(self):
        """开始实时静息校准：请被试保持静息 CALIB_LIVE_SECONDS 秒。"""
        if self.model is None:
            return
        self.calib_rows, self.calib_probs = [], []
        self.calib_buf.clear()
        self.calib_on = True

    def _calib_step(self, s):
        """校准期间：累计每秒特征，并用当前（未校准的）基线算一组窗口概率。"""
        self.calib_rows.append(frame_features([s])[0])
        self.calib_buf.append(s)
        if len(self.calib_buf) == WINDOW:
            X = frame_features(list(self.calib_buf))
            z = zscore(X[None], self.mu, self.sd)
            self.calib_probs.append(float(self.model.predict_proba(z)[0]))
        if len(self.calib_rows) >= CALIB_LIVE_SECONDS:
            F = np.stack(self.calib_rows).astype(np.float32)
            self.mu = F.mean(axis=0).astype(np.float32)
            self.sd = safe_sd(F)
            self.thr_cal = rest_threshold(self.calib_probs)
            self.calib_on = False
            self.calib_rows, self.calib_probs = [], []
            self.infer_probs.clear()
            self.tab_live.on_calibrated(int(F.shape[0]))
            self.say(f"静息校准完成：阈值设为 {self.thr_cal:.2f}")

    def _tick_once(self):
        drained = 0
        while drained < 80 and not self.q.empty():
            try:
                s = self.q.get_nowait()
            except queue.Empty:
                break
            self.recent.append(s)
            drained += 1
            if self.recorder is not None:
                self.recorder.add(s)
            if self.calib_on and self.model is not None:
                self._calib_step(s)
                continue
            if self.infer_on and self.model is not None:
                self.infer_buf.append(s)
                if len(self.infer_buf) == WINDOW:
                    X = frame_features(list(self.infer_buf))
                    if X.shape[0] == WINDOW:
                        z = zscore(X[None], self.mu, self.sd)
                        p = float(self.model.predict_proba(z)[0])
                        self.infer_probs.append(p)
                        self.infer_labels.append(s.label)
        try:
            while True:
                kind, payload = self.train_msgs.get_nowait()
                self.tab_train.on_msg(kind, payload)
        except queue.Empty:
            pass
        for t in (self.tab_device, self.tab_capture, self.tab_train, self.tab_live):
            try:
                t.tick()
            except Exception as e:  # GUI 不应因单次绘制异常而崩溃
                self.say(f"绘制异常：{e}")


# =============================================================================
# 11. 页面：设备与信号
# =============================================================================
class DeviceTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.raw_hist = deque(maxlen=1024)
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="数据源").grid(row=0, column=0, sticky="w")
        self.src_var = tk.StringVar(value="串口（蓝牙接收器）")
        cb = ttk.Combobox(top, textvariable=self.src_var, width=20, state="readonly",
                          values=["串口（蓝牙接收器）", "模拟器（无硬件演示）"])
        cb.grid(row=0, column=1, padx=6)
        ttk.Label(top, text="端口").grid(row=0, column=2, padx=(12, 2))
        self.port_var = tk.StringVar(value=DEFAULT_PORT)
        ports = []
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            ports = [DEFAULT_PORT]
        self.cb_port = ttk.Combobox(top, textvariable=self.port_var, width=9, values=ports or [DEFAULT_PORT])
        self.cb_port.grid(row=0, column=3)
        ttk.Label(top, text="波特率").grid(row=0, column=4, padx=(12, 2))
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        ttk.Entry(top, textvariable=self.baud_var, width=8).grid(row=0, column=5)
        self.btn = ttk.Button(top, text="连接", command=self.toggle, width=10)
        self.btn.grid(row=0, column=6, padx=12)
        ttk.Button(top, text="刷新端口", command=self.refresh_ports).grid(row=0, column=7)
        self.diag = ttk.Label(top, text="", foreground="#64748b")
        self.diag.grid(row=0, column=8, padx=10)

        mid = ttk.Frame(self)
        mid.pack(fill="x", pady=8)
        self.g_sig = BarGauge(mid, label="Signal（0=接触最好，>100 需重新佩戴）", color=C_OK)
        self.g_sig.unit = ""
        self.g_att = BarGauge(mid, label="设备 Attention", color=C_MAIN)
        self.g_med = BarGauge(mid, label="设备 Meditation", color="#a78bfa")
        for i, g in enumerate((self.g_sig, self.g_att, self.g_med)):
            g.grid(row=0, column=i, sticky="ew", padx=4)
            mid.columnconfigure(i, weight=1)

        self.raw_plot = LinePlot(self, height=170)
        self.raw_plot.pack(fill="x", pady=(4, 6))
        self.att_plot = LinePlot(self, height=150)
        self.att_plot.pack(fill="both", expand=True)

    def refresh_ports(self):
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            ports = []
        if ports:
            self.cb_port["values"] = ports
            if self.port_var.get() not in ports:
                self.port_var.set(ports[0])
        self.app.say(f"检测到串口：{', '.join(ports) if ports else '（无）'}")

    def toggle(self):
        if self.app.source is not None:
            self.app.detach_source()
            self.btn.config(text="连接")
            return
        if self.src_var.get().startswith("模拟"):
            src = SimulatorSource(self.app.q)
            self.app.attach_source(src, "模拟器")
        else:
            port, baud = self.port_var.get().strip(), int(self.baud_var.get() or DEFAULT_BAUD)
            try:
                src = SerialSource(self.app.q, port, baud)
                self.app.attach_source(src, f"{port} @{baud}")
            except Exception as e:
                messagebox.showerror("串口打开失败", f"{e}\n\n请确认接收器已插入、端口号正确、未被其他软件占用。")
                return
        self.btn.config(text="断开")

    def tick(self):
        src = self.app.source
        if src is None:
            self.diag.config(text="")
            return
        p = getattr(src, "parser", None)
        if p is not None:
            raw = list(p.raw_deque)
            n_big, n_raw, bad = p.n_big, p.n_raw, p.bad
            self.raw_hist = raw[-1024:]
            self.diag.config(text=f"大包 {n_big} · 原始包 {n_raw} · 坏包 {bad} · {getattr(src, 'message', '')}")
        elif isinstance(src, SimulatorSource):
            self.diag.config(text="模拟器运行中（数据为合成，非真实 EEG）")
        self.raw_plot.render([("raw µV", C_MAIN, list(self.raw_hist))], "原始 EEG 波形（最近 2 秒 @512Hz）")
        if self.app.recent:
            last = self.app.recent[-1]
            self.g_sig.render(last.signal, "0 为最佳")
            self.g_att.render(last.attention)
            self.g_med.render(last.meditation)
            att = [s.attention for s in self.app.recent][-180:]
            med = [s.meditation for s in self.app.recent][-180:]
            self.att_plot.render([("Attention", C_MAIN, att), ("Meditation", "#a78bfa", med)],
                                 "设备内建指标（1 Hz，最近 3 分钟）", ymin=0, ymax=100)


# =============================================================================
# 12. 页面：采集与标注
# =============================================================================
class CaptureTab(ttk.Frame):
    """采集与标注页：按协议交替采集「闭眼静息 / 闭眼默算」。

    心算任务刻意不与被试交互——被试一旦看屏幕读题、用手敲键盘，FP1 上就会叠加
    眼动（读题、眨眼）与手部肌电；这些成分与「专注」无关，却比专注本身更容易
    被模型抓住。第一版 GUI 出题采到的那份数据里，raw_max 单特征 AUC 就有 0.936、
    raw_std 0.913，而设备 Attention 反而在静息段更高（61.5 vs 49.6），
    说明分类器学到的很可能是「睁眼 + 敲键盘」而不是认知负荷。
    改成「闭眼从 1000 连续减 7」后，两种状态都是闭眼、都不动手，眼动与肌电在
    两类之间基本对齐，剩下的差异才更接近认知负荷本身。

    v3 协议（闭眼、双态都不动手）实测正好验证了这一点：raw_std 的单特征 AUC 从
    0.913（v2，睁眼+敲键盘）掉到 0.318，方向还反了——静息时幅度更大。所以 raw_std
    与 raw_max 现在只作为弱特征保留，剔除规则也不再按它们硬砍（见 build_dataset）。

    行为校验改为段末「报数」：被试说出自己减到的数，操作者记录一次（可选），
    写进 *_tasks.csv，只用于判断「这段心算是否真的做了」，不参与训练。
    """

    MATH_TEXT = "1000 − 7"
    IDLE_TEXT = "未开始"
    REST_TEXT = "闭眼静息"
    TIP_IDLE = "心算段：闭眼从 1000 连续减 7（不出声、不动手）；静息段：闭眼放松"
    TIP_MATH = "从 1000 开始连续减 7：默算，不出声、不睁眼、不动手；减到负数继续减"
    TIP_REST = "闭眼放松，不要思考题目；头与手保持不动"

    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.plan, self.idx, self.phase_end = [], 0, 0.0
        self.running = False
        self.path = None
        self.t_start = 0.0
        self.reports = []          # 每段心算的报数（None = 操作者没记）
        self._panel_state = None
        self.task_file = None
        self.task_writer = None

        box = ttk.LabelFrame(self, text="采集协议（标签只来自本协议，绝不使用设备 Attention 作为标签）", padding=8)
        box.pack(fill="x")
        ttk.Label(box, text="被试编号").grid(row=0, column=0, sticky="w")
        self.subj = tk.StringVar(value="S01")
        ttk.Entry(box, textvariable=self.subj, width=10).grid(row=0, column=1, padx=4)
        ttk.Label(box, text="每段秒数").grid(row=0, column=2, padx=(14, 0))
        self.secs = tk.StringVar(value="60")
        ttk.Spinbox(box, from_=20, to=180, increment=10, textvariable=self.secs,
                    width=6).grid(row=0, column=3, padx=4)
        ttk.Label(box, text="段数").grid(row=0, column=4, padx=(14, 0))
        self.nblocks = tk.StringVar(value="6")
        ttk.Spinbox(box, from_=2, to=12, increment=2, textvariable=self.nblocks,
                    width=5).grid(row=0, column=5, padx=4)
        self.btn = ttk.Button(box, text="开始采集", command=self.toggle)
        self.btn.grid(row=0, column=6, padx=(16, 6))
        ttk.Button(box, text="用模拟器试运行", command=self.use_sim).grid(row=0, column=7)
        ttk.Label(box, text="先静息后心算交替；每段前 10 秒为稳定期，建模时自动丢弃",
                  foreground="#64748b").grid(row=1, column=0, columnspan=8, sticky="w", pady=(6, 0))

        state = ttk.Frame(self)
        state.pack(fill="x", pady=(8, 4))
        self.lbl_phase = ttk.Label(state, text="未开始", font=("Microsoft YaHei UI", 18, "bold"),
                                   foreground="#0f172a", width=11)
        self.lbl_phase.pack(side="left", padx=(6, 0))
        self.bar = ttk.Progressbar(state, mode="determinate", maximum=100, length=240)
        self.bar.pack(side="left", padx=12)
        self.lbl_info = ttk.Label(state, text="", foreground="#475569")
        self.lbl_info.pack(side="left", padx=8)

        taskbox = ttk.LabelFrame(self, text="心算任务（闭眼默算：被试不看屏幕、不敲键盘）", padding=8)
        taskbox.pack(fill="x", pady=6)
        # 固定尺寸的容器 + 负数字号（以像素计）：高 DPI 下字体不会被放大，面板也不会跳动
        holder = tk.Frame(taskbox, width=250, height=70, bg="#f8fafc",
                          highlightbackground="#e2e8f0", highlightthickness=1)
        holder.grid(row=0, column=0, rowspan=2, sticky="w")
        holder.grid_propagate(False)
        holder.pack_propagate(False)  # 子控件用 pack，需同时关掉 pack 传播，否则面板随文字宽度跳动
        self.q_text = tk.Label(holder, text=self.IDLE_TEXT, bg="#f8fafc", fg="#0f172a",
                               font=("Microsoft YaHei UI", -30, "bold"))
        self.q_text.pack(expand=True)
        self.q_tip = ttk.Label(taskbox, text=self.TIP_IDLE, font=("Microsoft YaHei UI", 11),
                               foreground="#334155")
        self.q_tip.grid(row=0, column=1, sticky="w", padx=(14, 0))

        ops = ttk.Frame(taskbox)
        ops.grid(row=1, column=1, sticky="w", padx=(14, 0), pady=(8, 0))
        ttk.Label(ops, text="段末记录报数（选填）").pack(side="left")
        self.rep = tk.StringVar()
        self.rep_entry = ttk.Entry(ops, textvariable=self.rep, width=8, justify="center")
        self.rep_entry.pack(side="left", padx=(6, 4))
        self.rep_entry.bind("<Return>", lambda e: self.record_report())
        self.rep_entry.config(state="disabled")
        ttk.Button(ops, text="记录", width=6, command=self.record_report).pack(side="left")
        self.q_stats = ttk.Label(ops, text="已记录 0 段", foreground="#64748b")
        self.q_stats.pack(side="left", padx=(14, 0))

        # 先放日志（固定在底部），再让曲线吃掉剩余空间：
        # 窗口变矮时被压缩的是曲线而不是日志，小屏上也能看到采集记录。
        self.log = tk.Text(self, height=7, font=("Consolas", 9), bg="#0b1220", fg="#cbd5e1",
                           insertbackground="#cbd5e1", relief="flat")
        self.log.pack(side="bottom", fill="x")
        self.plot = LinePlot(self, height=150)
        self.plot.pack(side="top", fill="both", expand=True, pady=4)

    def use_sim(self):
        if self.app.source is None:
            self.app.attach_source(SimulatorSource(self.app.q), "模拟器")
            self.app.tab_device.btn.config(text="断开")

    def build_plan(self):
        try:
            n = int(self.nblocks.get())
            secs = float(self.secs.get())
        except ValueError:
            n, secs = 6, 60.0
        return [(i % 2, secs) for i in range(n)]   # 0=静息, 1=心算

    def toggle(self):
        if self.running:
            self.finish("手动结束")
            return
        if self.app.source is None:
            messagebox.showwarning("未连接设备", "请先在「① 设备与信号」连接串口或模拟器。")
            return
        self.plan = self.build_plan()
        self.idx = 0
        self.running = True
        self.t_start = time.time()
        self.reports = []
        self._panel_state = None
        self.rep.set("")
        self.rep_entry.config(state="normal")
        subj = self.subj.get().strip() or "S00"
        self.path = os.path.join(RAW_DIR, f"{subj}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.open_task_file()
        self.update_report_stats()
        self.app.start_recording(self.path)
        self.btn.config(text="结束并保存")
        self.app.source.label = self.plan[0][0]
        self.phase_end = time.time() + self.plan[0][1]
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] 开始 {len(self.plan)} 段，每段 "
                               f"{self.plan[0][1]:.0f}s → {self.path}\n")
        self.log.see("end")
        self.app.say("采集开始：两种状态都闭眼。心算段从 1000 连续减 7，默算、不出声、不动手。")

    def finish(self, why):
        if not self.running:
            return
        self.running = False
        self.btn.config(text="开始采集")
        self.app.source.label = -1 if self.app.source else -1
        n = self.app.stop_recording()
        self.report_quality()
        self.close_task_file()
        self.set_panel("idle", "已结束")
        self.rep_entry.config(state="disabled")
        self.lbl_phase.config(text="已结束", foreground="#64748b")
        self.bar["value"] = 0
        self.app.refresh_raw_list()
        msg = f"[{time.strftime('%H:%M:%S')}] {why}：写入 {n} 行 → {os.path.basename(self.path or '')}"
        self.log.insert("end", msg + "\n")
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] 心算行为校验：{self.report_summary()}\n")
        self.log.see("end")
        self.app.say(msg + "  " + self.report_summary())
        self.lbl_info.config(text=f"文件：{self.path}")

    def tick(self):
        if not self.running:
            return
        now = time.time()
        if now >= self.phase_end:
            self.idx += 1
            if self.idx >= len(self.plan):
                self.finish("协议完成")
                return
            lab, secs = self.plan[self.idx]
            self.app.source.label = lab
            self.phase_end = now + secs
            self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] 第 {self.idx + 1} 段："
                                   f"{'心算' if lab else '静息'}（{secs:.0f}s）\n")
            self.log.see("end")
        lab, secs = self.plan[self.idx]
        self.set_panel("math" if lab else "rest")
        remain = max(0.0, self.phase_end - now)
        self.lbl_phase.config(text=("心算 FOCUS" if lab else "静息 REST"),
                              foreground=("#b91c1c" if lab else "#15803d"))
        self.bar["value"] = 100 * (1 - remain / max(1e-6, secs))
        self.lbl_info.config(text=f"第 {self.idx + 1}/{len(self.plan)} 段 · 剩余 {remain:4.1f}s · "
                                  f"已记录 {self.app.recorder.n if self.app.recorder else 0} 行")
        att = [s.attention for s in self.app.recent][-240:]
        self.plot.render([("设备 Attention（仅参考）", C_MAIN if not lab else C_BAD, att)],
                         "采集过程实时曲线（标签由协议决定，与这条曲线无关）", ymin=0, ymax=100)

    # ---- 任务面板 ----
    def set_panel(self, state, text=None):
        """state: idle / math / rest。只在切换时重绘，避免每帧闪动。"""
        if state == self._panel_state and text is None:
            return
        self._panel_state = state
        if state == "math":
            self.q_text.config(text=self.MATH_TEXT, fg="#b91c1c",
                               font=("Consolas", -36, "bold"))
            self.q_tip.config(text=self.TIP_MATH)
        elif state == "rest":
            self.q_text.config(text=self.REST_TEXT, fg="#15803d",
                               font=("Microsoft YaHei UI", -32, "bold"))
            self.q_tip.config(text=self.TIP_REST)
        else:
            self.q_text.config(text=text or self.IDLE_TEXT, fg="#64748b",
                               font=("Microsoft YaHei UI", -26, "bold"))
            self.q_tip.config(text=self.TIP_IDLE)

    # ---- 行为校验（段末报数） ----
    def quality_path(self):
        if not self.path:
            return None
        stem = os.path.splitext(os.path.basename(self.path))[0]
        return os.path.join(RAW_DIR, f"{stem}_quality.txt")

    def report_quality(self):
        """采集一结束就打印体检结论，避免拿着有眼/手伪迹的数据去训练。"""
        rows = list(self.app.recorder.rows) if self.app.recorder else []
        lines, verdict = quality_lines(rows)
        head = {"ok": "体检通过", "warn": "体检告警", "unknown": "体检信息不足"}[verdict]
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {head}\n")
        for ln in lines:
            self.log.insert("end", f"    {ln}\n")
        self.log.see("end")
        try:
            with open(self.quality_path(), "w", encoding="utf-8", newline="") as f:
                f.write(head + "\n" + "\n".join(lines) + "\n")
        except OSError:
            pass
        if verdict == "warn":
            self.app.say("这次采集的体检有告警，详见日志或 quality.txt。")

    def task_path(self):
        if not self.path:
            return None
        stem = os.path.splitext(os.path.basename(self.path))[0]
        return os.path.join(RAW_DIR, f"{stem}_tasks.csv")

    def open_task_file(self):
        self.close_task_file()
        path = self.task_path()
        if not path:
            return
        try:
            self.task_writer = open(path, "w", encoding="utf-8", newline="")
            self.task_file = path
            self.task_writer.write("t,qid,question,answer,result\n")
            self.task_writer.flush()
        except OSError as e:
            self.task_writer = None
            self.log.insert("end", f"报数记录文件无法写入：{e}\n")

    def close_task_file(self):
        if self.task_writer is not None:
            try:
                self.task_writer.close()
            except OSError:
                pass
        self.task_writer = None

    def record_report(self):
        """记录被试在本段心算里减到的数（-1/留空表示没记）。"""
        if not self.running:
            return
        text = self.rep.get().strip()
        val = None
        if text:
            try:
                val = int(text)
            except ValueError:
                self.q_stats.config(text="请填整数，或留空")
                return
        self.reports.append(val)
        idx = len(self.reports)
        if self.task_writer is not None:
            t = round(time.time() - self.t_start, 2)
            q = f"从1000连续减7（第{idx}段心算）"
            try:
                self.task_writer.write(f"{t},{idx},{q},{val if val is not None else ''},-1\n")
                self.task_writer.flush()
            except OSError:
                pass
        self.rep.set("")
        self.update_report_stats()
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] 记下第 {idx} 段报数："
                               f"{val if val is not None else '未报'}\n")
        self.log.see("end")
        self.rep_entry.focus_set()

    def update_report_stats(self):
        n_ok = sum(1 for v in self.reports if v is not None)
        self.q_stats.config(text=f"已记录 {len(self.reports)} 段 · 有报数 {n_ok} 段")

    def report_summary(self):
        if not self.reports:
            return "未记录报数（不影响建模，但少了行为证据）"
        vals = [str(v) if v is not None else "未报" for v in self.reports]
        return f"记下 {len(self.reports)} 段心算报数：" + " / ".join(vals)

# =============================================================================
# 13. 页面：数据集
# =============================================================================
class DatasetTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.ds = None
        self.files = []

        left = ttk.LabelFrame(self, text="原始采集文件（data/raw/*.csv）", padding=8)
        left.pack(side="left", fill="both", expand=True)
        self.lb = tk.Listbox(left, selectmode="extended", height=16, font=("Consolas", 9),
                             activestyle="none")
        self.lb.pack(fill="both", expand=True)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text="全选", command=lambda: self.lb.select_set(0, "end")).pack(side="left")
        ttk.Button(row, text="刷新", command=self.app.refresh_raw_list).pack(side="left", padx=6)
        ttk.Button(row, text="生成演示数据（模拟）", command=self.make_demo).pack(side="left")

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        cfg = ttk.LabelFrame(right, text="数据集参数", padding=8)
        cfg.pack(fill="x")
        ttk.Label(cfg, text="窗口长度（秒）").grid(row=0, column=0, sticky="w")
        ttk.Label(cfg, text=str(WINDOW), font=UI_FONT_B).grid(row=0, column=1, padx=6)
        ttk.Label(cfg, text="稳定期丢弃（秒）").grid(row=1, column=0, sticky="w", pady=4)
        self.settle = tk.StringVar(value=str(SETTLE_SECONDS))
        ttk.Entry(cfg, textvariable=self.settle, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(cfg, text="验证集比例").grid(row=2, column=0, sticky="w")
        self.valr = tk.StringVar(value="0.2")
        ttk.Entry(cfg, textvariable=self.valr, width=8).grid(row=2, column=1, sticky="w")
        self.btns = ttk.Frame(cfg)
        self.btns.grid(row=0, column=2, rowspan=3, padx=16)
        ttk.Button(self.btns, text="构建数据集", command=self.build).pack(fill="x")
        ttk.Button(self.btns, text="导出 dataset.csv", command=self.save_csv).pack(fill="x", pady=4)
        ttk.Button(self.btns, text="生成 HTML 报告", command=self.save_report).pack(fill="x")

        self.txt = tk.Text(right, height=20, font=("Consolas", 9), bg="#0b1220", fg="#cbd5e1",
                           relief="flat", wrap="word")
        self.txt.pack(fill="both", expand=True, pady=8)
        self.log("数据集尚未构建。选择至少 2 个会话文件后点「构建数据集」。\n"
                 "提示：训练/验证按「会话文件」分组划分，避免同一段连续信号泄漏到两边。")

    def log(self, s, clear=False):
        if clear:
            self.txt.delete("1.0", "end")
        self.txt.insert("end", s + "\n")
        self.txt.see("end")

    def reload(self, files):
        """原始文件列表变化（启动/采集结束/生成演示数据）时刷新，并作废旧数据集。"""
        self.files = files
        self.lb.delete(0, "end")
        for f in files:
            self.lb.insert("end", os.path.basename(f))
        if files:
            self.lb.select_set(0, "end")
        self.invalidate()

    def invalidate(self):
        """文件列表变了，之前构建的数据集不再对应当前列表，必须重建。"""
        had = self.ds is not None or self.app.ds is not None
        self.ds = None
        self.app.ds = None
        if had:
            self.log("文件列表已变化：之前构建的数据集已作废，请重新点「构建数据集」。")
            try:
                self.app.tab_train.on_dataset(None)
            except AttributeError:
                pass

    def selected_paths(self):
        idx = self.lb.curselection()
        if not idx:
            # 列表刷新会清掉选中项；没勾选时按"全选"处理，避免训练页莫名提示未选择
            return list(self.files)
        return [self.files[i] for i in idx]

    def make_demo(self):
        paths = make_demo_set(RAW_DIR)
        self.app.refresh_raw_list()
        self.log(f"已生成 {len(paths)} 个模拟会话（文件名含 _demo_sim，仅用于流程演示）：", clear=True)
        for p in paths:
            self.log("  " + os.path.basename(p))
        self.app.say("模拟数据已生成。它不能替代真实采集。")

    def build(self):
        paths = self.selected_paths()
        if not paths:
            messagebox.showwarning("没有采集文件",
                                   "data/raw 里还没有会话 CSV。\n"
                                   "请先在「② 采集与标注」采集，或点「生成演示数据」。")
            return
        try:
            ds = build_dataset(paths, settle=int(float(self.settle.get() or SETTLE_SECONDS)))
            tr, va = split_dataset(ds, val_ratio=float(self.valr.get() or 0.2))
        except Exception as e:
            messagebox.showerror("构建失败", f"{type(e).__name__}: {e}")
            return
        self.ds, self.app.ds = ds, ds
        y = ds["y"]
        self.app.tab_train.on_dataset(ds)
        self.log("", clear=True)
        self.log(f"窗口总数 {len(y)}   静息 0 = {int((y == 0).sum())}   心算 1 = {int((y == 1).sum())}"
                 f"   正类比例 {y.mean():.3f}")
        self.log(f"特征维度 {ds['X'].shape[-1]}   窗口长度 {ds['X'].shape[1]} s   "
                 f"伪迹阈值 raw_std>{ds['std_thr']:.0f}")
        self.log("按会话：")
        for r in ds["per_file"]:
            self.log(f"  {r['file']:<34} 原始 {r['rows']:>5} s   窗口 {r['windows']:>5}")
        self.log("剔除统计：" + "，".join(f"{k}={v}" for k, v in ds["reject"].items()))
        self.log(f"划分：训练 {int(tr.sum())} / 验证 {int(va.sum())}（验证会话："
                 f"{', '.join(sorted(set(ds['groups'][va].tolist())))}）")
        self.app.say("数据集已构建，可进入「④ 训练与评估」。")

    def save_csv(self):
        if self.ds is None:
            messagebox.showinfo("提示", "请先构建数据集。")
            return
        export_dataset_csv(self.ds, DATASET_PATH)
        self.log(f"已导出 {DATASET_PATH}")
        self.app.say("dataset.csv 已导出。")

    def save_report(self):
        if self.ds is None:
            messagebox.showinfo("提示", "请先构建数据集。")
            return
        meta = self.app.meta or {}
        p = dataset_report_html(self.ds, meta, os.path.join(DATA_DIR, "dataset_report.html"))
        self.log(f"已生成 {p}")
        try:
            os.startfile(p)  # noqa: 仅 Windows 桌面环境
        except Exception:
            pass

    def tick(self):
        pass


# =============================================================================
# 14. 页面：训练与评估
# =============================================================================
class TrainTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.thread = None
        self.stop = False
        self.hist = None
        self.result = None

        top = ttk.LabelFrame(self, text="训练配置（1-D CNN：3 × Conv1D + GAP + Dense1）", padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="最大轮数").grid(row=0, column=0)
        self.epochs = tk.StringVar(value="300")
        ttk.Entry(top, textvariable=self.epochs, width=7).grid(row=0, column=1, padx=4)
        ttk.Label(top, text="学习率").grid(row=0, column=2, padx=(12, 0))
        self.lr = tk.StringVar(value="0.001")
        ttk.Entry(top, textvariable=self.lr, width=9).grid(row=0, column=3, padx=4)
        ttk.Label(top, text="批大小").grid(row=0, column=4, padx=(12, 0))
        self.bs = tk.StringVar(value="32")
        ttk.Entry(top, textvariable=self.bs, width=6).grid(row=0, column=5, padx=4)
        self.btn = ttk.Button(top, text="开始训练", command=self.toggle)
        self.btn.grid(row=0, column=6, padx=14)
        self.btn_stop = ttk.Button(top, text="停止", command=self.request_stop, state="disabled")
        self.btn_stop.grid(row=0, column=7)
        self.info = ttk.Label(top, text="未训练", foreground="#64748b")
        self.info.grid(row=0, column=8, padx=14)

        mid = ttk.Frame(self)
        mid.pack(fill="x", pady=8)
        self.loss_plot = LinePlot(mid, height=190)
        self.loss_plot.pack(side="left", fill="both", expand=True)
        self.cm = Heatmap(mid, size=200)
        self.cm.pack(side="left", padx=8)
        self.metrics = ttk.Label(mid, text="", justify="left", font=("Consolas", 9))
        self.metrics.pack(side="left", padx=8, anchor="n")

        self.pbar = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.pbar.pack(fill="x")
        self.txt = tk.Text(self, height=10, font=("Consolas", 9), bg="#0b1220", fg="#cbd5e1",
                           relief="flat", wrap="word")
        self.txt.pack(fill="both", expand=True, pady=6)

        exp = ttk.Frame(self)
        exp.pack(fill="x")
        ttk.Button(exp, text="保存模型 model.json", command=self.save_model).pack(side="left")
        ttk.Button(exp, text="导出 TFLite（需 tensorflow）", command=self.export_tflite).pack(side="left", padx=8)
        ttk.Button(exp, text="导出 C 头文件（int8）", command=self.export_header).pack(side="left")
        ttk.Button(exp, text="一键：演示数据 + 训练", command=self.quick_demo).pack(side="right")

    def log(self, s, clear=False):
        if clear:
            self.txt.delete("1.0", "end")
        self.txt.insert("end", s + "\n")
        self.txt.see("end")

    def on_dataset(self, ds):
        if ds is None:
            self.info.config(text="数据集：未构建（训练时会现场构建）")
        else:
            self.info.config(text=f"数据集：{len(ds['y'])} 窗 × {ds['X'].shape[1]} s")

    def request_stop(self):
        self.stop = True
        self.log("已请求停止：当前轮结束后结束训练。")

    def toggle(self):
        if self.thread and self.thread.is_alive():
            return
        paths = self.app.tab_dataset.selected_paths()
        ds = self.app.ds
        if not paths and ds is None:
            messagebox.showwarning("没有数据",
                                   "「③ 数据集」里还没有可用会话。\n"
                                   "先采集（或点「一键：演示数据 + 训练」），再回来训练。")
            return
        self.stop = False
        # 注意：Tk 变量只能在主线程读取，先取出数值再交给工作线程
        try:
            epochs = int(float(self.epochs.get() or 300))
            lr = float(self.lr.get() or 1e-3)
        except ValueError:
            messagebox.showerror("参数错误", "学习率/轮数必须是数字。")
            return
        self.btn.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.log("开始训练…", clear=True)
        if ds is not None:
            self.log(f"使用「③ 数据集」已构建的 {len(ds['y'])} 个窗口（不再重新构建）。")
        else:
            self.log(f"使用 {len(paths)} 个会话文件（未构建数据集，训练时现场构建）。")
        self.thread = threading.Thread(target=self._run, args=(paths, ds, epochs, lr), daemon=True)
        self.thread.start()

    def _run(self, paths, ds, epochs, lr):
        try:
            def prog(ep, tot, trl, vl, acc, f1, auc, note):
                self.app.train_msgs.put(("progress", (ep, tot, trl, vl, acc, f1, auc, note)))
            if ds is not None:
                res = train_from_dataset(ds, epochs=epochs, lr=lr,
                                         stop_flag=lambda: self.stop, progress=prog)
            else:
                res = run_training(paths, epochs=epochs, lr=lr,
                                   stop_flag=lambda: self.stop, progress=prog)
            self.app.train_msgs.put(("done", res))
        except Exception as e:
            self.app.train_msgs.put(("error", f"{type(e).__name__}: {e}"))

    def on_msg(self, kind, payload):
        if kind == "progress":
            ep, tot, trl, vl, acc, f1, auc, note = payload
            self.pbar["value"] = 100.0 * ep / max(1, tot)
            self.info.config(text=f"第 {ep}/{tot} 轮  train {trl:.4f}  val {vl:.4f}  "
                                  f"acc {acc:.3f}  auc {auc:.3f}")
            if note:
                self.log(f"[早停] 按验证 AUC 连续 60 轮未改善，第 {ep} 轮结束。")
        elif kind == "done":
            self.finish(payload)
        elif kind == "error":
            self.btn.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.log("训练失败：" + payload)
            messagebox.showerror("训练失败", payload)

    def finish(self, res):
        self.btn.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.pbar["value"] = 100
        self.result = res
        self.app.model, self.app.mu, self.app.sd = res["net"], res["mu"], res["sd"]
        self.app.meta = res["meta"]
        self.hist = res["hist"]
        m, b = res["metrics"], res["baseline"]
        mr, mv = res.get("metrics_rest", m), res["metrics_val_best"]
        th = res["threshold"]
        self.app.tab_live.refresh_model_info()
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and v == v else "—"  # noqa: E731
        self.metrics.config(text=(
            f"验证集（留出会话）\n"
            f"accuracy  {fmt(m['acc'])}\nF1        {fmt(m['f1'])}\n"
            f"AUC       {fmt(m['auc'])}\nprecision {fmt(m['precision'])}\n"
            f"recall    {fmt(m['recall'])}\n\n"
            f"阈值 0.50（固定模型，无需校准）\n"
            f"  → 上面这一组数字\n"
            f"阈值 {th['rest']:.2f}（本人静息 90 分位）：acc {fmt(mr['acc'])}\n"
            f"阈值 {th['val_best']:.2f}（验证集最优，乐观）：acc {fmt(mv['acc'])}\n\n"
            f"基线（阈值法 thr75）\naccuracy  {fmt(b['acc'])}\nF1        {fmt(b['f1'])}\n\n"
            f"参数量    {res['meta']['n_params']}\n"
            f"最佳轮    {res['meta']['best_epoch']}\n"
            f"训练/验证 {res['meta']['n_train']}/{res['meta']['n_val']}\n"
            f"验证会话  {', '.join(res['meta']['val_groups'])}"))
        self.cm.render(cm_matrix(m))
        self.log("训练完成。")
        self.log(f"  best epoch = {res['meta']['best_epoch']}（按 {res['meta']['select_criterion']} 选点），"
                 f"实际训练 {res['meta']['epochs_run']} 轮")
        self.log(f"  [主指标] 阈值 0.50（固定模型，无需任何校准）："
                 f"accuracy={m['acc']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
        self.log(f"  [参考] 阈值 {th['rest']:.2f}（本人静息 90 分位）：accuracy={mr['acc']:.4f}；"
                 f"阈值 {th['val_best']:.2f}（验证集最优，乐观）：accuracy={mv['acc']:.4f}")
        self.log(f"  [对照] brief Option A（3s 滑动平均 + thr 75，同一验证集）："
                 f"accuracy={b['acc']:.4f}  F1={b['f1']:.4f}")
        self.log(f"  混淆矩阵：[[tn={m['confusion']['tn']}, fp={m['confusion']['fp']}], "
                 f"[fn={m['confusion']['fn']}, tp={m['confusion']['tp']}]]")
        if th["rest"] < 0.05 or th["rest"] > 0.95:
            self.log("  [注意] 静息分位阈值贴边：该会话静息/心算的打分几乎不重叠，"
                     "换人部署时仍建议先做一次静息校准。")
        if res["meta"]["n_groups"] <= 2:
            self.log(f"  [注意] 只有 {res['meta']['n_groups']} 个会话，"
                     f"实际留出比例 {res['meta']['val_ratio_actual']:.0%}"
                     f"（请求 {res['meta']['val_ratio_requested']:.0%}）；"
                     f"单次结果的 95% 置信区间约 ±6–7 个百分点。")
        self.app.say(f"训练完成：acc={m['acc']:.3f}  F1={m['f1']:.3f}")

    def save_model(self):
        if not self.result:
            messagebox.showinfo("提示", "请先训练。")
            return
        p = save_model(MODEL_PATH, self.result["net"], self.result["mu"], self.result["sd"], self.result["meta"])
        self.log(f"已保存模型 {p}")
        self.app.say("模型已保存到 models/model.json")

    def export_tflite(self):
        if not self.result:
            messagebox.showinfo("提示", "请先训练。")
            return
        try:
            p, n = export_tflite(self.result["net"], self.result["mu"], self.result["sd"],
                                 os.path.join(MODEL_DIR, "model_int8.tflite"))
            self.log(f"已导出 {p}（{n/1024:.1f} KB，int8 全量化）")
        except ImportError:
            messagebox.showerror("缺少 TensorFlow",
                                 "本机未安装 tensorflow，无法直接转 TFLite。\n\n"
                                 "可改用「导出 C 头文件」，或在装有 tensorflow 的机器上运行：\n"
                                 "  python main.py --export-tflite")
        except Exception as e:
            messagebox.showerror("导出失败", f"{type(e).__name__}: {e}")

    def export_header(self):
        if not self.result:
            messagebox.showinfo("提示", "请先训练。")
            return
        p = export_edge_header(self.result["net"], self.result["mu"], self.result["sd"],
                               os.path.join(MODEL_DIR, "neurofocus_model.h"))
        self.log(f"已导出 {p}（int8 权重 + 标准化参数，可直接放进 MCU 工程）")

    def quick_demo(self):
        self.app.tab_dataset.make_demo()
        self.app.update_idletasks()
        self.epochs.set("120")
        self.toggle()

    def tick(self):
        if self.hist:
            h = self.hist
            self.loss_plot.render([("train", C_MAIN, h["train_loss"]), ("val", C_WARN, h["val_loss"])],
                                  "损失曲线（BCE）", ylabel=f"{len(h['epoch'])} epochs")


# =============================================================================
# 15. 页面：实时推演
# =============================================================================
class LiveTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.ema = None

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="加载 models/model.json", command=self.load).pack(side="left")
        self.btn = ttk.Button(top, text="开始实时推演", command=self.toggle, state="disabled")
        self.btn.pack(side="left", padx=8)
        ttk.Label(top, text="判定阈值").pack(side="left", padx=(16, 2))
        self.thr = tk.DoubleVar(value=0.5)
        ttk.Scale(top, from_=0.2, to=0.8, variable=self.thr, length=180).pack(side="left")
        self.lbl_thr = ttk.Label(top, text="0.50")
        self.lbl_thr.pack(side="left", padx=4)
        self.lbl_model = ttk.Label(top, text="未加载模型", foreground="#b91c1c")
        self.lbl_model.pack(side="right")

        cal = ttk.Frame(self)
        cal.pack(fill="x", pady=(6, 0))
        self.btn_cal = ttk.Button(cal, text=f"静息校准（{CALIB_LIVE_SECONDS} 秒）",
                                  command=self.start_calib, state="disabled")
        self.btn_cal.pack(side="left")
        self.lbl_cal = ttk.Label(cal, text="未校准：使用训练集平均基线（跨会话准确率会偏低）",
                                 foreground="#b45309")
        self.lbl_cal.pack(side="left", padx=10)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=10)
        self.gauge = BarGauge(body, height=120, label="P(专注) — 8 秒滑窗实时输出", color=C_MAIN)
        self.gauge.pack(fill="x")
        self.verdict = ttk.Label(body, text="—", font=("Microsoft YaHei UI", 26, "bold"),
                                 foreground="#94a3b8")
        self.verdict.pack(pady=6)
        self.note = ttk.Label(body, text=("模型只看 12 维特征的最近 8 秒；标签来自采集协议，不来自设备 Attention 值。"
                                          "默认按固定模型 + 0.5 阈值运行；换人或重新佩戴后建议先做一次「静息校准」。"),
                              foreground="#64748b")
        self.note.pack()
        self.plot = LinePlot(self, height=210)
        self.plot.pack(fill="both", expand=True, pady=6)

    def refresh_model_info(self):
        if self.app.model is None:
            self.lbl_model.config(text="未加载模型", foreground="#b91c1c")
            self.btn.config(state="disabled")
            self.btn_cal.config(state="disabled")
            return
        meta = self.app.meta or {}
        self.lbl_model.config(text=f"已加载 · {meta.get('n_params', '?')} 参数 · "
                                   f"val acc {meta.get('metrics', {}).get('acc', float('nan')):.3f}",
                              foreground="#15803d")
        self.btn.config(state="normal")
        self.btn_cal.config(state="normal")

    def start_calib(self):
        if self.app.model is None:
            messagebox.showinfo("提示", "请先训练或加载模型。")
            return
        if self.app.source is None:
            messagebox.showwarning("未连接数据源", "请先在「① 设备与信号」连接串口或模拟器。")
            return
        self.lbl_cal.config(text=f"校准中… 请保持静息、闭眼、不说话（{CALIB_LIVE_SECONDS} 秒）",
                            foreground="#b45309")
        self.app.start_calibration()

    def on_calibrated(self, n_rows):
        self.thr.set(self.app.thr_cal)
        self.lbl_cal.config(text=f"已用本人静息 {n_rows} 秒校准 · 阈值自动设为 {self.app.thr_cal:.2f}",
                            foreground="#15803d")

    def load(self):
        try:
            net, mu, sd, meta = load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("加载失败", f"{e}\n\n请先在「④ 训练与评估」训练并保存模型。")
            return
        self.app.model, self.app.mu, self.app.sd, self.app.meta = net, mu, sd, meta
        self.refresh_model_info()
        self.app.say("模型已加载，可开始实时推演。")

    def toggle(self):
        self.app.infer_on = not self.app.infer_on
        self.app.infer_buf.clear()
        self.app.infer_probs.clear()
        self.ema = None
        self.btn.config(text="停止推演" if self.app.infer_on else "开始实时推演")
        if self.app.infer_on and self.app.source is None:
            messagebox.showwarning("未连接数据源", "请先在「① 设备与信号」连接串口或模拟器。")
            self.app.infer_on = False
            self.btn.config(text="开始实时推演")

    def tick(self):
        self.lbl_thr.config(text=f"{self.thr.get():.2f}")
        if not self.app.infer_probs:
            return
        p = self.app.infer_probs[-1]
        self.ema = p if self.ema is None else 0.7 * self.ema + 0.3 * p
        self.gauge.render(100 * self.ema)
        thr = self.thr.get()
        if self.ema >= thr:
            self.verdict.config(text="专注 FOCUS", foreground="#b91c1c")
        else:
            self.verdict.config(text="静息 REST", foreground="#15803d")
        self.plot.render([("P(专注)", C_MAIN, list(self.app.infer_probs))],
                         "实时概率（每个点 = 一个 8 秒窗）", ymin=0, ymax=1)


# =============================================================================
# 16. 页面：说明与 AI 使用声明
# =============================================================================
class AboutTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        txt = tk.Text(self, font=("Microsoft YaHei UI", 10), bg="#ffffff", fg="#1e293b",
                      relief="flat", wrap="word", padx=16, pady=12)
        txt.pack(fill="both", expand=True)
        txt.insert("end", ABOUT_TEXT)
        txt.config(state="disabled")


ABOUT_TEXT = """NeuroFocus · 单通道 EEG 专注度解码系统  (BMEG3103 Final Project)

【设备与采集】
  · TGAM 单通道脑电模块，电极贴左前额 FP1（眉毛上方），参考/接地夹左耳垂根部。
  · 蓝牙接收器以 USB 转串口方式接入电脑（默认 COM7，57600 波特，8-N-1）。
  · TGAM 每秒输出一个 0x20 大包：signal 质量、8 个频段能量、Attention、Meditation；
    同时以 512 Hz 输出 0x04 小包的 16 位原始采样，用于信号质量检查与波形显示。
  · 标签只由采集协议产生：闭眼静息 = 0，连续心算（如 1000 连续减 7）= 1。
    设备的 Attention 数值只作为特征输入，绝不作为标签——否则等于用"答案"训练模型，
    会造成标签泄漏，也会在报告中构成方法学错误。

【任务与模型】
  · 输入：最近 8 秒（与 Project Brief 的 "last 8 attention values" 对齐）的 12 维特征序列：
          Attention、8 个频段能量（log1p）、Attention 差分、raw_std、raw_max。
          signal 与 meditation 刻意不进模型——前者是接触质量标记（恒 0 时标准化会把 z 值放大
          到 1e5），后者是厂商派生量且跨会话漂移（单特征 AUC 仅 0.42）。
  · 结构：Conv1D(16,k=3) → ReLU → Conv1D(20,k=3) → ReLU → Conv1D(24,k=3) → ReLU
          → GlobalAveragePooling → Dense(1) → sigmoid，共 3061 个参数（与 brief 的 3k 量级一致），
          适配 TFLite Micro。
  · 训练：BCEWithLogits + Adam，按"会话文件"分组划分训练/验证（防泄漏），
          early stopping 以**验证 AUC** 为准（patience=60）——用 val_loss 会选到第 1 轮。
  · 标准化：默认只用**训练会话**的均值/标准差，训练完固化进模型，部署时直接套用。
          本机实测：逐会话静息校准并不比它更好（0.863 vs 0.880），所以不做成必需步骤。
  · 部署校准（可选）：换人/重新佩戴后，可在「实时推演」页做 30 秒静息校准，用本人静息
          重新估 mu/sd 与判定阈值（本人静息打分的 90 分位，无标签）。
  · 评估：accuracy / precision / recall / F1 / AUC / 混淆矩阵，并与 brief 的
          阈值法基线（3 秒滑动平均 + 阈值 75）在**同一验证集**上对比。

【代码结构（单文件）】
  §1–§3 依赖自举、常量、TGAM 解析器与数据源（串口/回放/模拟器）
  §4–§5 特征工程、数据集构建、1-D CNN 的纯 NumPy 前向/反向与训练
  §6–§8 演示数据、训练编排、数据集/模型/边缘部署导出
  §9–§16 tkinter 图形控件与六个页面
  main() 同时提供一个 CLI（--selftest / --make-demo / --train / --export-tflite）

【AI 使用声明（必读）】
  本项目在 AI 辅助下完成。按课程要求，此处如实标注：
  1) 代码：main.py 的框架、TGAM 协议解析、NumPy 版 1-D CNN 的前向/反向传播、
     数据集清洗规则与 tkinter 界面，均在 AI 助手（DeepSeek）逐轮协助下生成与调试；
     每一段代码都经过本人在本机运行验证（--selftest、模拟数据全流程、真实设备 COM7 抓包）。
  2) 文本：README、报告与一页说明的初稿由 AI 起草，本人核对协议细节、数值与引用后定稿。
  3) 数据：项目中的 *_demo_sim.csv 为 AI 生成的合成数据，仅用于演示流程；
     任何写进报告的指标必须来自本人真实采集的数据，并保留原始 CSV 以便复核。
  4) 责任：AI 不承担任何学术或工程责任；报告中的所有结论、图表与误差分析由本人确认。
  使用边界：AI 用于"实现与解释"，不用于伪造实验结果。

【参考文件】
  · BMEG3103_Project_brief.pdf
  · TGAM脑波开发套件使用说明v2.docx
  · TGAM數據格式說明.pdf
  · arduino读取脑波数据的方法/mindwave_uart/mindwave_uart.ino（协议实现的对照参考）
"""


# =============================================================================
# 17. 命令行入口（与 GUI 共用同一套函数）
# =============================================================================
def cli_selftest():
    """不依赖硬件的自检：协议解析 + 特征 + 前向 + 反向数值梯度。"""
    print("[1] TGAM 协议解析")
    payload = bytes([0x02, 0x00, 0x83, 0x18]) + bytes(24) + bytes([0x04, 0x37, 0x05, 0x2A])
    frame = bytes([0xAA, 0xAA, len(payload)]) + payload + bytes([(~sum(payload)) & 0xFF])
    p = TGAMParser()
    out = p.feed(frame)
    assert p.n_big == 1 and p.bad == 0, (p.n_big, p.bad)
    print(f"    大包解析 OK：attention={out[0].attention} meditation={out[0].meditation}")
    raw = bytes([0xAA, 0xAA, 0x04, 0x80, 0x02, 0x10, 0x00])
    raw += bytes([(~sum(raw[3:6])) & 0xFF])
    p.feed(raw)
    assert p.n_raw == 1, p.n_raw
    print("    原始采样小包解析 OK")

    print("[2] 数据集与模型")
    tmp = os.path.join(DATA_DIR, "_selftest")
    os.makedirs(tmp, exist_ok=True)
    paths = make_demo_set(tmp, subjects=("T1", "T2", "T3", "T4"), blocks=6,
                            block_sec=60, mix=0.18)
    ds = build_dataset(paths, settle=5)
    print(f"    窗口 {len(ds['y'])}，特征 {ds['X'].shape}")
    assert ds["X"].shape[-1] == len(FEATURE_NAMES) == 12, ds["X"].shape
    assert len(ds["calib"]) == len(set(ds["groups"].tolist())), ds["calib"].keys()
    assert all(c["from"] == "rest" for c in ds["calib"].values()), ds["calib"]
    Xz = standardize_by_group(ds)
    assert np.isfinite(Xz).all(), "按会话标准化后出现 inf/nan"
    print(f"    本人静息基线：{len(ds['calib'])} 个会话，各 "
          f"{sorted({c['n'] for c in ds['calib'].values()})} 秒；"
          f"标准化后 |z| 最大 {float(np.abs(Xz).max()):.1f}")
    tr, va = split_dataset(ds)
    Xtr, Xva, mu, sd = standardize(ds["X"][tr], ds["X"][va])
    net = FocusCNN(ds["X"].shape[-1])
    print(f"    参数量 {net.n_params()}")
    p_hat = net.predict_proba(Xva[:10])
    assert p_hat.shape == (10,)

    print(f"[3] 反向传播数值梯度检查（全部 {net.n_params()} 个参数）")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((4, WINDOW, ds["X"].shape[-1])).astype(np.float32)
    yb = np.array([0, 1, 0, 1], dtype=np.float32)
    z = net.forward(xb)
    pp = 1.0 / (1.0 + np.exp(-z))
    net.backward(((pp - yb) / len(yb)).astype(np.float32))
    keys = list(net.state().keys())
    ana = np.concatenate([g.astype(np.float64).reshape(-1) for g in net.grads()])
    st = {k: v.copy() for k, v in net.state().items()}
    eps = 1e-3
    num = []
    for key in keys:
        flat = st[key].reshape(-1)
        gn = np.zeros(flat.size, dtype=np.float64)
        for i in range(flat.size):
            o = float(flat[i])
            flat[i] = o + eps
            net.load_state(st)
            lp = bce_with_logits(net.forward(xb, cache=False), yb)
            flat[i] = o - eps
            net.load_state(st)
            lm = bce_with_logits(net.forward(xb, cache=False), yb)
            flat[i] = o
            gn[i] = (lp - lm) / (2 * eps)
        num.append(gn)
    net.load_state(st)
    num = np.concatenate(num)
    cos = float(ana @ num / (np.linalg.norm(ana) * np.linalg.norm(num) + 1e-12))
    rel = np.abs(ana - num) / (np.abs(ana) + np.abs(num) + 1e-9)
    p90 = float(np.percentile(rel, 90))
    print(f"    余弦相似度 {cos:.6f}   相对误差中位数 {np.median(rel):.2e}   P90 {p90:.2e}")
    print(f"    （极少数元素落在 ReLU 折点上，其单点误差无意义，故以余弦相似度为准）")
    assert cos > 0.999 and np.median(rel) < 5e-3, (cos, float(np.median(rel)))

    print("[4] 端到端小训练")
    res = run_training(paths, epochs=120, lr=2e-3, val_ratio=0.25, seed=1)
    print(f"    acc={res['metrics']['acc']:.3f}  F1={res['metrics']['f1']:.3f}"
          f"（阈值 0.5，主指标）"
          f"   参考：静息 90 分位 thr {res['threshold']['rest']:.2f}"
          f" → acc={res['metrics_rest']['acc']:.3f}")
    print(f"    同验证集基线 acc={res['baseline']['acc']:.3f}（合成数据，仅验证链路）")
    assert res["threshold"]["rest"] == res["threshold"]["rest"], "阈值校准返回 nan"
    shutil.rmtree(tmp, ignore_errors=True)
    print("自检全部通过。")
    return 0


def cli_train(paths, epochs=300, lr=1e-3):
    res = run_training(paths, epochs=epochs, lr=lr,
                       progress=lambda ep, tot, a, b, c, d, e, n:
                       print(f"  epoch {ep:3d}/{tot} train {a:.4f} val {b:.4f} "
                             f"acc {c:.3f} auc {e:.3f}"))
    m = res["metrics"]
    mr, mv = res.get("metrics_rest", m), res["metrics_val_best"]
    th = res["threshold"]
    print(f"\n[主指标] 阈值 0.50（固定模型，无需校准）")
    print(f"accuracy={m['acc']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    print("混淆矩阵 [[tn, fp], [fn, tp]] =", cm_matrix(m))
    print(f"[参考] 阈值 {th['rest']:.2f}（本人静息 90 分位）: acc={mr['acc']:.4f}；"
          f"阈值 {th['val_best']:.2f}（验证集最优，乐观）: acc={mv['acc']:.4f}")
    print(f"[对照] Option A thr75（同一验证集）: acc={res['baseline']['acc']:.4f}")
    p = save_model(MODEL_PATH, res["net"], res["mu"], res["sd"], res["meta"])
    print("模型已保存:", p)
    export_dataset_csv(res["ds"], DATASET_PATH)
    print("数据集已导出:", DATASET_PATH)
    dataset_report_html(res["ds"], res["meta"], os.path.join(DATA_DIR, "dataset_report.html"))
    print("数据集报告已导出:", os.path.join(DATA_DIR, "dataset_report.html"))
    hdr = os.path.join(MODEL_DIR, "model_int8.h")
    export_edge_header(res["net"], res["mu"], res["sd"], hdr)
    print("int8 C 头文件已导出:", hdr)
    return 0



def cli_record(port=DEFAULT_PORT, baud=DEFAULT_BAUD, subject="S01",
               blocks=6, block_sec=60, out_dir=None):
    """命令行采集：按“静息 / 心算”交替提示录音，不需要打开 GUI。

    用法：python main.py --record --port COM7 --subject S01 --blocks 6 --block-sec 60
    采集前确认：电极贴 FP1（左前额眉毛上方）、耳夹夹左耳垂根部、Signal 显示 0。
    """
    out_dir = out_dir or RAW_DIR
    os.makedirs(out_dir, exist_ok=True)
    q = queue.Queue()
    src = SerialSource(q, port, baud)
    src.start()
    t0 = time.time()
    while not src.ok and time.time() - t0 < 6:
        time.sleep(0.2)
    if not src.ok:
        print(f"[错误] 无法打开 {port}：{src.message}")
        print("       请确认接收器已插入、设备已开机、端口号正确（设备管理器→端口）。")
        return 3
    print(f"[连接] {port} @ {baud}，等待首个数据包…")
    while q.empty() and time.time() - t0 < 12:
        time.sleep(0.2)
    if q.empty():
        print("[警告] 12 秒内没有收到数据：请检查设备是否开机、已配对、电极已戴好。")

    path = os.path.join(out_dir, f"{subject}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    rec = Recorder(path)
    rec.open()
    tasks_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_tasks.csv")
    tasks_f = open(tasks_path, "w", encoding="utf-8", newline="")
    tasks_f.write("t,qid,question,answer,result\n")
    tasks_f.flush()
    print(f"[保存] {path}")
    print()
    print("提示：安静坐好，先做 1 分钟静息适应；听到提示音后再切换任务。")
    print("      两种状态都闭眼；心算段默算，不出声、不睁眼、不动手。")

    plan = [(i % 2, block_sec) for i in range(blocks)]
    for i, (lab, secs) in enumerate(plan):
        src.label = lab
        task = ("心算：闭眼默算，从 1000 开始连续减 7（不出声、不动手）" if lab
                else "静息：闭眼放松，不要刻意想事情")
        print()
        print(f"===== 第 {i + 1}/{len(plan)} 段 · {'专注 1' if lab else '静息 0'} =====")
        print(task)
        if lab:
            qid = i // 2 + 1
            q = f"从1000连续减7（第{qid}段心算）"
            tasks_f.write(f"{round(time.time() - t0, 2)},{qid},{q},,-1\n")
            tasks_f.flush()
        try:
            import winsound
            winsound.Beep(880 if lab else 440, 250)
        except Exception:
            pass
        for k in range(int(secs)):
            time.sleep(1.0)
            while not q.empty():
                rec.add(q.get())
            print(f"   剩余 {int(secs) - k - 1:3d}s   已记录 {rec.n} 行")
    time.sleep(1.2)
    while not q.empty():
        rec.add(q.get())
    n = rec.n
    rows = list(rec.rows)
    rec.close()
    tasks_f.close()
    src.stop()
    print()
    print(f"[完成] 共 {n} 行 → {path}")
    print(f"       心算段记录 → {tasks_path}（每段一行，供事后核对心算是否执行）")
    n0 = sum(1 for r in rows if r.label == 0)
    print(f"       静息 {n0} 行 / 心算 {n - n0} 行")
    lines, verdict = quality_lines(rows)
    print("       [体检] " + {"ok": "通过", "warn": "告警", "unknown": "信息不足"}[verdict])
    for ln in lines:
        print("       " + ln)
    print("       下一步：在 GUI「③ 数据集」勾选文件并构建，或运行 python main.py --train")
    return 0


def main(argv=None):
    for d in (DATA_DIR, RAW_DIR, MODEL_DIR, DOC_DIR):
        os.makedirs(d, exist_ok=True)
    ap = argparse.ArgumentParser(description="NeuroFocus：单通道 EEG 专注度解码（BMEG3103 Final Project）")
    ap.add_argument("--selftest", action="store_true", help="无硬件自检（协议/梯度/端到端）")
    ap.add_argument("--make-demo", action="store_true", help="生成模拟会话 CSV")
    ap.add_argument("--train", nargs="*", metavar="CSV", help="训练并保存模型")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--export-tflite", action="store_true", help="把已保存模型转 int8 TFLite")
    ap.add_argument("--record", action="store_true", help="命令行按协议采集真实数据（静息/心算交替）")
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--subject", default="S01")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--block-sec", type=int, default=60)
    args = ap.parse_args(argv)

    if args.selftest:
        return cli_selftest()
    if args.record:
        return cli_record(args.port, args.baud, args.subject, args.blocks, args.block_sec)
    if args.make_demo:
        print("\n".join(make_demo_set(RAW_DIR)))
        return 0
    if args.train is not None:
        paths = args.train or sorted(p for p in glob.glob(os.path.join(RAW_DIR, "*.csv"))
                                     if is_raw_csv(p))
        if not paths:
            print("没有可用 CSV，请先 --make-demo 或采集真实数据。")
            return 2
        return cli_train(paths, epochs=args.epochs, lr=args.lr)
    if args.export_tflite:
        net, mu, sd, meta = load_model(MODEL_PATH)
        p, n = export_tflite(net, mu, sd, os.path.join(MODEL_DIR, "model_int8.tflite"))
        print(f"{p} ({n} bytes)")
        return 0

    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.detach_source(), root.destroy()))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
