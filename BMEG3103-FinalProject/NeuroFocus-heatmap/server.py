#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NeuroFocus 注意力热力图 · 本地推理服务

独立程序：**不修改** NeuroFocus-main，只用 importlib 加载它的 main.py，
复用 TGAMParser / SerialSource / SimulatorSource / ReplaySource /
frame_features / zscore / FocusCNN / rest_threshold 等现成实现。

对外接口（只监听 127.0.0.1）：

    GET  /health          -> 服务与模型状态
    GET  /model           -> 模型元数据摘要
    GET  /ports           -> 本机串口列表（给插件选 COM 口用）
    GET  /bridge          -> Arduino 桥的状态帧摘要
    GET  /events          -> SSE 事件流（state / score / calib / log / bridge）
    POST /start           -> {"source": "sim|serial|replay", ...} 开始采集 + 推理
    POST /stop            -> 停止
    POST /calibrate       -> 开始 30 秒静息校准，完成后推送新的判定阈值
    POST /bridge          -> 给 Arduino 桥发命令（#PING / #LED:1 / #THR:40 / #BLINK）

命令行示例：

    py -3 server.py                       # 模拟数据源，直接出分（不用硬件）
    py -3 server.py --source serial --serial-port COM7
    py -3 server.py --source replay --replay "..\\NeuroFocus-main\\data\\raw\\S01_xxx.csv"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def _is_main_branch(nf_dir: str) -> bool:
    """判断这个目录是不是 main 分支那份 NeuroFocus（12 维 / 3061 参数）。

    同一台机器上经常同时存在两份：`NeuroFocus`（two-stage-14f，14 维）和
    `NeuroFocus-main`（12 维）。热力图只认 main 那份——14f 的模型 meta 里 features=0、
    也没有 `zscore`，硬跑会一直报 `no attribute 'zscore'`。
    """
    path = os.path.join(nf_dir, "main.py")
    if not os.path.isfile(path) or not os.path.isfile(os.path.join(nf_dir, "models", "model.json")):
        return False
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return "def zscore" in f.read()
    except OSError:
        return False


def _find_nf_dir() -> str:
    """按常见布局找 **main 分支**那份 NeuroFocus：环境变量 > 同级 > 上一级 > 本机已知路径。

    支持的三种摆法：
      - 交付/仓库布局：NeuroFocus-heatmap 与 NeuroFocus 并列（本仓库就是这样）
      - 早期交付布局：与 NeuroFocus-main 并列
      - 开发布局：dev/<proj>/server.py 与 <proj 上级>/NeuroFocus-main
    优先选「main 分支」特征（有 `def zscore` + models/model.json）的目录；
    一个都不符合时退回第一个装了 main.py 的目录，让报错信息更直白。
    """
    env = os.environ.get("NF_DIR")
    cands = []
    if env:
        cands.append(env)
    cands += [
        os.path.join(HERE, os.pardir, "NeuroFocus"),                # 仓库布局：BMEG3103-FinalProject/NeuroFocus
        os.path.join(HERE, os.pardir, "NeuroFocus-main"),           # 交付布局：两者并列
        os.path.join(HERE, os.pardir, os.pardir, "NeuroFocus-main"),  # 开发布局：dev/<proj>
        r"D:\Learning\BMEG3103\Final Project\NeuroFocus-main",
    ]
    cands = [os.path.normpath(os.path.abspath(c)) for c in cands]
    for c in cands:
        if _is_main_branch(c):
            return c
    for c in cands:
        if os.path.isfile(os.path.join(c, "main.py")):
            return c
    return cands[0]


DEFAULT_NF_DIR = _find_nf_dir()
HOST = "127.0.0.1"
PORT = 8765
VERBOSE = False


# ---------------------------------------------------------------------------
# 复用 NeuroFocus-main
# ---------------------------------------------------------------------------
def load_neurofocus(nf_dir: str):
    """把 NeuroFocus-main/main.py 当作库加载（不改它、也不被它当成脚本运行）。"""
    path = os.path.join(os.path.abspath(nf_dir), "main.py")
    if not os.path.isfile(path):
        raise SystemExit(f"[x] 找不到 NeuroFocus 的 main.py: {path}\n"
                         f"    用 --nf-dir 指定 NeuroFocus-main 的路径")
    spec = importlib.util.spec_from_file_location("neurofocus_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["neurofocus_main"] = mod

    # main.py 在被 import 时就会 os.makedirs 出 data/ 等目录。如果 NeuroFocus-main
    # 放在只读位置（Program Files、沙箱里、或别人给的只读拷贝），这里会直接
    # PermissionError 把服务拽崩。热力图只在内存里出分，压根不需要那些目录，
    # 所以加载期间让 makedirs 静默失败，加载完立刻还原。
    real_makedirs = os.makedirs

    def _soft_makedirs(name, mode=0o777, exist_ok=False):
        try:
            return real_makedirs(name, mode=mode, exist_ok=exist_ok)
        except OSError:
            return None

    os.makedirs = _soft_makedirs
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real_makedirs
    if getattr(mod, "np", None) is None:
        raise SystemExit("[x] NeuroFocus 没能加载到 numpy，先装：py -3 -m pip install numpy")
    return mod


def list_serial_ports() -> list:
    """列出本机串口（USB-TTL / 蓝牙 SPP 接收器都会出现在这里）。

    只用 pyserial 的 list_ports，不打开端口，所以可以随时刷新；
    没有 pyserial 时返回空表，前端会提示先装依赖。
    """
    out = []
    try:
        from serial.tools import list_ports  # type: ignore
    except Exception:  # noqa: BLE001
        return out
    try:
        for p in list_ports.comports():
            out.append({"device": str(p.device),
                        "description": str(p.description or ""),
                        "hwid": str(p.hwid or "")})
    except Exception:  # noqa: BLE001
        return out
    out.sort(key=lambda d: d["device"])
    return out


# ---------------------------------------------------------------------------
# Arduino 桥（TGAM 字节流里混着的 plen=0x0C 状态帧）
# ---------------------------------------------------------------------------
BRIDGE_PLEN = 0x0C
BRIDGE_TAG_STATE = 0x01
BRIDGE_TAG_MARK = 0x02


class BridgeParser:
    """从串口字节流里捞出 Arduino 桥自己的状态帧。

    固件把两样东西混在同一条串口上：TGAM 原始帧（plen=0x20 / 0x04，原样转发）
    和本程序的状态帧（plen=0x0C，16 字节）。这里只摘 0x0C 那种；其余字节原样
    跳过，所以 main.py 的 TGAMParser 仍然能看到完整的 TGAM 字节流。
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.buf = bytearray()
        self.n_frames = 0
        self.marks = 0
        self.synthetic = False
        self.last = {}

    def feed(self, chunk: bytes) -> list:
        out = []
        with self.lock:
            self.buf += chunk
            b = self.buf
            while True:
                i = b.find(b"\xaa\xaa")
                if i < 0:
                    if len(b) > 4:
                        del b[:-2]      # 留两个字节，可能是下一帧的半个帧头
                    break
                if i:
                    del b[:i]
                if len(b) < 3:
                    break
                plen = b[2]
                if plen != BRIDGE_PLEN:
                    del b[0]            # 不是桥帧（多半是 TGAM 帧），跳过
                    continue
                if len(b) < BRIDGE_PLEN + 4:
                    break               # 还没收全
                payload = bytes(b[3:3 + BRIDGE_PLEN])
                if ((~sum(payload)) & 0xFF) != b[3 + BRIDGE_PLEN]:
                    del b[0]
                    continue
                del b[:BRIDGE_PLEN + 4]
                ev = self._decode(payload)
                if ev:
                    out.append(ev)
        return out

    def _decode(self, p: bytes) -> dict:
        flags = p[1]
        ev = {
            "kind": "bridge",
            "tag": p[0],
            "flags": flags,
            "link": bool(flags & 0x01),
            "artifact": bool(flags & 0x02),
            "btn": bool(flags & 0x04),
            "led": bool(flags & 0x08),
            "att": p[2],
            "signal": p[3],
            "raw_std": p[4] * 8.0,
            "raw_max": p[5] * 24.0,
            "med": p[6],
            "marks": p[7],
            "bad": p[8],
            "raw_n": p[9],
            "cmd": p[10],
            "big": p[11],
            "synthetic": self.synthetic,
            "t": time.time(),
        }
        self.n_frames += 1
        if ev["tag"] == BRIDGE_TAG_MARK:
            self.marks += 1
        self.last = ev
        return ev

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.last)


# ---------------------------------------------------------------------------
# 演示用模拟数据源
# ---------------------------------------------------------------------------
def z_to_sample(nf, mu, sd, z):
    """把「标准化空间」的一个点反算成 nf.Sample（原始单位）。

    12 维特征的原始量纲太野（band 可达 1e6、raw_std 只有两位数），直接凭感觉
    编原始值很容易编到训练分布之外；反过来在 z 空间造点、再按模型自己的
    mu/sd 反算，就能保证造出来的样本落在模型见过的范围内。
    """
    bands = {}
    for k, b in enumerate(nf.BANDS):
        f = max(0.0, float(mu[1 + k] + z[1 + k] * sd[1 + k]))
        bands[b] = int(min(1e9, math.expm1(20.0 * f)))
    att = int(round(100.0 * max(0.03, min(0.99, float(mu[0] + z[0] * sd[0])))))
    s = nf.Sample(t=time.time(), signal=0, attention=att, meditation=50, bands=bands)
    s.raw_n = 512
    s.raw_std = 2000.0 * max(0.002, min(3.0, float(mu[10] + z[10] * sd[10])))
    s.raw_max = 6000.0 * max(0.002, min(3.0, float(mu[11] + z[11] * sd[11])))
    return s


DEMO_ATT_SHIFT = 1.2     # attention 在专注/静息两端的 z 偏移
DEMO_BAND_SHIFT = 0.30   # 八个频段整体跟着动的幅度
DEMO_RAW_SHIFT = 0.20    # raw_std / raw_max 的同步变化
DEMO_NOISE = 0.70        # 每秒的随机扰动（z 单位）
DEMO_SMOOTH = 0.86       # 随机游走的惯性


def _focus_curve(t: float) -> float:
    """平滑的专注度曲线 0~1：两个不同周期叠加，避免机械的正弦。"""
    a = 0.5 + 0.5 * math.sin(2 * math.pi * t / 95.0)
    b = 0.5 + 0.5 * math.sin(2 * math.pi * t / 31.0 + 1.7)
    return 0.62 * a + 0.38 * b


def make_demo_source(nf, mu, sd, out_queue):
    """演示用模拟源：没有头环也能把热力图跑起来（数据是编的，别当测量结果）。

    为什么不直接用 NeuroFocus 的 SimulatorSource：它给的 raw_std≈700~900、
    raw_max≈1900~2600，而这份 models/model.json 的训练量纲是 raw_std≈76、
    raw_max≈299；标准化后这两维直接顶到 ±8 截断，模型恒定输出 1.000，
    热力图会一片全绿、看不出任何起伏。

    这里改成「z 空间随机游走 + 平滑的专注度曲线」，并偶尔插入一段走神，
    让概率真的在 0.1~0.9 之间滑动，方便演示绿/橙两种颜色和点击重测。

    继承 nf.BaseSource，因此 start/stop/ok/message/_emit 与串口源完全一致。
    """
    import random

    class _DemoSource(nf.BaseSource):
        def __init__(self):
            super().__init__(out_queue)
            self._rand = random.Random()

        def run(self):
            np = nf.np
            rng = np.random.default_rng()
            z = rng.normal(0, 1, 12)
            t0 = time.time()
            next_dip = t0 + self._rand.uniform(15, 40)
            dip_until = 0.0
            self.ok = True
            self.message = "演示用模拟数据（非真实测量）"

            while not self.stopped():
                time.sleep(1.0)
                now = time.time()
                u = _focus_curve(now - t0)

                if now >= next_dip and now >= dip_until:
                    dip_until = now + self._rand.uniform(6, 14)
                    next_dip = dip_until + self._rand.uniform(20, 45)
                if now < dip_until:
                    u *= 0.30

                v = 2.0 * u - 1.0
                drift = np.zeros(12)
                drift[0] = DEMO_ATT_SHIFT * v
                drift[1:9] = DEMO_BAND_SHIFT * v
                drift[10] = DEMO_RAW_SHIFT * v
                drift[11] = DEMO_RAW_SHIFT * v
                z = DEMO_SMOOTH * z + (1.0 - DEMO_SMOOTH) * (
                    drift + rng.normal(0, DEMO_NOISE, 12)
                )

                self._emit([z_to_sample(nf, mu, sd, z)])

    return _DemoSource()


# ---------------------------------------------------------------------------
# 事件分发（SSE）
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=512)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._subs.discard(q)

    def publish(self, msg: dict):
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# 推理引擎
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, nf, model_path: str, hub: Hub):
        self.nf = nf
        self.hub = hub
        self.model_path = model_path
        self.net, self.mu, self.sd, self.meta = nf.load_model(model_path)
        self.lock = threading.RLock()
        self.source = None
        self.worker = None
        self.q = None
        self.running = False
        self.source_kind = ""
        self.serial_port = ""
        self.thr = float((self.meta.get("threshold") or {}).get("default", 0.5))
        self.win = deque(maxlen=nf.WINDOW)
        self.probs = deque(maxlen=3600)
        self.ema = None
        self.calib_left = 0
        self.calib_probs: list[float] = []
        self.started_at = 0.0
        self.n_scores = 0
        self.last_error = ""
        self.bridge = BridgeParser()
        self.bridge_last = {}
        self.bridge_marks = []
        self.bridge_pub_at = 0.0

    # -- 状态 --------------------------------------------------------------
    def model_summary(self) -> dict:
        m = self.meta or {}
        met = m.get("metrics") or {}
        return {
            "path": self.model_path,
            "params": int(m.get("n_params") or 0),
            "features": len(m.get("features") or []),
            "window": int(m.get("window") or getattr(self.nf, "WINDOW", 8)),
            "trained_at": m.get("trained_at"),
            "acc": met.get("acc"),
            "f1": met.get("f1"),
            "auc": met.get("auc"),
        }

    def status(self) -> dict:
        return {
            "running": self.running,
            "thr": round(self.thr, 3),
            "source": self.source_kind,
            "port": self.serial_port,
            "scores": self.n_scores,
            "calibrating": self.calib_left > 0,
            "error": self.last_error,
            "bridge": {
                "frames": self.bridge.n_frames,
                "marks": self.bridge.marks,
                "synthetic": self.bridge.synthetic,
                "last": self.bridge_last,
            },
            "model": self.model_summary(),
        }

    def publish_state(self, state: str, text: str = ""):
        self.hub.publish({"type": "state", "state": state, "text": text,
                          "thr": round(self.thr, 3), "running": self.running,
                          "source": self.source_kind, "port": self.serial_port,
                          "model": self.model_summary()})

    # -- 启停 --------------------------------------------------------------
    def start(self, source: str = "sim", serial_port: str | None = None,
              baud: int | None = None, replay: str | None = None,
              speed: float = 1.0) -> dict:
        nf = self.nf
        with self.lock:
            if self.running:
                return {"ok": True, "already": True}
            self.q = queue.Queue()
            port_label = ""

            if source == "serial":
                if nf.serial is None:
                    return {"ok": False, "error": "没有安装 pyserial：py -3 -m pip install pyserial"}
                port = serial_port or getattr(nf, "DEFAULT_PORT", "COM7")
                baudrate = int(baud or getattr(nf, "DEFAULT_BAUD", 57600))
                src = nf.SerialSource(self.q, port, baudrate)
                src.raw_callback = self._on_raw   # 顺手把字节流分一份给 Arduino 桥解析
                port_label = port
            elif source == "replay":
                if not replay:
                    return {"ok": False, "error": "replay 模式需要给出 CSV 路径"}
                src = nf.ReplaySource(self.q, replay, speed=float(speed or 1.0), loop=True)
            elif source == "sim":
                src = make_demo_source(nf, self.mu, self.sd, self.q)
            else:
                return {"ok": False, "error": f"未知数据源: {source}"}

            self.win.clear()
            self.probs.clear()
            self.ema = None
            self.calib_left = 0
            self.calib_probs = []
            self.n_scores = 0
            self.last_error = ""
            self.bridge = BridgeParser()
            self.bridge.synthetic = (source == "sim")
            self.bridge_last = {}
            self.bridge_marks = []
            self.bridge_pub_at = 0.0
            self.thr = float((self.meta.get("threshold") or {}).get("default", 0.5))
            self.source = src
            self.source_kind = source
            self.serial_port = port_label
            self.running = True
            self.started_at = time.time()

            src.start()
            self.worker = threading.Thread(target=self._loop, name="nfh-infer", daemon=True)
            self.worker.start()

        # 数据源是后台线程：串口打不开、CSV 不存在这类错误要等线程跑起来才知道。
        # 最多等 2 秒，等不到「已连上」就报错并把状态退回待机，免得面板一直显示
        # 「测量中」却一个分数都不来。
        err = self._wait_source_ready(src, timeout=5.0 if source == "serial" else 2.0)
        if err:
            self._abort_start(src, err)
            return {"ok": False, "error": err}

        self.publish_state("running", f"数据源已启动（{source}）")
        return {"ok": True, "source": source}

    def _wait_source_ready(self, src, timeout: float = 2.0) -> str:
        """等数据源报告「已连上」。

        注意：不能调 src.is_alive()。NeuroFocus 的 BaseSource 把
        `self._stop = threading.Event()` 塞进了 Thread 的 `_stop()` 位置，
        于是 is_alive() 会 `TypeError: 'Event' object is not callable`。
        这里只看 ok / message。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if getattr(src, "ok", False):
                return ""
            time.sleep(0.05)
        msg = (getattr(src, "message", "") or "").strip()
        if msg and msg != "未启动":
            return msg
        return f"数据源 {timeout:.0f} 秒内没有就绪（端口被占用？设备没开机？）"

    def _abort_start(self, src, err: str):
        with self.lock:
            self.running = False
            try:
                src.stop()
            except Exception:
                pass
            try:
                if self.worker:
                    self.worker.join(timeout=1.0)
            except Exception:
                pass
            self.source = None
            self.worker = None
            self.last_error = err
            self.publish_state("idle", err)

    def stop(self) -> dict:
        with self.lock:
            if not self.running:
                return {"ok": True}
            self.running = False
            self.calib_left = 0
            self.calib_probs = []
            try:
                if self.source:
                    self.source.stop()
            except Exception:
                pass
            try:
                if self.worker:
                    self.worker.join(timeout=2.0)
            except Exception:
                pass
            self.source = None
            self.worker = None
            self.publish_state("idle", "已停止")
            return {"ok": True}

    def calibrate(self, seconds: int | None = None) -> dict:
        with self.lock:
            if not self.running:
                return {"ok": False, "error": "请先开始测量"}
            sec = int(seconds or getattr(self.nf, "CALIB_LIVE_SECONDS", 30))
            self.calib_probs = []
            self.calib_left = sec
            self.hub.publish({"type": "calib", "done": False, "seconds": sec,
                              "text": f"请保持静息、闭眼、不说话（{sec} 秒）"})
            return {"ok": True, "seconds": sec}

    def set_threshold(self, thr: float) -> dict:
        with self.lock:
            self.thr = float(min(0.95, max(0.05, thr)))
            val = self.thr
        # 顺手把阈值同步给 Arduino：片上 LED 用它做本地反馈
        sent = self.send_bridge("#THR:%d" % int(round(val * 100)))
        return {"ok": True, "thr": round(val, 3), "bridge": sent.get("ok", False)}

    # -- Arduino 桥 --------------------------------------------------------
    def _on_raw(self, chunk: bytes):
        """串口原始字节的回调（跑在 SerialSource 自己的线程里）。"""
        try:
            events = self.bridge.feed(chunk)
        except Exception:  # noqa: BLE001
            return
        for ev in events:
            ev = dict(ev)
            ev["type"] = "bridge"
            self.bridge_last = ev
            if ev["tag"] == BRIDGE_TAG_MARK:
                # 按键 MARK：毫秒级事件，原样立刻推给面板（面板负责打视频时间戳）
                self.bridge_marks.append(ev["t"])
                ev["mark"] = True
                self.hub.publish(ev)
                continue
            now = time.time()
            if now - self.bridge_pub_at >= 0.5:   # 状态帧 1 Hz，限一下流
                self.bridge_pub_at = now
                self.hub.publish(ev)

    def _sim_bridge(self, s, p: float):
        """模拟数据源也造一份桥状态，没有 Arduino 也能把界面先调通。"""
        now = time.time()
        if now - self.bridge_pub_at < 1.0:
            return
        self.bridge_pub_at = now
        artifact = float(getattr(s, "raw_std", 0.0)) > 600.0
        ev = {"type": "bridge", "kind": "bridge", "tag": BRIDGE_TAG_STATE,
              "flags": 0x01 | (0x02 if artifact else 0) | 0x08,
              "link": True, "artifact": artifact, "btn": False, "led": True,
              "att": int(getattr(s, "attention", 0)),
              "signal": int(getattr(s, "signal", 0)),
              "raw_std": round(float(getattr(s, "raw_std", 0.0)), 1),
              "raw_max": round(float(getattr(s, "raw_max", 0.0)), 1),
              "med": int(getattr(s, "meditation", 0)),
              "marks": self.bridge.marks, "bad": 0,
              "raw_n": int(getattr(s, "raw_n", 0)), "cmd": 0, "big": 0,
              "synthetic": True, "t": now}
        self.bridge.n_frames += 1
        self.bridge_last = ev
        self.hub.publish(ev)

    def send_bridge(self, cmd: str) -> dict:
        """给 Arduino 桥发一条命令（#PING / #LED:1 / #THR:40 / #BLINK）。"""
        cmd = (cmd or "").strip()
        if not cmd:
            return {"ok": False, "error": "命令为空"}
        if not cmd.startswith("#"):
            cmd = "#" + cmd
        if self.source_kind != "serial" or self.source is None:
            return {"ok": False, "error": "当前不是串口数据源，命令发不出去"}
        ser = getattr(self.source, "ser", None)
        if ser is None:
            return {"ok": False, "error": "串口还没打开"}
        try:
            ser.write((cmd + "\n").encode("ascii", "ignore"))
            try:
                ser.flush()
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"写串口失败: {e}"}
        return {"ok": True, "cmd": cmd}

    # -- 主循环 ------------------------------------------------------------
    def _loop(self):
        nf = self.nf
        src = self.source
        last_src_msg = ""
        while self.running:
            # 数据源自己坏了（串口掉线、设备没电）不会让队列报错，只会让分数断流，
            # 所以这里顺手把它的 message 变化转成一条 log 事件推给面板。
            if src is not None:
                msg = (getattr(src, "message", "") or "").strip()
                if not getattr(src, "ok", True) and msg and msg != "未启动" and msg != last_src_msg:
                    last_src_msg = msg
                    self.last_error = msg
                    self.hub.publish({"type": "log", "level": "error", "text": msg})

            try:
                s = self.q.get(timeout=0.4)
            except queue.Empty:
                continue
            self.win.append(s)
            if len(self.win) < nf.WINDOW:
                continue
            try:
                X = nf.frame_features(list(self.win))          # (8, 12)
                Z = nf.zscore(X, self.mu, self.sd)             # 标准化 + 截断 ±8
                p = float(nf.sigmoid(self.net.forward(Z[None, :, :], cache=False))[0])
            except Exception as e:                              # noqa: BLE001
                self.last_error = f"推理失败: {e}"
                self.hub.publish({"type": "log", "level": "error", "text": self.last_error})
                if VERBOSE:
                    traceback.print_exc()
                continue

            self.n_scores += 1
            self.probs.append(p)
            self.ema = p if self.ema is None else 0.7 * self.ema + 0.3 * p

            sig = int(s.signal)
            ok = (sig not in nf.BAD_SIGNAL_CODES) and (0 <= sig < 100)

            if self.calib_left > 0:
                self.calib_left -= 1
                if ok:
                    self.calib_probs.append(p)
                if self.calib_left == 0:
                    self.thr = float(nf.rest_threshold(self.calib_probs))
                    self.send_bridge("#THR:%d" % int(round(self.thr * 100)))
                    self.hub.publish({"type": "calib", "done": True,
                                      "n": len(self.calib_probs),
                                      "thr": round(self.thr, 3),
                                      "text": f"静息校准完成（{len(self.calib_probs)} 窗），阈值 {self.thr:.2f}"})

            self.hub.publish({
                "type": "score",
                "t": time.time(),
                "p": round(p, 4),
                "ema": round(self.ema, 4),
                "thr": round(self.thr, 3),
                "ok": ok,
                "signal": sig,
                "attention": int(s.attention),
                "source": self.source_kind,
            })

            if self.source_kind == "sim":
                self._sim_bridge(s, p)


# ---------------------------------------------------------------------------
# HTTP / SSE
# ---------------------------------------------------------------------------
APP: "App | None" = None


class App:
    def __init__(self, engine: Engine, hub: Hub):
        self.engine = engine
        self.hub = hub

    def status(self) -> dict:
        return {"ok": True, **self.engine.status(), "time": time.time()}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NeuroFocusHeatmap/0.1"

    def log_message(self, fmt, *args):  # noqa: A003
        if VERBOSE:
            super().log_message(fmt, *args)

    # -- helpers -----------------------------------------------------------
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            return {}

    # -- routes ------------------------------------------------------------
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/health":
            return self._json(APP.status())
        if path == "/model":
            return self._json(APP.engine.model_summary())
        if path == "/ports":
            return self._json({"ok": True, "ports": list_serial_ports(),
                               "default": getattr(APP.engine.nf, "DEFAULT_PORT", "COM7")})
        if path == "/bridge":
            b = APP.engine.bridge
            return self._json({"ok": True, "frames": b.n_frames, "marks": b.marks,
                               "synthetic": b.synthetic, "last": b.snapshot()})
        if path == "/events":
            return self._events()
        return self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        body = self._body()
        eng = APP.engine
        if path == "/start":
            return self._json(eng.start(
                source=str(body.get("source") or "sim").lower(),
                serial_port=body.get("serial_port") or body.get("port"),
                baud=body.get("baud"),
                replay=body.get("replay"),
                speed=body.get("speed") or 1.0,
            ))
        if path == "/stop":
            return self._json(eng.stop())
        if path == "/calibrate":
            return self._json(eng.calibrate(body.get("seconds")))
        if path == "/threshold":
            try:
                return self._json(eng.set_threshold(float(body.get("thr"))))
            except (TypeError, ValueError):
                return self._json({"ok": False, "error": "thr 需要数字"}, 400)
        if path == "/bridge":
            # 给 Arduino 桥发命令：#PING / #LED:1 / #LED:0 / #THR:40 / #BLINK
            return self._json(eng.send_bridge(str(body.get("cmd") or "")))
        return self._json({"ok": False, "error": "not found"}, 404)

    # -- SSE ---------------------------------------------------------------
    def _events(self):
        q = APP.hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        self.close_connection = True
        try:
            hello = {"type": "state", "state": "connected" if not APP.engine.running else "running",
                     "text": "", "thr": round(APP.engine.thr, 3),
                     "running": APP.engine.running, "model": APP.engine.model_summary()}
            self.wfile.write(b"data: " + json.dumps(hello, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()
            last = time.time()
            while True:
                try:
                    msg = q.get(timeout=1.0)
                except queue.Empty:
                    if time.time() - last > 10:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last = time.time()
                    continue
                self.wfile.write(b"data: " + json.dumps(msg, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
                last = time.time()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            APP.hub.unsubscribe(q)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    global VERBOSE, APP
    ap = argparse.ArgumentParser(description="NeuroFocus 注意力热力图 · 本地推理服务")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--http-port", type=int, default=PORT, help="本服务的端口（默认 8765）")
    ap.add_argument("--nf-dir", default=DEFAULT_NF_DIR, help="NeuroFocus-main 目录")
    ap.add_argument("--model", default=None, help="模型 JSON（默认 NeuroFocus-main/models/model.json）")
    ap.add_argument("--source", default="sim", choices=["sim", "serial", "replay", "none"],
                    help="启动时自动打开的数据源（none = 等插件发指令）")
    ap.add_argument("--serial-port", default=None, help="串口，如 COM7")
    ap.add_argument("--baud", type=int, default=None)
    ap.add_argument("--replay", default=None, help="回放用的原始 CSV")
    ap.add_argument("--no-autostart", action="store_true", help="启动时不自动开始采集")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    VERBOSE = bool(args.verbose)

    nf = load_neurofocus(args.nf_dir)
    model_path = args.model or getattr(nf, "MODEL_PATH",
                                       os.path.join(args.nf_dir, "models", "model.json"))
    if not os.path.isfile(str(model_path)):
        raise SystemExit(f"[x] 找不到模型文件: {model_path}")

    hub = Hub()
    engine = Engine(nf, str(model_path), hub)
    APP = App(engine, hub)

    srv = Server((args.host, args.http_port), Handler)
    threading.Thread(target=srv.serve_forever, name="nfh-http", daemon=True).start()

    ms = engine.model_summary()
    print("=" * 64)
    print("  NeuroFocus 注意力热力图 · 本地推理服务")
    print("=" * 64)
    print(f"  模型      : {os.path.basename(str(model_path))}  "
          f"{ms['params']} 参数 · {ms['features']} 维 · {ms['window']} 秒窗")
    if ms.get("acc") is not None:
        print(f"  验证指标  : acc {ms['acc']:.3f} · F1 {(ms['f1'] or 0):.3f} · AUC {(ms['auc'] or 0):.3f}")
    print(f"  监听      : http://{args.host}:{args.http_port}")
    print(f"  事件流    : http://{args.host}:{args.http_port}/events")
    print("-" * 64)

    if args.source != "none" and not args.no_autostart:
        r = engine.start(source=args.source, serial_port=args.serial_port,
                         baud=args.baud, replay=args.replay)
        if r.get("ok"):
            print(f"  数据源    : {args.source} 已启动")
        else:
            print(f"  [!] 自动启动失败: {r.get('error')}")
    else:
        print("  数据源    : 未启动（等插件点「开始测量」）")
    print("  按 Ctrl+C 退出")
    print("=" * 64)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在停止…")
        engine.stop()
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
