#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nfh_bridge 协议自检（不需要 Arduino）

做两件事：
1. 按固件里写的格式拼出「TGAM 原始帧 + 桥状态帧」混在一起的字节流，
   确认 server.BridgeParser 能把桥帧准确摘出来（含拆包、坏校验、噪声）；
2. 把同一条字节流喂给 NeuroFocus 的 TGAMParser，确认桥帧不会干扰它
   —— 这是整个方案能不能「透明透传」的关键。

用法：
    py -3 tools/bridge_selftest.py [NeuroFocus-main 目录]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))

import server  # noqa: E402


def frame(payload: bytes) -> bytes:
    return b"\xaa\xaa" + bytes([len(payload)]) + payload + bytes([(~sum(payload)) & 0xFF])


def bridge_frame(tag=0x01, flags=0x01 | 0x08, att=72, signal=0, std=40, peak=90,
                 med=50, marks=0, bad=0, raw_n=200, cmd=0, big=3) -> bytes:
    pl = bytes([tag, flags, att, signal, std, peak, med, marks, bad, raw_n, cmd, big])
    return frame(pl)


def tgam_big(signal=0, att=60, med=40) -> bytes:
    p = bytearray(32)
    p[0] = 0x02
    p[1] = signal
    p[2] = 0x83
    p[3] = 0x18
    for k in range(8):                      # 8 个三字节大端频段，随便填
        v = 1000 + k * 7
        p[4 + 3 * k] = (v >> 16) & 0xFF
        p[5 + 3 * k] = (v >> 8) & 0xFF
        p[6 + 3 * k] = v & 0xFF
    p[28] = 0x04
    p[29] = att
    p[30] = 0x05
    p[31] = med
    return frame(bytes(p))


def tgam_raw(v: int) -> bytes:
    v &= 0xFFFF
    return frame(bytes([0x80, 0x02, (v >> 8) & 0xFF, v & 0xFF]))


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        if not cond:
            ok = False

    # --- 1. 桥帧单独解析 ----------------------------------------------------
    bp = server.BridgeParser()
    evs = bp.feed(bridge_frame(att=72))
    check(len(evs) == 1, f"单帧解析出 1 个事件（实际 {len(evs)}）")
    e = evs[0] if evs else {}
    check(e.get("att") == 72, f"attention=72（实际 {e.get('att')}）")
    check(e.get("link") is True and e.get("led") is True, "flags 解析出 link/led")
    check(abs(e.get("raw_std", 0) - 320.0) < 0.01, f"raw_std = 40*8 = 320（实际 {e.get('raw_std')}）")

    # --- 2. 拆包：一个字节一个字节喂 ---------------------------------------
    bp = server.BridgeParser()
    got = []
    for byte in bridge_frame(att=11) + bridge_frame(tag=0x02, flags=0x05, att=12):
        got += bp.feed(bytes([byte]))
    check(len(got) == 2, f"逐字节喂入解析出 2 帧（实际 {len(got)}）")
    check(any(g["tag"] == 0x02 for g in got), "MARK 帧（tag=0x02）被识别")
    check(bp.marks == 1, f"MARK 计数 = 1（实际 {bp.marks}）")

    # --- 3. 坏校验不能被当成帧 ---------------------------------------------
    bad = bytearray(bridge_frame(att=33))
    bad[-1] ^= 0xFF
    bp = server.BridgeParser()
    check(bp.feed(bytes(bad)) == [], "校验错误的帧被丢弃")
    check(bp.feed(bridge_frame(att=33)) != [], "坏帧之后流还能重新对上帧头")

    # --- 4. 与 TGAM 帧混流时互不干扰 --------------------------------------
    # 目录按顺序找：命令行 > 交付布局（../..）> 本机常见安装位置
    cands = []
    if len(sys.argv) > 1:
        cands.append(sys.argv[1])
    cands.append(server._find_nf_dir())
    nf_dir = next((c for c in cands if os.path.isfile(os.path.join(c, "main.py"))), None)
    if nf_dir is None:
        print("找不到 NeuroFocus 的 main.py，试过：\n  " + "\n  ".join(os.path.abspath(c) for c in cands))
        return 1
    nf = server.load_neurofocus(nf_dir)

    stream = bytearray()
    for _ in range(4):
        # 真实顺序：一秒内先连续来 raw 小包，末尾才是那个 0x20 大包
        for k in range(32):
            stream += tgam_raw(k * 3 - 40)
        stream += tgam_big(att=60)
        stream += bridge_frame(att=60, raw_n=32)
    stream += b"\x00\xff\xaa"                        # 噪声尾巴

    bp = server.BridgeParser()
    evs = bp.feed(bytes(stream))
    check(len(evs) == 4, f"混流里摘出 4 个桥帧（实际 {len(evs)}）")

    parser = nf.TGAMParser()
    samples = parser.feed(bytes(stream))
    check(len(samples) == 4, f"TGAMParser 仍解析出 4 个大包（实际 {len(samples)}）")
    check(parser.bad == 0, f"TGAMParser 坏包计数 = 0（实际 {parser.bad}）")
    if samples:
        s = samples[0]
        check(s.attention == 60, f"attention 透传正确（实际 {s.attention}）")
        check(s.raw_n == 32, f"raw 采样数 = 32（实际 {s.raw_n}）")
        check(s.raw_std > 0, f"raw_std 由 raw 小包算出（{s.raw_std:.1f}）")

    print("\n" + ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
