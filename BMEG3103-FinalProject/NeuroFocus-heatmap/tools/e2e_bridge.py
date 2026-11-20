#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端自检：起一个临时 server.py（sim 源），验证 /bridge 与 SSE bridge 事件。

不需要 Arduino：模拟源会自己造桥状态帧（synthetic=true），用来确认
面板要消费的那条链路是通的。

用法：
    py -3 tools/e2e_bridge.py
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
PORT = int(os.environ.get("NFH_PORT", "8791"))
API = f"http://127.0.0.1:{PORT}"


def http(path, body=None, timeout=10):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "text/plain"
    req = urllib.request.Request(API + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def wait_up(deadline=15.0):
    end = time.time() + deadline
    while time.time() < end:
        try:
            return http("/health", timeout=2)
        except Exception:
            time.sleep(0.3)
    raise SystemExit("server 没起来")


def read_sse(seconds=12.0):
    events = []
    end = time.time() + seconds
    req = urllib.request.Request(API + "/events")
    with urllib.request.urlopen(req, timeout=seconds + 5) as r:
        buf = b""
        while time.time() < end:
            chunk = r.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                raw, buf = buf.split(b"\n\n", 1)
                for line in raw.split(b"\n"):
                    if line.startswith(b"data: "):
                        try:
                            events.append(json.loads(line[6:].decode("utf-8")))
                        except Exception:
                            pass
    return events


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        if not cond:
            ok = False

    cmd = [sys.executable, os.path.join(ROOT, "server.py"),
           "--source", "none", "--http-port", str(PORT)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        h = wait_up()
        check(h.get("ok") is True, "/health 正常")
        check(isinstance(h.get("bridge"), dict), "/health 里带 bridge 摘要")

        r = http("/start", {"source": "sim"})
        check(r.get("ok") is True, f"/start sim 成功（{r}）")

        evs = read_sse(16.0)
        scores = [e for e in evs if e.get("type") == "score"]
        bridges = [e for e in evs if e.get("type") == "bridge"]
        # 模型要攒满 WINDOW=8 秒才出第一分，所以 16 秒里大概 8~9 个
        check(len(scores) >= 4, f"16 秒内收到 {len(scores)} 个 score")
        check(len(bridges) >= 4, f"收到 {len(bridges)} 个 bridge 状态帧")
        if bridges:
            b = bridges[-1]
            for k in ("link", "artifact", "att", "signal", "raw_std", "marks", "synthetic"):
                check(k in b, f"bridge 帧带字段 {k}")
            check(b.get("synthetic") is True, "sim 帧标记 synthetic=true")

        g = http("/bridge")
        check(g.get("ok") is True and g.get("frames", 0) > 0,
              f"GET /bridge 有计数（frames={g.get('frames')}）")

        # sim 源不是串口：发命令应当被明确拒绝，而不是假装成功
        s = http("/bridge", {"cmd": "#PING"})
        check(s.get("ok") is False, f"sim 下 #PING 被拒：{s.get('error')}")

        http("/stop", {})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    print("\n" + ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
