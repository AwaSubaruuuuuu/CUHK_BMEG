"""在「标准化空间」里造模拟样本，找一组参数让演示时 p 能真在 0.1~0.9 之间滑动。

现在的坑：NeuroFocus 自带 SimulatorSource 的 raw_std/raw_max 是 700~2600，
而训练数据的量纲对应 raw_std≈76、raw_max≈299，标准化后直接顶到截断 ±8，
于是模型恒定输出 1.000，热力图全是绿的，没法演示。

这里反过来做：先在 z 空间造点（z ~ N(0,1) 是训练集的样子），再反算回原始单位。
"""
import importlib.util
import math
import os
import sys

import numpy as np

# NeuroFocus 目录：命令行参数 > NF_DIR 环境变量 > 与 extension/ 并列的兄弟目录
# （仓库布局里是 ../NeuroFocus，早期交付布局里是 ../NeuroFocus-main）
# 用法：py -3 probe_msim.py [NeuroFocus 路径]
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [
    sys.argv[1] if len(sys.argv) > 1 else "",
    os.environ.get("NF_DIR", ""),
    os.path.join(_HERE, "..", "..", "NeuroFocus"),        # 仓库布局
    os.path.join(_HERE, "..", "NeuroFocus"),              # 交付布局（并列）
    os.path.join(_HERE, "..", "..", "NeuroFocus-main"),
    os.path.join(_HERE, "..", "NeuroFocus-main"),
]
NF = next(
    (os.path.abspath(c) for c in _CANDS if c and os.path.isfile(os.path.join(c, "main.py"))),
    os.path.abspath(os.path.join(_HERE, "..", "..", "NeuroFocus")),
)
if not os.path.isfile(os.path.join(NF, "main.py")):
    raise SystemExit("找不到 NeuroFocus 的 main.py，用参数或 NF_DIR 指定它的目录：\n  " + NF)
spec = importlib.util.spec_from_file_location("neurofocus_main", os.path.join(NF, "main.py"))
nf = importlib.util.module_from_spec(spec)
sys.modules["neurofocus_main"] = nf
_rm = os.makedirs
os.makedirs = lambda *a, **k: None
spec.loader.exec_module(nf)
os.makedirs = _rm
np = nf.np

net, mu, sd, meta = nf.load_model(os.path.join(NF, "models", "model.json"))
MU = np.asarray(mu, dtype=float)
SD = np.asarray(sd, dtype=float)


def sample_from_z(z):
    """z: (12,) 标准化空间 -> nf.Sample（原始单位）。"""
    bands = {}
    for k, b in enumerate(nf.BANDS):
        f = max(0.0, float(MU[1 + k] + z[1 + k] * SD[1 + k]))
        bands[b] = int(min(1e9, math.expm1(20.0 * f)))
    att = float(np.clip(MU[0] + z[0] * SD[0], 0.03, 0.99)) * 100.0
    std = 2000.0 * float(np.clip(MU[10] + z[10] * SD[10], 0.002, 3.0))
    mx = 6000.0 * float(np.clip(MU[11] + z[11] * SD[11], 0.002, 3.0))
    s = nf.Sample(t=0.0, signal=0, attention=int(round(att)), meditation=50, bands=bands)
    s.raw_n = 512
    s.raw_std = std
    s.raw_max = mx
    return s


def run(phase_len=30, att_shift=1.0, noise=1.0, band_shift=0.0, seed=0, minutes=2.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, 12)
    out = []
    n = int(minutes * 60)
    for i in range(n):
        focus = ((i // phase_len) % 2) == 0
        drift = (0.25 if focus else -0.25) * np.ones(12)
        drift[0] += att_shift if focus else -att_shift
        drift[1:9] += band_shift if focus else -band_shift
        z = 0.86 * z + 0.14 * (drift + rng.normal(0, noise, 12))
        out.append(z.copy())
    probs = []
    win = []
    for zz in out:
        win.append(sample_from_z(zz))
        if len(win) < nf.WINDOW:
            continue
        X = nf.frame_features(win[-nf.WINDOW:])
        Z = nf.zscore(X, MU, SD)
        probs.append(float(nf.sigmoid(net.forward(Z[None, :, :], cache=False))[0]))
    focus_flags = [(((i + nf.WINDOW - 1) // phase_len) % 2) == 0 for i in range(len(probs))]
    f = [p for p, ff in zip(probs, focus_flags) if ff]
    r = [p for p, ff in zip(probs, focus_flags) if not ff]
    return probs, f, r


print("att_shift noise  band_shift |  focus mean  rest mean  gap   min   max  frac>0.5")
for att_shift in (0.8, 1.2):
    for noise in (0.7, 1.0):
        for band_shift in (0.0, 0.3):
            probs, f, r = run(att_shift=att_shift, noise=noise, band_shift=band_shift)
            fm = sum(f) / len(f) if f else float("nan")
            rm = sum(r) / len(r) if r else float("nan")
            print(
                f"{att_shift:9.1f} {noise:5.1f} {band_shift:10.1f} | {fm:10.3f} {rm:9.3f} "
                f"{(fm-rm):6.3f} {min(probs):5.3f} {max(probs):5.3f} "
                f"{sum(1 for p in probs if p>0.5)/len(probs):7.2f}"
            )

print("\n== 细看一组（att_shift=1.2, noise=0.7, band_shift=0.3）==")
probs, f, r = run(att_shift=1.2, noise=0.7, band_shift=0.3)
line = "".join("F" if p > 0.5 else "." for p in probs)
print(line)
print("".join(str(int(p * 9)) for p in probs))
