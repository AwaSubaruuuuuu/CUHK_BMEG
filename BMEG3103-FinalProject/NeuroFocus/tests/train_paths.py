"""回归：④ 训练页在两种情况下都能开训。

1) ③ 里构建过数据集 → 直接拿它训练（不重新构建），即使左侧列表没有选中项；
2) 没有构建过 → 用 ③ 列表里的文件现场构建。
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import tkinter as tk

import main as nf

nf.messagebox.showwarning = lambda *a, **k: print("WARN", a, flush=True)
nf.messagebox.showerror = lambda *a, **k: print("ERR", a, flush=True)

root = tk.Tk()
app = nf.App(root)
tr = app.tab_train
ds_tab = app.tab_dataset
root.update_idletasks()

demo = sorted(nf.glob.glob(os.path.join(nf.RAW_DIR, "*_demo_sim.csv")))
assert demo, "需要 data/raw/*_demo_sim.csv"


def wait_train(limit=180):
    t0 = time.time()
    while time.time() - t0 < limit:
        root.update()
        time.sleep(0.03)
        if tr.result is not None:
            return True
    return False


# --- 1) 已构建数据集、列表清空选中 ---
ds_tab.reload(demo)
ds = nf.build_dataset(demo[:2])
ds_tab.ds = app.ds = ds
tr.on_dataset(ds)
ds_tab.lb.selection_clear(0, "end")
assert ds_tab.selected_paths() == ds_tab.files, "没勾选时应回退为列表全部文件"
tr.epochs.set("25")
tr.toggle()
assert wait_train(), "训练超时"
print("case1 ok：用已构建的 %d 窗训练，acc=%.3f" % (len(ds["y"]), tr.result["metrics"]["acc"]))
assert tr.result["ds"] is ds, "应直接复用已构建的数据集"

# --- 2) 没构建过数据集：现场构建 ---
tr.result = None
ds_tab.invalidate()
tr.epochs.set("25")
tr.toggle()
assert wait_train(), "训练超时"
print("case2 ok：现场构建后训练，窗数=%d" % len(tr.result["ds"]["y"]))
assert len(tr.result["ds"]["y"]) > len(ds["y"])

root.destroy()
print("train paths ok")
