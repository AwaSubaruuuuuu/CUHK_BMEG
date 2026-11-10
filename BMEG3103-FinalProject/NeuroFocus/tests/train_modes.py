"""回归：④ 训练页的四种模式都能跑通（用很小的轮数）。

这层本来没有自动化测试，模式改写后最容易在「线程 + 队列 + UI 状态」上出问题，
所以这里把四种模式依次点一遍，检查：
  1. 快速训练：meta.split_by 正确、指标非 nan；
  2. 交叉验证：折数等于会话数、表格行数对得上、agg 来自各折；
  3. 最终模型：meta.fit=all-sessions、n_val=0、轮数取 CV 最佳轮中位数；
  4. 外部预训练：meta.cv 存在、权重文件落盘（仅当已导入外部数据）。
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import tkinter as tk

import main as nf

nf.messagebox.showwarning = lambda *a, **k: print("WARN", a[0], flush=True)
nf.messagebox.showerror = lambda *a, **k: print("ERR", a[0], flush=True)

root = tk.Tk()
app = nf.App(root)
tr = app.tab_train
ds_tab = app.tab_dataset
root.update_idletasks()

demo = sorted(nf.glob.glob(os.path.join(nf.RAW_DIR, "*_demo_sim.csv")))
assert len(demo) >= 3, "需要至少 3 份 data/raw/*_demo_sim.csv"
ds = nf.build_dataset(demo[:3])
ds_tab.reload(demo)
ds_tab.ds = app.ds = ds
tr.on_dataset(ds)
tr.epochs.set("12")
tr.lr.set("0.002")


def run(mode, folds="3", expect="result"):
    tr.result = None
    if expect == "cv":
        tr.cv = None
        tr.cvtable.delete(*tr.cvtable.get_children())
    tr.mode.set(mode)
    tr.folds.set(folds)
    tr.update_hint()
    tr.toggle()
    t0 = time.time()
    done_at = None
    while time.time() - t0 < 600:
        root.update()
        time.sleep(0.02)
        if tr.thread and tr.thread.is_alive():
            continue
        # 线程结束后还要让 app.tick 把队列里的结果搬到 UI 上
        done_at = done_at or time.time()
        if (tr.cv is not None) if expect == "cv" else (tr.result is not None):
            break
        if time.time() - done_at > 5:
            break
    assert not (tr.thread and tr.thread.is_alive()), "训练线程超时：%s" % mode


# 1) 快速训练
run(tr.MODES[0])
assert tr.result is not None and tr.result["meta"]["split_by"] == "session", tr.result
print("[1] 快速训练 OK：%d 窗，验证组 %s，acc %.3f"
      % (len(tr.result["ds"]["y"]), tr.result["meta"]["val_groups"], tr.result["metrics"]["acc"]))

# 2) 交叉验证
run(tr.MODES[1], folds="3", expect="cv")
assert tr.cv is not None and tr.cv["k"] == 3, tr.cv
assert len(tr.cvtable.get_children()) == 3, "交叉验证表格应有 3 行"
a = tr.cv["agg"]["acc"]
assert 0.0 <= a["mean"] <= 1.0 and a["std"] >= 0.0
print("[2] 交叉验证 OK：3 折 acc %.3f ± %.3f，表格 %d 行，基线 %.3f"
      % (a["mean"], a["std"], len(tr.cvtable.get_children()),
         tr.cv["agg"]["baseline_acc"]["mean"]))

# 3) 最终模型：轮数应取自上一步 CV 的最佳轮中位数
want = nf.epochs_from_cv(tr.cv, cap=12)
run(tr.MODES[2])
meta = tr.result["meta"]
assert meta["fit"] == "all-sessions" and meta["n_val"] == 0
assert meta["epochs_run"] == want, (meta["epochs_run"], want)
print("[3] 最终模型 OK：%d 窗 / %d 会话 / %d 轮（CV 最佳轮中位数）/ n_val=0"
      % (meta["n_train"], len(meta["trained_on"]), meta["epochs_run"]))

# 4) 外部预训练
if app.external_files:
    run(tr.MODES[3], folds="2")
    assert tr.result is not None and "cv" in tr.result["meta"], "预训练结果缺少 cv"
    assert os.path.isfile(nf.MODEL_PATH_PRETRAIN)
    c = tr.cv["agg"]["acc"]
    print("[4] 外部预训练 OK：%d 窗，%d 折 acc %.3f ± %.3f → %s"
          % (len(tr.result["ds"]["y"]), tr.cv["k"], c["mean"], c["std"],
             os.path.basename(nf.MODEL_PATH_PRETRAIN)))
else:
    print("[4] 跳过外部预训练（data/external 为空）")

root.destroy()
print("train modes ok")
