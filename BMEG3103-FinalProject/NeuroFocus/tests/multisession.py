"""回归：多会话划分 / 交叉验证 / 两阶段训练 / 外部数据导入。

覆盖这次改动的关键不变量：
  1. build_dataset 带回 subjects / sources；
  2. 按会话留出的训练组与验证组不共享任何会话；
  3. 留一会话交叉验证给出 k 折与 mean±std；
  4. 只有 1 个会话时明确报警，而不是偷偷随机切分；
  5. 用外部预训练权重初始化能跑通，meta.init 记录正确；
  6. fit_final 用全部会话训练、n_val=0；
  7. 外部试次能被 build_dataset 识别（source=external）。
"""
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())

import main as nf

tmp = os.path.join(nf.DATA_DIR, "_test_multi")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
paths = nf.make_demo_set(tmp, subjects=("M1", "M2", "M3", "M4"), blocks=4, block_sec=40)

# 1) 身份层
assert nf.source_of(paths[0]) == "sim"
assert nf.subject_of(paths[0]) == "M1"
ds = nf.build_dataset(paths)
assert ds["subjects"].shape == ds["y"].shape
assert set(ds["subjects"].tolist()) == {"M1", "M2", "M3", "M4"}
assert len(set(ds["groups"].tolist())) == 4
print("[1] 身份层 OK：%d 窗 / 4 会话 / 被试 %s" % (len(ds["y"]), sorted(set(ds["subjects"].tolist()))))

# 2) 按会话留出
tr, va = nf.split_dataset(ds, val_ratio=0.25, seed=1, by="session")
assert set(ds["groups"][tr].tolist()).isdisjoint(set(ds["groups"][va].tolist()))
print("[2] 按会话留出 OK：验证会话 %s" % sorted(set(ds["groups"][va].tolist())))

# 3) 留一会话交叉验证
cv = nf.cv_sessions(ds, folds=4, by="session", epochs=40, lr=2e-3)
assert cv["k"] == 4 and len(cv["folds"]) == 4
a = cv["agg"]["acc"]
assert 0.0 <= a["mean"] <= 1.0 and a["std"] >= 0.0
print("[3] 留一会话 4 折 OK：acc %.3f ± %.3f，基线 %.3f"
      % (a["mean"], a["std"], cv["agg"]["baseline_acc"]["mean"]))

# 4) 单会话护栏
one = nf.build_dataset(paths[:1])
assert "只有 1 个分组" in nf.split_note(one, "session")
try:
    nf.group_kfold(one, "session", k=4)
    raise AssertionError("单会话不应能构造交叉验证折")
except ValueError:
    pass
print("[4] 单会话护栏 OK：明确提示指标会偏乐观")

# 5) 两阶段：外部权重初始化
res0 = nf.train_from_dataset(ds, epochs=30, lr=2e-3, seed=0)
st = {k: v.copy() for k, v in res0["net"].state().items()}
res1 = nf.train_from_dataset(ds, epochs=15, lr=1e-3, seed=0, init_state=st, init_name="external")
assert res1["meta"]["init"] == "external"
assert res1["meta"]["n_val"] > 0
print("[5] 预训练初始化 OK：acc %.3f → %.3f（meta.init=%s）"
      % (res0["metrics"]["acc"], res1["metrics"]["acc"], res1["meta"]["init"]))

# 6) 最终模型：全部会话、无验证集
fin = nf.fit_final(ds, epochs=10, lr=2e-3)
assert fin["meta"]["fit"] == "all-sessions"
assert fin["meta"]["n_train"] == len(ds["y"]) and fin["meta"]["n_val"] == 0
assert len(fin["meta"]["trained_on"]) == 4
print("[6] 最终模型 OK：%d 窗 / %d 会话，n_val=0" % (fin["meta"]["n_train"], len(fin["meta"]["trained_on"])))

# 7) 外部数据（若已导入）
ext = sorted(p for p in nf.glob.glob(os.path.join(nf.EXTERNAL_DIR, "*.csv")) if nf.is_raw_csv(p))
if ext:
    ed = nf.build_dataset(ext[:6])
    assert set(ed["sources"].tolist()) == {"external"}
    assert ed["reject"]["伪迹"] == 0
    print("[7] 外部数据 OK：%d 个试次 → %d 窗（source=external）" % (len(ext[:6]), len(ed["y"])))
else:
    print("[7] 跳过外部数据（data/external 为空）")

shutil.rmtree(tmp, ignore_errors=True)
print("multisession ok")
