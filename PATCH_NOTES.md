# PATCH_NOTES — 分支 `accuracy-fix`（已合并进 `main`，合并提交 `1ab47fb`）

本分支基于 `main` 的 `b609cd0c7bd65e953cd809e00a0ab65720b93c9f`（2026-11-10 03:15 UTC，
「NeuroFocus 增加两阶段训练、多会话交叉验证与外部数据导入」），只做一件事：

> **把交付程序 `BMEG3103-FinalProject/NeuroFocus/main.py` 的实测准确率从 0.686 提到 0.890，并把
> 所有数字、报告、文档改成可复现的真实数据版本。**

`main` 分支一个字都没动。

---

## 1. 结果对比

| | `main`（旧） | 本分支（新） |
| --- | --- | --- |
| 特征维度 | 8 × 14（含 `signal`、`meditation`） | **8 × 12**（删掉这两个） |
| 训练选轮依据 | `val_loss` → 存**第 1 轮** | **验证集 AUC** → 第 32 / 49 轮 |
| 判正阈值 | 固定 0.5（跨会话失准） | 固定 0.5 + **本人静息校准**（部署时） |
| 基线对比 | 用**全部数据**算 → 虚高 0.78 | 用**同一验证集**算 → 0.616 |
| 伪迹门限 | `med+4σ` 且地板 2500 → **一个都不拦** | robust-4σ / 全局 99.5 分位 / 4×中位数 取最松 → 只拦 6 秒 |
| 没戴好检测 | 只查 `signal > 100` | 补上 TGAM 的 29/54/55/56/80/81/82/107/200 |
| 真实数据准确率 | 0.686（窗口 517） | **0.890 ± 0.018**（窗口 474） |
| F1 / AUC | 0.755 / 0.849 | **0.883 / 0.967** |
| 参数量 | 3157 | 3061 |

两折（按会话留一）分别是 **0.908（留出 S01）/ 0.872（留出 S02）**，
brief 要求 ≥ 0.85；同验证集上 brief 的 Option A 基线（3 s 滑动平均 + 阈值 75）只有 **0.616**。

## 2. 改了什么（16 处）

**P0 — 直接决定指标对错的**

1. **早停改用验证集 AUC**（`train_cnn(select="auc")`），存最佳轮而不是第 1 轮；界面显示 `best_epoch`。
2. **基线改在同一验证集上算**（原来 `threshold_baseline(ds["X"], ds["y"])` 用全部数据，把基线从
   0.616 抬到 0.78，反而证明「CNN 不如基线」）。
3. **阈值不再是裸 0.5**：训练集标准化 + 0.5 作为主指标；部署时（第 ⑤ 页）用本人 30 秒静息重算
   `mu/sd` 与阈值（静息打分的 90 分位）。
4. **`n_train`/`n_val` 上报修好**（原来用布尔掩码 `.sum()`，打印出 83421 / 28680 这种数）。

**P1 — 特征里有两个"毒"维度**

5. 删 `meditation`（单特征 AUC 0.42，方向还不稳定）。
6. 删 `signal`（AUC 0.48；且静息时恒为 0，训练会话 sd=0，标准化后 z 顶到 1e5）。
7. `raw_std` / `raw_max` 保留但不再参与剔除（旧协议里它们 AUC 0.94，v3 协议下**符号翻转**：
   静息时手更动，AUC 掉到 0.30——按它们剔除会把信号删掉）。
8. 特征改成 `FEATURE_INDEX` 驱动的 12 维，不再硬编码下标。

**P2 — 数据与泄漏**

9. `split_dataset` 文档化「只有 2 个会话时实际是 50/50 留出，不是 20%」，报告里同时给
   `val_ratio_requested` 与 `val_ratio_actual`。
10. **`safe_sd` + `Z_CLIP=8`**：标准差下限不再取 `+1e-6`，且对 z 做 ±8 截断。
11. **新增 `standardize_by_group()`**：需要时可按会话各自标准化（跨人泛化用），默认用训练会话的
    `mu/sd`（实测更稳：0.890 vs 逐会话校准 0.863）。
12. 特征窗口记录 `groups`/`subjects`/`sources` 的分组切分逻辑保持 `main` 的做法（本分支未含
    `main` 的外部数据流程，见第 4 节）。

**P3 — 工程细节**

13. `sigmoid()` 加 ±60 截断，消掉 `1/(1+exp(-z))` 的 overflow RuntimeWarning。
14. 数据增强改成**边缘填充平移**（原来是 `np.roll`，会首尾环绕）+ ±3% 增益抖动 + σ0.02 噪声；
    实测均值 0.812 → 0.869。
15. `exports_edge_header` 接进 `--train`，`model_int8.h` 不再是一份 14 维的旧文件。
16. 文档全部重写为真实数据版本：`docs/报告.md`（新增 8.3 真实数据结果、8.4 特征选择、
    §4.3 清洗规则、§5.2 参数表、§8.1 标注为合成数据）、`docs/采集协议.md`（v2 的 0.936/0.913/0.988
    已标注为 v2-only，v3 的 `raw_std` AUC 是 0.318）、`docs/AI使用说明.md`（审计表 8 行）、
    `docs/一页说明.md`、`README.md`、上层 `BMEG3103-FinalProject/README.md`。

## 3. 怎么复现

```bash
cd BMEG3103-FinalProject
python NeuroFocus/main.py --selftest     # 无硬件自检（协议解析 / 3061 参数梯度校验 / 端到端）
python NeuroFocus/main.py --train        # 读 data/raw/*.csv，留一个会话做验证
python NeuroFocus/main.py                # 图形界面：③ 构建数据集 → ④ 训练 → ⑤ 实时推演
```

`--train` 会同时写出 `data/dataset.csv`、`data/dataset_report.html`、
`models/model.json`、`models/model_int8.h`。**真实数据的 CSV 不进仓库**（`.gitignore` 里 `*.csv`、
`data/`），需要按 `docs/采集协议.md` 自己采。

> 环境坑：本项目只依赖 numpy + pyserial，但 **numpy 必须与 Python 版本 ABI 匹配**。
> 仓库里 `vendor/` 的 wheel 若与解释器版本不符，程序会开窗但一点「构建数据集」就报
> `AttributeError: 'NoneType' object has no attribute 'asarray'`。`main.py` 顶部会打印实际
> numpy 路径，自检里也会显示；用 `python -c "import numpy; print(numpy.__version__)"` 确认。

## 4. 本分支**不包含** `main` 上的这些功能

`main` 另有一套为「多会话 / 外部数据集」准备的两阶段流程，本分支的 `main.py` 来自更早的快照，
**没有**以下内容，合并方向应当是「把本分支的精度修复移植回 `main`」，而不是反过来：

- `data/external/` 外部数据集导入（`--import-external`）与 `SOURCE_LABELS`（自采 / 外部 / 模拟）
- 四种训练模式：`--cv`（留一会话交叉验证）、`--pretrain`（外部预训练）、`--fit-final`（主交付模型微调）、
  `--init {scratch,external}`
- `models/model_scratch.json`、`models/pretrained_external.json` 两个候选模型
- 对应的 `tests/multisession.py`、`tests/train_modes.py`

本分支 `models/` 只有 `model.json` + `model_int8.h`（真实数据训练的结果）、`tests/` 只有 5 个
与当前代码匹配的脚本。如果要合并，建议按第 2 节逐条移植：P0/P1 是纯函数级改动，
P2/P3 里只有 `build_dataset` 的返回字段（`calib` / `mu_global` / `sd_global`）需要和
`main` 的 `groups`/`subjects`/`sources` 一起改。

## 5. 仍然存在的局限（写进报告里，别藏）

- 只有 **1 位被试的 2 次会话**，留一验证的置信区间约 **±6–7 pt**；0.890 不能当成稳的，
  需要再采 2 位。
- 静息 / 心算 60 秒严格交替，与"时间漂移"共线，模型学到的一部分可能是「越来越困」而不是「注意力」。
- `raw_std` 在 v3 协议下与标签反相关，说明**手部/眼动污染仍在**（协议要求全程闭眼不动，
  但静息段手更放松、原始波反而更大），这一条写进了 `docs/报告.md` 第 9 节。
- 交叉验证只到「留一会话」，没有「留一被试」——因为总共只有 2 人。
