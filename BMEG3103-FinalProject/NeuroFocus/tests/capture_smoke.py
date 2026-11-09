# -*- coding: utf-8 -*-
"""采集页「闭眼默算」协议的自动化烟测（无显示器环境用 root.update() 泵事件）。

覆盖：
1. 旧的出题/判分组件已移除（被试不再与界面交互）；
2. 模拟器采集时，静息段显示「闭眼静息」、心算段显示「1000 − 7」，提示语随段切换；
3. 段末报数写进 *_tasks.csv（answer 可为空，result 固定 -1 表示不计分）；
4. 非法输入不入库；采集结束后统计与行数一致；
5. *_tasks.csv 不会被当成 EEG 会话混进数据集。
"""
import os
import sys
import time
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import main as nf


def pump(root, seconds):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.02)


def main():
    # 1) 交互组件确实没了
    assert not hasattr(nf, "MentalMath"), "旧的心算出题类应已移除"
    print("[1] 无出题/判分类 OK")

    # 2) GUI
    root = tk.Tk()
    root.geometry("1280x820")
    app = nf.App(root)
    app.attach_source(nf.SimulatorSource(app.q), "模拟器(烟测)")
    tab = app.tab_capture
    assert not hasattr(tab, "q_entry"), "被试侧不应再有答案输入框"
    tab.secs.set("6")
    tab.nblocks.set("2")          # 第 1 段静息、第 2 段心算
    tab.subj.set("MTEST")
    pump(root, 0.4)
    assert tab.q_text.cget("text") == "未开始"
    tab.toggle()

    pump(root, 1.5)
    assert tab.plan[0][0] == 0
    assert tab.q_text.cget("text") == "闭眼静息", tab.q_text.cget("text")
    assert tab.q_tip.cget("text") == nf.CaptureTab.TIP_REST
    print("[2] 静息段显示闭眼静息 OK")

    end = time.time() + 20
    while time.time() < end and tab.plan[tab.idx][0] != 1:
        pump(root, 0.2)
    assert tab.plan[tab.idx][0] == 1, "未进入心算段"
    pump(root, 0.8)
    assert tab.q_text.cget("text") == "1000 − 7", tab.q_text.cget("text")
    assert tab.q_tip.cget("text") == nf.CaptureTab.TIP_MATH
    print("[3] 心算段显示 1000 − 7 OK")

    # 4) 段末报数
    tab.rep.set("937")
    tab.record_report()
    pump(root, 0.3)
    assert tab.reports == [937], tab.reports
    assert "1 段" in tab.q_stats.cget("text")
    print("[4] 报数记录 OK：", tab.report_summary())

    # 5) 非法输入不入库
    tab.rep.set("abc")
    tab.record_report()
    pump(root, 0.2)
    assert tab.reports == [937], "非法输入不应入列"
    assert "整数" in tab.q_stats.cget("text")
    tab.rep.set("")
    tab.record_report()             # 空值 = 未报，允许
    assert tab.reports == [937, None]
    print("[5] 非法输入被拒、空值记为未报 OK")

    end = time.time() + 20
    while tab.running and time.time() < end:
        pump(root, 0.2)
    assert not tab.running
    pump(root, 0.3)

    tpath = tab.task_path()
    assert tpath and os.path.exists(tpath), "未生成 tasks.csv"
    lines = open(tpath, encoding="utf-8").read().strip().splitlines()
    assert lines[0] == "t,qid,question,answer,result"
    body = lines[1:]
    assert len(body) == len(tab.reports), f"tasks 行数 {len(body)} != 报数次数 {len(tab.reports)}"
    assert body[0].split(",")[3] == "937"
    assert all(l.rsplit(",", 1)[1] == "-1" for l in body), "本协议报数不计分，result 恒为 -1"
    print(f"[6] tasks.csv OK：{os.path.basename(tpath)} {len(body)} 行")

    # 6) tasks.csv 不能被当成 EEG 会话
    app.refresh_raw_list()
    listed = [os.path.basename(p) for p in app.raw_files]
    assert os.path.basename(tab.path) in listed, "原始会话应出现在文件列表"
    assert os.path.basename(tpath) not in listed, "tasks.csv 不应出现在数据集文件列表"
    assert not nf.is_raw_csv(tpath)
    demo = sorted(nf.glob.glob(os.path.join(nf.RAW_DIR, "*_demo_sim.csv")))
    if demo:
        ds = nf.build_dataset([demo[0], tpath])
        assert ds["X"].shape[1:] == (nf.WINDOW, 14)
        print(f"[7] tasks.csv 已排除在数据集之外 OK：{ds['X'].shape[0]} 窗")
    else:
        print("[7] 无 demo 会话，跳过入库检查")

    for p in (tab.path, tpath):
        try:
            os.remove(p)
        except OSError:
            pass
    if tab.quality_path():
        try:
            os.remove(tab.quality_path())
        except OSError:
            pass

    # 7) 体检函数：人造两组，心算幅度 3 倍、alpha 掉一半 → 应告警
    def fake(label, n, std, alpha):
        return [nf.Sample(t=0.0, signal=0, attention=50, meditation=50,
                          bands={"delta": 100000, "highAlpha": alpha},
                          raw_n=512, raw_std=std, raw_max=std * 3, label=label)
                for _ in range(n)]

    rows = fake(0, 60, 50.0, 30000) + fake(1, 60, 150.0, 9000)
    lines, verdict = nf.quality_lines(rows)
    assert verdict == "warn", (verdict, lines)
    assert "睁眼" in lines[-1], lines[-1]
    rows = fake(0, 60, 50.0, 30000) + fake(1, 60, 55.0, 28000)
    lines, verdict = nf.quality_lines(rows)
    assert verdict == "ok", (verdict, lines)
    assert nf.quality_lines(fake(0, 5, 50.0, 30000))[1] == "unknown"
    print("[8] 采集体检 quality_lines OK（对齐通过 / 眼动告警 / 数据不足）")

    root.destroy()
    print("capture panel ok")


if __name__ == "__main__":
    main()
