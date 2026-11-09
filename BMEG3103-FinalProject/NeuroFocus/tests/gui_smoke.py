import sys, time, os, importlib.util
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
spec=importlib.util.spec_from_file_location("nf","main.py"); nf=importlib.util.module_from_spec(spec); sys.modules["nf"]=nf
spec.loader.exec_module(nf)
nf.messagebox.showerror=lambda *a,**k: print("MSGBOX-ERR",a,flush=True)
nf.messagebox.showwarning=lambda *a,**k: print("MSGBOX-WARN",a,flush=True)
nf.messagebox.showinfo=lambda *a,**k: print("MSGBOX-INFO",a,flush=True)
import tkinter as tk
print("create root",flush=True)
root=tk.Tk(); app=nf.App(root); root.update(); print("app ok",flush=True)
app.attach_source(nf.SimulatorSource(app.q), "sim"); print("attached",flush=True)
def pump(sec):
    t0=time.time()
    while time.time()-t0<sec:
        root.update(); time.sleep(0.05)
pump(3); print("samples:", len(app.recent),flush=True)
for subj in ("T01","T02"):
    cap=app.tab_capture
    cap.subj.set(subj); cap.nblocks.set("2"); cap.secs.set("6")
    cap.toggle(); print("capture start",subj,flush=True)
    pump(14)
    print(subj, "exists:", os.path.exists(cap.path),flush=True)
print("raw files:", len(app.raw_files),flush=True)
app.tab_dataset.build(); print("build done",flush=True)
print("dataset:", None if app.ds is None else app.ds["X"].shape,flush=True)
tr=app.tab_train; tr.epochs.set("30"); tr.toggle(); print("train start",flush=True)
t0=time.time()
while time.time()-t0<90:
    root.update(); time.sleep(0.05)
    if tr.result: break
print("trained:", tr.result is not None,flush=True)
if tr.result:
    print("acc:", round(tr.result["metrics"]["acc"],3),flush=True)
    app.tab_live.load(); app.tab_live.toggle(); pump(10)
    print("probs:", len(app.infer_probs),flush=True)
app.detach_source(); root.destroy()
for p in nf.glob.glob(os.path.join(nf.RAW_DIR, "T0*")):
    try: os.remove(p)
    except OSError: pass
print("integration ok",flush=True)
