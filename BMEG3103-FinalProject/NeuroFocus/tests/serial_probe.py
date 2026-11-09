"""连真机做 6 秒探测：确认 SerialSource 能收到数据并落成一行行特征。"""
import sys, os, time, queue
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as M

port = sys.argv[1] if len(sys.argv) > 1 else M.DEFAULT_PORT
q = queue.Queue()
src = M.SerialSource(q, port, M.DEFAULT_BAUD)
src.start()
t0 = time.time()
while not src.ok and time.time() - t0 < 5:
    time.sleep(0.1)
if not src.ok:
    print("OPEN FAILED:", src.message)
    sys.exit(2)
rows = []
while time.time() - t0 < 6:
    try:
        rows.append(q.get(timeout=0.5))
    except queue.Empty:
        pass
src.stop(); src.join(timeout=2)
print("port", port, "rows", len(rows), "bad_frames", getattr(src, "n_bad", "n/a"))
for r in rows[-3:]:
    print(f"t={r.t:6.1f} sig={r.signal:3d} att={r.attention:3d} med={r.meditation:3d} "
          f"delta={r.bands['delta']:8.0f} beta={r.bands['highBeta']:8.0f} "
          f"raw_n={r.raw_n:5d} std={r.raw_std:7.1f}")
