import json, sys, time, urllib.request, collections

# 用法：py -3 probe_sim.py [url] [秒数]
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/events"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
rows = []
t_end = time.time() + secs
with urllib.request.urlopen(url, timeout=100) as r:
    for line in r:
        if time.time() > t_end:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        m = json.loads(line[5:])
        if m.get("type") != "score":
            continue
        rows.append((time.strftime("%H:%M:%S", time.localtime(m["t"])), m["p"], m["ema"], m["attention"]))
        print(rows[-1][0], f"p={m['p']:.3f} ema={m['ema']:.3f} att={m['attention']}")

print("\n== summary ==")
print("n =", len(rows))
if rows:
    ps = [r[1] for r in rows]
    print(f"min={min(ps):.3f} max={max(ps):.3f} mean={sum(ps)/len(ps):.3f}")
    for thr in (0.2, 0.1, 0.5):
        print(f"  p>={thr}: {sum(1 for p in ps if p >= thr)}/{len(ps)}")
    bucket = collections.Counter()
    for _, p, _, _ in rows:
        bucket[round(p, 1)] += 1
    print("hist:", dict(sorted(bucket.items())))
