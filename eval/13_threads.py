"""Does cactus.set_threads(N), called BEFORE the model loads (cactus-compute issue #188), change Mirage's
speed and CPU use? Each setting runs in a fresh process on the same 20 COCO images (after 2 warm-ups).

Usage: python 13_threads.py <out.json> [settings=default,1,2,4,8]
Records per setting: median/p90 latency, mean process CPU use in cores (psutil), and whether the
describe() output is identical to the default setting.
"""
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

if len(sys.argv) > 2 and sys.argv[1] == "--child":
    setting = sys.argv[2]
    os.environ["CACTUS_TELEMETRY"] = "0"
    import cactus
    import psutil
    if setting != "default":
        cactus.set_threads(int(setting))
    m = cactus.Mirage()
    ids = json.load(open(os.path.join(ROOT, "results", "coco_mirage", "_sample_ids.json")))[:20]
    paths = [os.path.join(ROOT, "data", "coco_val2017", f"{i:012d}.jpg") for i in ids]
    for _ in range(2):
        m.describe(paths[0])
    proc = psutil.Process()
    lat, outs = [], []
    c0, w0 = proc.cpu_times(), time.perf_counter()
    for p in paths:
        t = time.perf_counter()
        r = m.describe(p)
        lat.append((time.perf_counter() - t) * 1000)
        outs.append([(o.get("name"), [round(x) for x in o.get("bbox", [])]) for o in r.get("objects", [])])
    c1, w1 = proc.cpu_times(), time.perf_counter()
    cores = ((c1.user - c0.user) + (c1.system - c0.system)) / (w1 - w0)
    lat.sort()
    print(json.dumps({"setting": setting, "median_ms": round(lat[len(lat) // 2]), "p90_ms": round(lat[int(len(lat) * .9)]),
                      "mean_ms": round(statistics.mean(lat)), "cpu_cores_used": round(cores, 2), "outputs": outs}))
    sys.exit(0)

out_path = sys.argv[1]
settings = (sys.argv[2] if len(sys.argv) > 2 else "default,1,2,4,8").split(",")
res = []
for s in settings:
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", s], capture_output=True, text=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if not line:
        res.append({"setting": s, "error": (r.stderr or "")[-400:]})
        continue
    res.append(json.loads(line[-1]))
    print({k: v for k, v in res[-1].items() if k != "outputs"}, flush=True)
ref = next((r["outputs"] for r in res if r.get("setting") == "default" and "outputs" in r), None)
for r in res:
    if "outputs" in r:
        r["outputs_identical_to_default"] = r["outputs"] == ref
        del r["outputs"]
import importlib.metadata as md  # noqa: E402
doc = {"cactus_compute": md.version("cactus-compute"), "images": 20, "logical_cpus": os.cpu_count(), "results": res}
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(doc, f, indent=1)
print(json.dumps(doc, indent=1))
