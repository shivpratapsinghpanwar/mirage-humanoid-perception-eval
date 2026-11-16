"""Probe Mirage options and measure warm latency per mode, thread scaling and RAM.

Usage: python 02_options_latency.py <image_dir> <out.json>
"""
import glob
import json
import os
import statistics
import sys
import time

os.environ.setdefault("CACTUS_TELEMETRY", "0")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from PIL import Image  # noqa: E402

import cactus  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402

img_dir, out_path = sys.argv[1], sys.argv[2]
proc = psutil.Process()
rss0 = proc.memory_info().rss

m = SafeMirage()
rss_loaded = proc.memory_info().rss

# fixed set of 8 images, decoded once to uint8 arrays (camera-frame style input)
names = ["000000000139.jpg", "000000000785.jpg", "000000000872.jpg", "000000001000.jpg",
         "000000001268.jpg", "000000001296.jpg", "000000002153.jpg", "000000002299.jpg"]
frames = [np.asarray(Image.open(os.path.join(img_dir, n)).convert("RGB")) for n in names]
m.describe(frames[0])  # warm-up

report = {"machine": "Intel i5-11320H (4C/8T) laptop CPU, Windows 11, 2-bit .cact weights",
          "rss_mb": {"before_load": rss0 / 1e6, "after_load": rss_loaded / 1e6}}


def bench(label, reps=2, **opts):
    times, counts, sample = [], [], None
    for f in frames:
        for _ in range(reps):
            a = time.perf_counter()
            r = m.describe(f, **opts)
            times.append((time.perf_counter() - a) * 1000)
        counts.append(len(r.get("objects", [])))
        sample = sample or r
    row = {"median_ms": statistics.median(times), "p90_ms": sorted(times)[int(0.9 * len(times)) - 1],
           "min_ms": min(times), "mean_objects": statistics.mean(counts),
           "keys": list(sample.keys()), "opts": opts}
    print(f"{label:<38} median {row['median_ms']:7.0f} ms  p90 {row['p90_ms']:7.0f}  "
          f"objs {row['mean_objects']:.1f}  keys={row['keys']}", flush=True)
    return row


modes = {}
modes["default (full describe)"] = bench("default (full describe)")
modes["objects only"] = bench("objects only", sections=["objects"])
modes["objects, bbox+score only"] = bench("objects, bbox+score only", sections=["objects"], fields=["bbox", "score"])
modes["objects, conf=0.2"] = bench("objects, conf=0.2", sections=["objects"], conf=0.2)
modes["objects, conf=0.5"] = bench("objects, conf=0.5", sections=["objects"], conf=0.5)
modes["text only"] = bench("text only", sections=["text"])
modes["full, max_objects=8"] = bench("full, max_objects=8", max_objects=8)
form = {"person_present": {"enum": ["yes", "no"]}}
modes["extract(person_present yes/no)"] = bench("extract(person_present yes/no)", form=form)
report["modes"] = modes

# unknown-option handling
for bad in [{"not_an_option": 1}, {"sections": ["nonsense"]}]:
    try:
        r = m.describe(frames[0], **bad)
        report.setdefault("bad_option", []).append({"opts": bad, "result": "accepted", "keys": list(r.keys())})
    except Exception as e:  # noqa: BLE001
        report.setdefault("bad_option", []).append({"opts": bad, "result": f"error: {e}"})
print("bad options:", report["bad_option"], flush=True)

# determinism
a = m.describe(frames[0])
b = m.describe(frames[0])
report["deterministic_same_input"] = (a == b)
print("deterministic:", report["deterministic_same_input"], flush=True)

# thread scaling on objects-only and full
threads = {}
for t in [1, 2, 4, 8]:
    cactus.set_threads(t)
    m.describe(frames[0])
    row = {}
    for label, opts in [("full", {}), ("objects", {"sections": ["objects"]})]:
        ts = []
        for f in frames[:4]:
            s = time.perf_counter()
            m.describe(f, **opts)
            ts.append((time.perf_counter() - s) * 1000)
        row[label] = statistics.median(ts)
    threads[t] = row
    print(f"threads={t}: full {row['full']:.0f} ms, objects {row['objects']:.0f} ms", flush=True)
cactus.set_threads(0)
report["threads"] = threads

# input resolution sensitivity (model canvas is 512): same image at 3 sizes
im = Image.open(os.path.join(img_dir, names[0])).convert("RGB")
res = {}
for w in [320, 640, 1280, 1920]:
    arr = np.asarray(im.resize((w, int(w * im.height / im.width))))
    m.describe(arr)
    ts = []
    for _ in range(3):
        s = time.perf_counter()
        m.describe(arr)
        ts.append((time.perf_counter() - s) * 1000)
    res[w] = statistics.median(ts)
    print(f"input width {w}: {res[w]:.0f} ms", flush=True)
report["input_width_ms"] = res
report["rss_mb"]["peak_after_runs"] = proc.memory_info().peak_wset / 1e6 if hasattr(proc.memory_info(), "peak_wset") else proc.memory_info().rss / 1e6

report["nan_json_sanitized"] = {"calls": m.calls, "sanitized": m.sanitized}
print("NaN-JSON sanitized:", m.sanitized, "of", m.calls, "calls", flush=True)
json.dump(report, open(out_path, "w"), indent=1, default=str)
print(json.dumps(report["rss_mb"]))
