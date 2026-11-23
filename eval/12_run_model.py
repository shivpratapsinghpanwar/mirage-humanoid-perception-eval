"""Run one model over COCO-300 and the probe images, one JSON per image in the shared format.

Usage: python 12_run_model.py <model> [--device cpu|cuda] [--tasks coco,null,stab] [--threshold 0.05]
                              [--limit N] [--threads N] [--out results/runs/<model>/<device>]
Writes <out>/<task>/<key>.json, <out>/<task>/_index.jsonl ({"key", "ms", "error"} per image) and
<out>/run_meta.json (model info, peak memory, hardware). Resumable: existing outputs are skipped.
Latency = wall time from image path to output, after 2 warm-up runs (not recorded).
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--device", default="cpu")
ap.add_argument("--tasks", default="coco,null,stab")
ap.add_argument("--threshold", type=float, default=0.05)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out_root = a.out or os.path.join(ROOT, "results", "runs", a.model, a.device.split(":")[0])


def items(task):
    if task == "coco":
        ids = json.load(open(os.path.join(ROOT, "results", "coco_mirage", "_sample_ids.json")))
        return [(str(i), os.path.join(ROOT, "data", "coco_val2017", f"{i:012d}.jpg")) for i in ids]
    d = os.path.join(ROOT, "results", "probes", task)
    return [(os.path.splitext(f)[0], os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".png")]


kw = {"threads": a.threads} if a.threads else {}
t_load = time.perf_counter()
model = adapters.load(a.model, device=a.device, threshold=a.threshold, **kw)
load_s = time.perf_counter() - t_load
tasks = a.tasks.split(",")
first = items(tasks[0])[0][1]
for _ in range(2):  # warm-up
    model(first)

counts = {}
for task in tasks:
    od = os.path.join(out_root, task)
    os.makedirs(od, exist_ok=True)
    todo = items(task)[: a.limit or None]
    idx = open(os.path.join(od, "_index.jsonl"), "a", encoding="utf-8")
    n = 0
    for k, (key, path) in enumerate(todo):
        op = os.path.join(od, f"{key}.json")
        if os.path.exists(op):
            continue
        t = time.perf_counter()
        try:
            r, err = model(path), None
        except Exception as e:  # noqa: BLE001 - record and continue
            r, err = None, repr(e)[:300]
        ms = (time.perf_counter() - t) * 1000
        if r is not None:
            with open(op, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False)
        idx.write(json.dumps({"key": key, "ms": ms, "error": err}) + "\n")
        idx.flush()
        n += 1
        if n % 50 == 0:
            print(f"{a.model} {task} {k + 1}/{len(todo)}", flush=True)
    idx.close()
    counts[task] = n
    print(f"{a.model} {task}: {n} new outputs", flush=True)

meta = {"model": a.model, **model.info(), "params": adapters.count_params(model), "load_s": round(load_s, 2), "tasks": counts,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "cpu": platform.processor(), "os": platform.platform()}
try:
    import psutil
    mi = psutil.Process().memory_info()
    meta["peak_rss_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)
except ImportError:
    pass
if a.device.startswith("cuda"):
    import torch
    meta["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
mp = os.path.join(out_root, "run_meta.json")
prev = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
with open(mp, "w", encoding="utf-8", newline="\n") as f:
    json.dump({**prev, **meta}, f, indent=1)
print(json.dumps(meta, indent=1))
