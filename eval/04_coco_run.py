"""Run Mirage (default full describe) on a seeded random sample of COCO val2017.

Usage: python 04_coco_run.py <val2017_dir> <instances_val2017.json> <out_dir> [n=300] [seed=1234]
Writes <out_dir>/<image_id>.json (raw Mirage output) and <out_dir>/_index.jsonl
(per-image latency). Resumable: existing outputs are skipped.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402

img_dir, ann_path, out_dir = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1234
os.makedirs(out_dir, exist_ok=True)

coco = json.load(open(ann_path, encoding="utf-8"))
has_ann = {a["image_id"] for a in coco["annotations"] if not a.get("iscrowd")}
ids = sorted(i["id"] for i in coco["images"] if i["id"] in has_ann)
random.Random(seed).shuffle(ids)
sample = ids[:n]
json.dump(sample, open(os.path.join(out_dir, "_sample_ids.json"), "w"))

m = SafeMirage()
idx = open(os.path.join(out_dir, "_index.jsonl"), "a", encoding="utf-8")
t_start = time.time()
done = 0
for k, iid in enumerate(sample):
    out = os.path.join(out_dir, f"{iid}.json")
    if os.path.exists(out):
        continue
    path = os.path.join(img_dir, f"{iid:012d}.jpg")
    a = time.perf_counter()
    try:
        r = m.describe(path)
        err = None
    except Exception as e:  # noqa: BLE001
        r, err = None, repr(e)
    ms = (time.perf_counter() - a) * 1000
    if r is not None:
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    idx.write(json.dumps({"image_id": iid, "ms": ms, "error": err}) + "\n")
    idx.flush()
    done += 1
    if done % 25 == 0:
        el = time.time() - t_start
        print(f"{k + 1}/{n}  {el:.0f}s elapsed  sanitized={m.sanitized}", flush=True)
print(f"done {done} images; NaN-JSON sanitized {m.sanitized}/{m.calls}", flush=True)
