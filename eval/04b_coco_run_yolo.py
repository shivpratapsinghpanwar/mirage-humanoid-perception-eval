"""Baseline: run an Ultralytics YOLO model on the same COCO sample, saving Mirage-style JSON.

Usage: python 04b_coco_run_yolo.py <val2017_dir> <mirage_out_dir (for _sample_ids.json)> <out_dir> [weights]
bbox saved as [x1,y1,x2,y2] on the same 0-1000 per-axis scale Mirage uses.
"""
import json
import os
import shutil
import statistics
import sys
import time

from ultralytics import YOLO

img_dir, mirage_dir, out_dir = sys.argv[1:4]
weights = sys.argv[4] if len(sys.argv) > 4 else "yolo26n.pt"
os.makedirs(out_dir, exist_ok=True)
shutil.copy(os.path.join(mirage_dir, "_sample_ids.json"), os.path.join(out_dir, "_sample_ids.json"))
sample = json.load(open(os.path.join(out_dir, "_sample_ids.json")))

model = YOLO(weights)
model.predict(os.path.join(img_dir, f"{sample[0]:012d}.jpg"), verbose=False, device="cpu")  # warm-up
idx = open(os.path.join(out_dir, "_index.jsonl"), "w", encoding="utf-8")
times = []
for iid in sample:
    path = os.path.join(img_dir, f"{iid:012d}.jpg")
    a = time.perf_counter()
    res = model.predict(path, verbose=False, device="cpu")[0]
    ms = (time.perf_counter() - a) * 1000
    times.append(ms)
    H, W = res.orig_shape
    objs = []
    for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        objs.append({"name": res.names[int(c)], "score": float(s),
                     "bbox": [b[0] / W * 1000, b[1] / H * 1000, b[2] / W * 1000, b[3] / H * 1000]})
    json.dump({"objects": objs}, open(os.path.join(out_dir, f"{iid}.json"), "w"))
    idx.write(json.dumps({"image_id": iid, "ms": ms, "error": None}) + "\n")
print(f"{weights}: {len(sample)} images, median {statistics.median(times):.1f} ms/img (end-to-end incl. decode)")
