"""Run a detector on every .png/.jpg in a folder; write <stem>.json next to a given out dir.

Usage: python run_dir.py mirage|yolo <in_dir> <out_dir> [yolo_weights]
Mirage output is saved raw (full describe). YOLO output uses Mirage's schema
(objects[name, score, bbox on 0-1000 per-axis xyxy]).
"""
import json
import os
import sys
import time

kind, in_dir, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg")))

if kind == "mirage":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mirage_safe import SafeMirage
    m = SafeMirage()

    def run(p):
        return m.describe(p)
else:
    from ultralytics import YOLO
    model = YOLO(sys.argv[4] if len(sys.argv) > 4 else "yolo26n.pt")

    def run(p):
        res = model.predict(p, verbose=False, device="cpu")[0]
        H, W = res.orig_shape
        return {"objects": [{"name": res.names[int(c)], "score": float(s),
                             "bbox": [b[0] / W * 1000, b[1] / H * 1000, b[2] / W * 1000, b[3] / H * 1000]}
                            for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]}

t0 = time.time()
for f in files:
    r = run(os.path.join(in_dir, f))
    json.dump(r, open(os.path.join(out_dir, os.path.splitext(f)[0] + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"{kind}: {len(files)} files in {time.time() - t0:.0f}s"
      + (f"; NaN-JSON sanitized {m.sanitized}/{m.calls}" if kind == "mirage" else ""))
