"""Print a seeded random sample of (GT name -> Mirage name) pairs for box-matched objects whose
names disagree under the automatic rule, for manual audit (synonym vs wrong).

Usage: python 09_label_audit.py <instances_val2017.json> <mirage_coco_dir> [n=80]
"""
import importlib.util
import json
import os
import random
import sys

ann, d = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 80
spec = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(__file__), "05_coco_score.py"))
# reuse helpers without running the scorer's main block: exec only the definitions we need
src = open(spec.origin, encoding="utf-8").read().split("S = collections.Counter()")[0]
sys.argv = [sys.argv[0], ann, d]
g = {}
exec(compile(src, spec.origin, "exec"), g)
coco, cats, imgs, gt_by_img, sample = g["coco"], g["cats"], g["imgs"], g["gt_by_img"], g["sample"]
pairs = []
for iid in sample:
    r = json.load(open(os.path.join(d, f"{iid}.json"), encoding="utf-8"))
    W, H = imgs[iid]["width"], imgs[iid]["height"]
    preds = [{"name": o.get("name", ""), "box": [o["bbox"][0] / 1000 * W, o["bbox"][1] / 1000 * H,
                                                  o["bbox"][2] / 1000 * W, o["bbox"][3] / 1000 * H]}
             for o in r.get("objects", [])]
    gts = [{"name": cats[a["category_id"]], "box": [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                                    a["bbox"][1] + a["bbox"][3]]} for a in gt_by_img[iid]]
    for pi, gi, _ in g["match"](preds, gts):
        if not g["label_ok"](preds[pi]["name"], gts[gi]["name"]):
            pairs.append((iid, gts[gi]["name"], preds[pi]["name"]))
random.Random(11).shuffle(pairs)
print("total mismatched matched-pairs:", len(pairs))
for k, (iid, gn, pn) in enumerate(pairs[:n]):
    print(f"{k:02d}  {iid:>7}  GT={gn:<14} Mirage={pn}")
