"""Mask quality: IoU between Mirage polygon masks and COCO GT segmentations, for box-matched pairs.

Usage: python 07_mask_iou.py <instances_val2017.json> <mirage_coco_dir>
"""
import collections
import json
import os
import statistics
import sys

import numpy as np
from PIL import Image, ImageDraw

ann_path, pred_dir = sys.argv[1:3]
coco = json.load(open(ann_path, encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {i["id"]: i for i in coco["images"]}
gt_by = collections.defaultdict(list)
for a in coco["annotations"]:
    if not a.get("iscrowd") and isinstance(a["segmentation"], list):
        gt_by[a["image_id"]].append(a)
sample = json.load(open(os.path.join(pred_dir, "_sample_ids.json")))
HOUSE = {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "chair", "couch", "potted plant", "bed",
         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
         "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "toothbrush",
         "backpack", "handbag", "umbrella", "suitcase", "banana", "apple", "orange", "sandwich"}


def raster(polys, w, h):
    im = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        if len(p) >= 6:
            d.polygon([(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)], fill=1)
    return np.asarray(im, dtype=bool)


def biou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0


res = collections.defaultdict(list)
box_iou_of_matched = collections.defaultdict(list)
no_mask = 0
for iid in sample:
    pf = os.path.join(pred_dir, f"{iid}.json")
    if not os.path.exists(pf):
        continue
    r = json.load(open(pf, encoding="utf-8"))
    W, H = imgs[iid]["width"], imgs[iid]["height"]
    preds = []
    for o in r.get("objects", []):
        b = o["bbox"]
        preds.append((o, [b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H]))
    gts = [(a, [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]) for a in gt_by[iid]]
    pairs = sorted(((biou(pb, gb), i, j) for i, (_, pb) in enumerate(preds) for j, (_, gb) in enumerate(gts)), reverse=True)
    up, ug = set(), set()
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i in up or j in ug:
            continue
        up.add(i)
        ug.add(j)
        o, a = preds[i][0], gts[j][0]
        m = o.get("mask")
        if not m:
            no_mask += 1
            continue
        polys = m if isinstance(m[0], list) else [m]
        polys = [[(p[k] / 1000 * W if k % 2 == 0 else p[k] / 1000 * H) for k in range(len(p))] for p in polys]
        pm = raster(polys, W, H)
        gm = raster(a["segmentation"], W, H)
        inter = (pm & gm).sum()
        union = (pm | gm).sum()
        if union == 0:
            continue
        name = cats[a["category_id"]]
        grp = "person" if name == "person" else ("household" if name in HOUSE else "other")
        size = "small" if a["area"] < 32 ** 2 else ("medium" if a["area"] < 96 ** 2 else "large")
        res[grp].append(inter / union)
        res[size].append(inter / union)
        res["all"].append(inter / union)
        box_iou_of_matched["all"].append(v)

out = {k: {"n": len(v), "mean_mask_IoU": round(statistics.mean(v), 3), "median": round(statistics.median(v), 3),
           "share_>=0.75_%": round(100 * sum(x >= 0.75 for x in v) / len(v), 1)} for k, v in res.items() if v}
out["matched_without_mask"] = no_mask
out["mean_box_IoU_of_matched"] = round(statistics.mean(box_iou_of_matched["all"]), 3)
print(json.dumps(out, indent=1))
