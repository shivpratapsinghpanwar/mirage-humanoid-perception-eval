"""Copy the COCO val2017 photos used in this evaluation into data/coco_val2017/ and write their
ground truth and per-photo licence/attribution to data/coco_val2017_used.json.

Photos bundled: the 300-image sample, the 30 dog/cat photos of the extract() test, and the three
backgrounds of the synthetic signs. Ground truth = non-crowd and crowd boxes with category names.

Usage: python 10_bundle_coco.py <val2017_dir> <instances_val2017.json> <repo_root>
"""
import gc
import json
import os
import shutil
import sys

img_dir, ann_path, root = sys.argv[1:4]
res = os.path.join(root, "results")
ids = set(json.load(open(os.path.join(res, "coco_mirage", "_sample_ids.json"))))
ids |= {r["id"] for r in json.load(open(os.path.join(res, "extract_fairness.json"), encoding="utf-8"))["animal"]}
ids |= {139, 785, 1000}  # sign backgrounds (eval/06a_make_probes.py)

coco = json.load(open(ann_path, encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
licenses = {l["id"]: {"name": l["name"], "url": l["url"]} for l in coco["licenses"]}
images = {i["id"]: {"file_name": i["file_name"], "width": i["width"], "height": i["height"],
                    "license": licenses[i["license"]]["name"], "license_url": licenses[i["license"]]["url"],
                    "flickr_url": i.get("flickr_url"), "coco_url": i.get("coco_url"), "annotations": []}
          for i in coco["images"] if i["id"] in ids}
for a in coco["annotations"]:
    if a["image_id"] in images:
        images[a["image_id"]]["annotations"].append(
            {"category": cats[a["category_id"]], "bbox_xywh": [round(v, 2) for v in a["bbox"]],
             "area": round(a["area"], 1), "iscrowd": a["iscrowd"]})
del coco
gc.collect()

out_dir = os.path.join(root, "data", "coco_val2017")
os.makedirs(out_dir, exist_ok=True)
for iid, meta in images.items():
    dst = os.path.join(out_dir, meta["file_name"])
    if not os.path.exists(dst):
        shutil.copyfile(os.path.join(img_dir, meta["file_name"]), dst)
doc = {"source": "COCO 2017 val (https://cocodataset.org). Annotations: CC BY 4.0, COCO Consortium. "
                 "Photos: Flickr, each under the licence listed here.",
       "images": {str(k): images[k] for k in sorted(images)}}
json.dump(doc, open(os.path.join(root, "data", "coco_val2017_used.json"), "w", encoding="utf-8"), indent=1)
print(f"{len(images)} photos bundled; licences:",
      {n: sum(1 for m in images.values() if m["license"] == n) for n in sorted({m["license"] for m in images.values()})})
