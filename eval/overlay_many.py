"""Like overlay.py but for several images with one annotation load; marks only person/chair boxes.

Usage: python overlay_many.py <val2017_dir> <instances_val2017.json> <mirage_dir> <out_dir> <id> [<id> ...]
Left = Mirage boxes (red) labelled; right = COCO GT (green) labelled.
"""
import json
import os
import sys

from PIL import Image, ImageDraw

img_dir, ann_path, mdir, out_dir = sys.argv[1:5]
ids = [int(x) for x in sys.argv[5:]]
coco = json.load(open(ann_path, encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
gt = {i: [a for a in coco["annotations"] if a["image_id"] == i] for i in ids}
del coco
os.makedirs(out_dir, exist_ok=True)
for iid in ids:
    im = Image.open(os.path.join(img_dir, f"{iid:012d}.jpg")).convert("RGB")
    W, H = im.size
    r = json.load(open(os.path.join(mdir, f"{iid}.json"), encoding="utf-8"))
    L, R = im.copy(), im.copy()
    d, d2 = ImageDraw.Draw(L), ImageDraw.Draw(R)
    for o in r.get("objects", []):
        x1, y1, x2, y2 = [o["bbox"][0] / 1000 * W, o["bbox"][1] / 1000 * H, o["bbox"][2] / 1000 * W, o["bbox"][3] / 1000 * H]
        d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=2)
        d.text((x1 + 2, y1 + 1), o["name"], fill=(255, 255, 0))
    for a in gt[iid]:
        x, y, w, h = a["bbox"]
        d2.rectangle([x, y, x + w, y + h], outline=(40, 255, 40), width=2)
        d2.text((x + 2, y + 1), cats[a["category_id"]], fill=(255, 255, 0))
    c = Image.new("RGB", (W * 2 + 10, H), (255, 255, 255))
    c.paste(L, (0, 0))
    c.paste(R, (W + 10, 0))
    c.save(os.path.join(out_dir, f"audit_{iid}.png"))
    print("saved", iid)
