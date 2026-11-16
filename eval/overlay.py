"""Draw Mirage boxes/masks (0-1000 per-axis scale, assumed xyxy) next to COCO GT.

Usage: python overlay.py <image> <mirage.json> <out.png> [coco_instances.json]
Left panel = Mirage (boxes + mask polygons), right panel = COCO ground truth.
"""
import json
import os
import sys

from PIL import Image, ImageDraw

img_path, res_path, out_path = sys.argv[1:4]
ann_path = sys.argv[4] if len(sys.argv) > 4 else None

im = Image.open(img_path).convert("RGB")
W, H = im.size
r = json.load(open(res_path, encoding="utf-8"))


def sx(v):
    return v / 1000.0 * W


def sy(v):
    return v / 1000.0 * H


left = im.copy()
d = ImageDraw.Draw(left, "RGBA")
for o in r.get("objects", []):
    x1, y1, x2, y2 = o["bbox"]
    m = o.get("mask")
    if m and isinstance(m, list) and len(m) >= 6 and isinstance(m[0], (int, float)):
        pts = [(sx(m[i]), sy(m[i + 1])) for i in range(0, len(m) - 1, 2)]
        d.polygon(pts, fill=(0, 200, 255, 70), outline=(0, 200, 255, 255))
    d.rectangle([sx(x1), sy(y1), sx(x2), sy(y2)], outline=(255, 40, 40, 255), width=2)
    sc = o.get("score")
    d.text((sx(x1) + 2, sy(y1) + 1), f"{o['name']} {sc:.2f}" if sc is not None else f"{o['name']} NaN",
           fill=(255, 255, 0, 255))
for t in r.get("text", []) or []:
    b = t.get("bbox")
    if b:
        d.rectangle([sx(b[0]), sy(b[1]), sx(b[2]), sy(b[3])], outline=(0, 255, 0, 255), width=2)
        d.text((sx(b[0]), sy(b[3]) + 1), str(t.get("text", t.get("content", ""))), fill=(0, 255, 0, 255))

right = im.copy()
if ann_path:
    coco = json.load(open(ann_path, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_id = int(os.path.splitext(os.path.basename(img_path))[0])
    d2 = ImageDraw.Draw(right, "RGBA")
    for a in coco["annotations"]:
        if a["image_id"] != img_id:
            continue
        x, y, w, h = a["bbox"]
        d2.rectangle([x, y, x + w, y + h], outline=(40, 255, 40, 255), width=2)
        d2.text((x + 2, y + 1), cats[a["category_id"]], fill=(255, 255, 0, 255))

canvas = Image.new("RGB", (W * 2 + 10, H), (255, 255, 255))
canvas.paste(left, (0, 0))
canvas.paste(right, (W + 10, 0))
scale = 1.6 if W < 700 else 1.0
canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)))
canvas.save(out_path)
print("saved", out_path, canvas.size)
