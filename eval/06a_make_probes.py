"""Generate humanoid-relevant probe images (stored outside eval/, under results/probes/).

Usage: python 06a_make_probes.py <val2017_dir> <instances_val2017.json> <mirage_coco_dir> <probe_dir>
Creates:
  null/      blank, gray, noise, gradient frames                 (hallucination)
  stab/      originals + perturbations of 8 person+household images (+ meta.json for box transforms)
  ocr/       synthetic signs pasted on indoor scenes at 3 text sizes + rotated  (+ meta.json GT strings)
"""
import io
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

img_dir, ann_path, mirage_dir, probe_dir = sys.argv[1:5]
rng = np.random.default_rng(0)
for sub in ("null", "stab", "ocr"):
    os.makedirs(os.path.join(probe_dir, sub), exist_ok=True)

# ---------- A. null / hallucination frames ----------
W, H = 640, 480
Image.new("RGB", (W, H), (0, 0, 0)).save(os.path.join(probe_dir, "null", "black.png"))
Image.new("RGB", (W, H), (255, 255, 255)).save(os.path.join(probe_dir, "null", "white.png"))
Image.new("RGB", (W, H), (128, 128, 128)).save(os.path.join(probe_dir, "null", "gray.png"))
Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(os.path.join(probe_dir, "null", "noise.png"))
g = np.tile(np.linspace(30, 220, W, dtype=np.uint8)[None, :, None], (H, 1, 3))
Image.fromarray(g).save(os.path.join(probe_dir, "null", "gradient.png"))
# beige wall with mild texture (what a robot sees when facing a wall)
wall = np.clip(np.full((H, W, 3), (205, 190, 165), np.float32) + rng.normal(0, 6, (H, W, 1)), 0, 255).astype(np.uint8)
Image.fromarray(wall).filter(ImageFilter.GaussianBlur(1)).save(os.path.join(probe_dir, "null", "wall.png"))

# ---------- B. stability set ----------
coco = json.load(open(ann_path, encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
household = {"cup", "bottle", "bowl", "chair", "couch", "dining table", "tv", "laptop", "book", "vase",
             "remote", "cell phone", "refrigerator", "microwave", "oven", "sink", "potted plant", "bed"}
per_img = {}
for a in coco["annotations"]:
    if a.get("iscrowd"):
        continue
    d = per_img.setdefault(a["image_id"], {"person_large": 0, "house": 0})
    n = cats[a["category_id"]]
    if n == "person" and a["area"] >= 96 ** 2:
        d["person_large"] += 1
    if n in household and a["area"] >= 32 ** 2:
        d["house"] += 1
sample = set(json.load(open(os.path.join(mirage_dir, "_sample_ids.json"))))
# pick from images NOT in the main sample, to keep sets independent
cands = sorted(i for i, d in per_img.items() if d["person_large"] >= 1 and d["house"] >= 2 and i not in sample)
random.Random(7).shuffle(cands)
stab_ids = cands[:8]
meta = {"images": [], "variants": {}}


def jpeg(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def pan(im, f):
    w, h = im.size
    dx, dy = int(w * f), int(h * f * 0.5)
    return im.crop((dx, dy, w, h)).resize((w, h), Image.BICUBIC), (dx, dy)


for iid in stab_ids:
    im = Image.open(os.path.join(img_dir, f"{iid:012d}.jpg")).convert("RGB")
    w, h = im.size
    base = f"{iid}"
    meta["images"].append({"id": iid, "w": w, "h": h})
    im.save(os.path.join(probe_dir, "stab", f"{base}__orig.png"))
    variants = {}
    for f in (0.02, 0.05):
        v, (dx, dy) = pan(im, f)
        variants[f"pan{int(f * 100)}"] = {"crop": [dx, dy, w, h]}
        v.save(os.path.join(probe_dir, "stab", f"{base}__pan{int(f * 100)}.png"))
    for name, fn in [("dark75", lambda x: Image.fromarray(np.clip(np.asarray(x, np.float32) * 0.75, 0, 255).astype(np.uint8))),
                     ("bright125", lambda x: Image.fromarray(np.clip(np.asarray(x, np.float32) * 1.25, 0, 255).astype(np.uint8))),
                     ("blur", lambda x: x.filter(ImageFilter.GaussianBlur(1.5))),
                     ("jpeg35", lambda x: jpeg(x, 35)),
                     ("lowres", lambda x: x.resize((w // 3, h // 3), Image.BILINEAR).resize((w, h), Image.BILINEAR))]:
        fn(im).save(os.path.join(probe_dir, "stab", f"{base}__{name}.png"))
        variants[name] = {"crop": [0, 0, w, h]}
    meta["variants"][base] = variants
json.dump(meta, open(os.path.join(probe_dir, "stab", "meta.json"), "w"), indent=1)

# ---------- C. OCR signs ----------
texts = ["EXIT", "PUSH", "ROOM 204", "WET FLOOR", "DANGER", "KITCHEN", "LIFT 3", "NO ENTRY"]
# Published probes used Arial Bold (Windows). Other systems fall back to DejaVu Sans Bold, so regenerated
# sign images differ slightly; set PROBE_FONT to a .ttf path to choose the font explicitly.
font_path = next((p for p in [os.environ.get("PROBE_FONT"), "C:/Windows/Fonts/arialbd.ttf",
                              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                              "/Library/Fonts/Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
                  if p and (os.path.exists(p) or not os.path.isabs(p))), "DejaVuSans-Bold.ttf")
indoor = [139, 785, 1000]  # backgrounds (indoor-ish COCO scenes)
ocr_meta = []
for k, t in enumerate(texts):
    bg_id = indoor[k % len(indoor)]
    bg = Image.open(os.path.join(img_dir, f"{bg_id:012d}.jpg")).convert("RGB")
    for size_name, px in [("large", 60), ("medium", 26), ("small", 13)]:
        for rot in ([0, 15] if size_name == "large" else [0]):
            font = ImageFont.truetype(font_path, px)
            tw, th = font.getbbox(t)[2:]
            sign = Image.new("RGB", (tw + px, th + px), (245, 245, 245))
            ImageDraw.Draw(sign).text((px // 2, px // 2 - font.getbbox(t)[1] // 2), t, fill=(10, 10, 10), font=font)
            if rot:
                sign = sign.rotate(rot, expand=True, fillcolor=(245, 245, 245))
            img = bg.copy()
            x = (img.width - sign.width) // 2
            y = int(img.height * 0.25)
            img.paste(sign, (x, y))
            fn = f"{k:02d}_{size_name}{'_rot' + str(rot) if rot else ''}.png"
            img.save(os.path.join(probe_dir, "ocr", fn))
            ocr_meta.append({"file": fn, "text": t, "size": size_name, "text_px": px, "rot": rot})
json.dump(ocr_meta, open(os.path.join(probe_dir, "ocr", "meta.json"), "w"), indent=1)
print("stab ids:", stab_ids)
print("null:", len(os.listdir(os.path.join(probe_dir, "null"))), "stab:", len(os.listdir(os.path.join(probe_dir, "stab"))) - 1,
      "ocr:", len(ocr_meta))
