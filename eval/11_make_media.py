"""Build the README media from the saved outputs: comparison gallery, stability GIFs, probe grids,
charts, and two MP4s. No model is run here; every figure is drawn from results/ and data/.

Usage: python 11_make_media.py <repo_root>
Needs: pillow, numpy, matplotlib, imageio-ffmpeg (H.264 encoding).
Existing gallery/GIF/video files are kept (delete them to rebuild).
"""
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = sys.argv[1]
R = lambda *p: os.path.join(root, *p)  # noqa: E731
MEDIA = R("media")
for sub in ("gallery", "stability", "charts", "video"):
    os.makedirs(os.path.join(MEDIA, sub), exist_ok=True)

# ---------- palette (dataviz reference palette, validated: blue/orange/aqua) ----------
BLUE, ORANGE, AQUA = (42, 120, 214), (235, 104, 52), (27, 175, 122)
COLOR = {"gt": AQUA, "mirage": BLUE, "yolo": ORANGE}
TITLE = {"gt": "Ground truth (COCO)", "mirage": "Mirage", "yolo": "YOLO26n"}
DARK, INK, WHITE = (26, 26, 25), (11, 11, 11), (255, 255, 255)


def font(px, bold=False):
    for p in (["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"] if bold else
              ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]) + \
             ["/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "DejaVuSans%s.ttf" % ("-Bold" if bold else "")]:
        try:
            return ImageFont.truetype(p, px)
        except OSError:
            continue
    return ImageFont.load_default()


F_LABEL, F_TITLE, F_HEAD, F_SMALL = font(13), font(15, True), font(17, True), font(12)
GT = json.load(open(R("data", "coco_val2017_used.json"), encoding="utf-8"))["images"]


def coco_photo(iid):
    return R("data", "coco_val2017", f"{int(iid):012d}.jpg")


def objs_0_1000(path, W, H, with_mask=False):
    """Boxes from a Mirage/YOLO JSON (0-1000 scale) -> pixel boxes."""
    if not os.path.exists(path):
        return []
    out = []
    for o in json.load(open(path, encoding="utf-8")).get("objects", []):
        b = o["bbox"]
        s = o.get("score")
        lab = o.get("name", "?") + (f" {s:.2f}" if isinstance(s, (int, float)) else "")
        mask = None
        if with_mask and isinstance(o.get("mask"), list) and len(o["mask"]) >= 6:
            m = o["mask"]
            mask = [(m[i] / 1000 * W, m[i + 1] / 1000 * H) for i in range(0, len(m) - 1, 2)]
        out.append({"box": (b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H),
                    "label": lab, "mask": mask})
    return out


def gt_objs(iid):
    out = []
    for a in GT[str(iid)]["annotations"]:
        x, y, w, h = a["bbox_xywh"]
        out.append({"box": (x, y, x + w, y + h), "label": a["category"] + (" (crowd)" if a["iscrowd"] else ""),
                    "mask": None})
    return out


def panel(img, objs, kind, pw, ph, max_labels=14):
    """Letterbox `img` into pw x ph, draw masks/boxes/labels; returns (panel image, n objects)."""
    W, H = img.size
    s = min(pw / W, ph / H)
    nw, nh = int(W * s), int(H * s)
    ox, oy = (pw - nw) // 2, (ph - nh) // 2
    base = Image.new("RGB", (pw, ph), DARK)
    base.paste(img.resize((nw, nh), Image.BILINEAR), (ox, oy))
    col = COLOR[kind]
    over = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for o in objs:
        if o["mask"]:
            d.polygon([(ox + x * s, oy + y * s) for x, y in o["mask"]], fill=col + (70,))
    base = Image.alpha_composite(base.convert("RGBA"), over)
    d = ImageDraw.Draw(base)
    order = sorted(objs, key=lambda o: -(o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
    for o in order:
        x1, y1, x2, y2 = o["box"]
        d.rectangle([ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s], outline=col + (255,), width=2)
    for o in order[:max_labels]:
        x1, y1 = ox + o["box"][0] * s, oy + o["box"][1] * s
        t = o["label"][:24]
        tw = d.textlength(t, font=F_LABEL)
        ty = max(0, y1 - 17)
        d.rectangle([x1, ty, x1 + tw + 10, ty + 17], fill=(11, 11, 11, 200))
        d.rectangle([x1, ty, x1 + 3, ty + 17], fill=col + (255,))
        d.text((x1 + 7, ty + 1), t, fill=WHITE, font=F_LABEL)
    return base.convert("RGB"), len(objs)


def titled(p, kind, n, extra=""):
    bar = Image.new("RGB", (p.width, 28), INK)
    d = ImageDraw.Draw(bar)
    d.rectangle([8, 9, 18, 19], fill=COLOR[kind])
    d.text((26, 4), f"{TITLE[kind]} · {n} objects{extra}", fill=WHITE, font=F_TITLE)
    out = Image.new("RGB", (p.width, p.height + 28))
    out.paste(bar, (0, 0))
    out.paste(p, (0, 28))
    return out


def header(width, text, right=""):
    bar = Image.new("RGB", (width, 40), (252, 252, 251))
    d = ImageDraw.Draw(bar)
    d.text((12, 9), text, fill=INK, font=F_HEAD)
    if right:
        d.text((width - 12 - d.textlength(right, font=F_SMALL), 13), right, fill=(82, 81, 78), font=F_SMALL)
    return bar


def stack(rows):
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, sum(r.height for r in rows)), (252, 252, 251))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def hstack(imgs, gap=4):
    out = Image.new("RGB", (sum(i.width for i in imgs) + gap * (len(imgs) - 1), max(i.height for i in imgs)),
                    (252, 252, 251))
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def coco_frame(iid, head, right="", pw=512, ph=384):
    img = Image.open(coco_photo(iid)).convert("RGB")
    W, H = img.size
    panels = []
    for kind, objs in (("gt", gt_objs(iid)),
                       ("mirage", objs_0_1000(R("results", "coco_mirage", f"{iid}.json"), W, H, True)),
                       ("yolo", objs_0_1000(R("results", "coco_yolo26n", f"{iid}.json"), W, H))):
        p, n = panel(img, objs, kind, pw, ph)
        panels.append(titled(p, kind, n))
    body = hstack(panels, gap=0)
    return stack([header(body.width, head, right), body])


def writer(path, size, fps):
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=None, macro_block_size=1,
                                    output_params=["-crf", "28", "-preset", "medium", "-map_metadata", "-1",
                                                   "-fflags", "+bitexact", "-movflags", "+faststart"])
    w.send(None)
    return w


sample = json.load(open(R("results", "coco_mirage", "_sample_ids.json")))

# ---------- 1. gallery: seeded random sample + selected examples ----------
rand_ids = random.Random(2026).sample(sample, 8)
gallery = [(f"random_{k + 1:02d}_{iid}.jpg", iid, f"COCO val2017 #{iid} · random sample {k + 1}/8 (seed 2026)")
           for k, iid in enumerate(rand_ids)]
gallery.append((f"selected_{73946}.jpg", 73946, "COCO val2017 #73946 · selected example: adult labelled as furniture"))
for name, iid, head in gallery:
    out = os.path.join(MEDIA, "gallery", name)
    if not os.path.exists(out):
        coco_frame(iid, head, "boxes + Mirage masks drawn from results/").save(out, quality=86)
        print("gallery", name, flush=True)

# ---------- 2. video: all 300 COCO images ----------
vid = os.path.join(MEDIA, "video", "coco300_gt_mirage_yolo.mp4")
if not os.path.exists(vid):
    first = coco_frame(sample[0], "x")
    w = writer(vid, first.size, fps=2)
    for k, iid in enumerate(sample):
        fr = coco_frame(iid, f"COCO val2017 #{iid}", f"image {k + 1}/300 · still images, 1 s each")
        for _ in range(2):  # 1 s per image at 2 fps
            w.send(np.asarray(fr).tobytes())
    w.close()
    print("video", vid, flush=True)

# ---------- 3. stability GIFs (8 scenes) + MP4 ----------
stab_meta = json.load(open(R("results", "probes", "stab", "meta.json"), encoding="utf-8"))
VARIANTS = [("orig", "original"), ("pan2", "2% pan"), ("pan5", "5% pan"), ("dark75", "brightness ×0.75"),
            ("bright125", "brightness ×1.25"), ("blur", "Gaussian blur"), ("jpeg35", "JPEG quality 35"),
            ("lowres", "⅓ resolution, upscaled")]
scenes = sorted({f.split("__")[0] for f in os.listdir(R("results", "probes", "stab")) if f.endswith(".png")})


def stab_frame(scene, v, label):
    img = Image.open(R("results", "probes", "stab", f"{scene}__{v}.png")).convert("RGB")
    W, H = img.size
    ps = []
    for kind, sub in (("mirage", "probe_out_mirage"), ("yolo", "probe_out_yolo")):
        p, n = panel(img, objs_0_1000(R("results", sub, "stab", f"{scene}__{v}.json"), W, H, kind == "mirage"),
                     kind, 448, 336, max_labels=10)
        ps.append(titled(p, kind, n))
    body = hstack(ps, gap=0)
    return stack([header(body.width, f"Scene #{scene} · {label}", "same scene, small image changes")]
                 + [body])


svid = os.path.join(MEDIA, "video", "stability_8_scenes.mp4")
need_vid = not os.path.exists(svid)
sw = None
for scene in scenes:
    gif = os.path.join(MEDIA, "stability", f"{scene}.gif")
    frames = [stab_frame(scene, v, lab) for v, lab in VARIANTS] if (need_vid or not os.path.exists(gif)) else []
    if frames and not os.path.exists(gif):
        pal = [f.quantize(colors=160, method=Image.Quantize.MEDIANCUT) for f in frames]
        pal[0].save(gif, save_all=True, append_images=pal[1:], duration=1300, loop=0, optimize=True)
        print("gif", gif, flush=True)
    if need_vid:
        if sw is None:
            sw = writer(svid, frames[0].size, fps=2)
        for f in frames:
            for _ in range(3):  # 1.5 s per frame
                sw.send(np.asarray(f).tobytes())
if sw is not None:
    sw.close()
    print("video", svid, flush=True)


# ---------- 4. probe grids ----------
def wrap(d, text, f, width, max_lines):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = wd
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and len(" ".join(lines)) < len(text):
        lines[-1] = lines[-1][: max(0, len(lines[-1]) - 1)] + "…"
    return lines


def tile(img, lines, tw, th, cap_h, line_colors=None):
    t = Image.new("RGB", (tw, th + cap_h), (252, 252, 251))
    s = min(tw / img.width, th / img.height)
    im = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
    t.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, tw - 1, th - 1], outline=(195, 194, 183))
    y = th + 6
    for k, ln in enumerate(lines):
        d.text((6, y), ln, fill=(line_colors or {}).get(k, INK), font=F_SMALL)
        y += 16
    return t


def grid(tiles, cols, title, note):
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (cols * tw + (cols + 1) * 8, rows * th + (rows + 1) * 8), (252, 252, 251))
    for k, t in enumerate(tiles):
        g.paste(t, (8 + (k % cols) * (tw + 8), 8 + (k // cols) * (th + 8)))
    return stack([header(g.width, title, note), g])


SEC = (82, 81, 78)
# 4a. blank/noise frames
tiles = []
for name in ("white", "black", "gray", "gradient", "noise", "wall"):
    img = Image.open(R("results", "probes", "null", f"{name}.png")).convert("RGB")
    m = json.load(open(R("results", "probe_out_mirage", "null", f"{name}.json"), encoding="utf-8"))
    y = json.load(open(R("results", "probe_out_yolo", "null", f"{name}.json"), encoding="utf-8"))
    mo = ", ".join(o.get("name", "?") for o in m.get("objects", [])) or "no objects"
    mt = [t.get("string", "") for t in m.get("text", []) if t.get("string")]
    reason = m.get("reasoning", "")
    yo = ", ".join(f"{o['name']} {o['score']:.2f}" for o in y.get("objects", [])) or "nothing"
    d0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = [f"{name} frame", f"Mirage objects: {mo}"[:60]]
    if mt:
        lines.append(f"Mirage text: {' | '.join(mt)}"[:60])
    lines += wrap(d0, f"Mirage reasoning ({len(reason):,} chars): {reason}", F_SMALL, 300, 3)
    lines.append(f"YOLO26n: {yo}"[:60])
    tiles.append(tile(img, lines, 312, 200, 6 + 16 * 7, {0: INK, len(lines) - 1: SEC}))
grid(tiles, 3, "Blank and noise frames: what each model reports",
     "all 6 probes shown · results/probe_out_*/null").save(os.path.join(MEDIA, "null_frames.png"), optimize=True)
print("null grid", flush=True)

# 4b. synthetic signs: describe() vs extract()
reads = {r["file"]: r for r in json.load(open(R("results", "score_probes.json"), encoding="utf-8"))["ocr_reads"]}
ext = {r["file"]: r for r in json.load(open(R("results", "extract_ocr.json"), encoding="utf-8"))["rows"]}
meta = json.load(open(R("results", "probes", "ocr", "meta.json"), encoding="utf-8"))
tiles = []
for e in meta:
    gt = e["text"].upper()
    got = reads.get(e["file"], {}).get("read", [])
    best = next((s for s in got if s.upper() == gt), got[0] if got else "")
    ok = best.upper() == gt
    ex = str(ext.get(e["file"], {}).get("extract"))
    img = Image.open(R("results", "probes", "ocr", e["file"])).convert("RGB")
    lines = [f"{os.path.splitext(e['file'])[0]} · sign says \"{gt}\"",
             f"describe(): \"{best}\" ({'correct' if ok else 'wrong'})"[:44],
             f"extract() string: \"{ex}\" (wrong)"[:44]]
    tiles.append(tile(img, lines, 232, 174, 56, {0: INK, 1: (0, 99, 0) if ok else (208, 59, 59), 2: (208, 59, 59)}))
grid(tiles, 4, "Reading signs: describe() vs extract()",
     "all 32 probes · sign pasted on a COCO indoor photo").save(os.path.join(MEDIA, "ocr_signs.png"), optimize=True)
print("ocr grid", flush=True)

# 4c. extract() dog vs cat
rows = json.load(open(R("results", "extract_order.json"), encoding="utf-8"))["rows"]
tiles = []
for r in sorted(rows, key=lambda r: (r["gt"], r["id"])):
    img = Image.open(coco_photo(r["id"])).convert("RGB")
    ex_ok = r["dog_first"] == r["gt"]
    de_ok = r["describe_animals"] == r["gt"]
    lines = [f"#{r['id']} · photo of a {r['gt']}",
             f"extract(): {r['dog_first']} (P(dog) {r['p_dog_dog_first']:.2f})",
             f"describe() names: {r['describe_animals']}"]
    tiles.append(tile(img, lines, 196, 147, 56, {1: (0, 99, 0) if ex_ok else (208, 59, 59),
                                                 2: (0, 99, 0) if de_ok else (208, 59, 59)}))
grid(tiles, 6, "extract() with the vendor's own form {\"animal\": {\"enum\": [\"dog\", \"cat\"]}}",
     "all 30 photos · green = right, red = wrong").save(os.path.join(MEDIA, "extract_dog_cat.jpg"), quality=86)
print("dog/cat grid", flush=True)

# ---------- 5. charts (values read from results/*.json) ----------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_M, C_Y = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "text.color": "#0b0b0b", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "savefig.dpi": 150, "savefig.facecolor": "#fcfcfb"})
sm = json.load(open(R("results", "score_coco_mirage.json"), encoding="utf-8"))
sy = json.load(open(R("results", "score_coco_yolo26n.json"), encoding="utf-8"))


def grouped(fname, title, sub, cats, mv, yv, ylabel="%", ymax=100, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(cats))
    bw = 0.36
    for off, vals, col, lab in ((-bw / 2 - 0.01, mv, C_M, "Mirage"), (bw / 2 + 0.01, yv, C_Y, "YOLO26n")):
        bars = ax.bar(x + off, vals, bw, color=col, label=lab, linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.012, fmt.format(v),
                    ha="center", va="bottom", fontsize=9, color="#52514e")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, ymax * 1.06)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", length=0)
    fig.suptitle(title, x=0.012, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(sub, loc="left", fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2,
               bbox_to_anchor=(0.99, 0.995))
    fig.savefig(os.path.join(MEDIA, "charts", fname))
    plt.close(fig)


k = "class_agnostic_recall@0.5"
grouped("recall_by_size.png", "Objects found (class-agnostic recall @ IoU 0.5)",
        "300 COCO val2017 images, 2,287 objects · higher is better",
        ["all", "small", "medium", "large"], [sm[k][c] for c in ("all", "small", "medium", "large")],
        [sy[k][c] for c in ("all", "small", "medium", "large")])
pm, py = sm["person"], sy["person"]
grouped("people.png", "People", "671 person instances in 159 images · higher is better",
        ["instance recall", "instance precision", "large-person recall", "image-level recall"],
        [pm["instance_recall@0.5_%"], pm["instance_precision@0.5_%"], pm["recall_by_size_%"]["large"],
         pm["image_level_recall_%"]],
        [py["instance_recall@0.5_%"], py["instance_precision@0.5_%"], py["recall_by_size_%"]["large"],
         py["image_level_recall_%"]])
st = json.load(open(R("results", "score_probes.json"), encoding="utf-8"))["stability"]
vk = ["pan2", "pan5", "dark75", "bright125", "blur", "jpeg35", "lowres"]
vl = ["2% pan", "5% pan", "bright ×0.75", "bright ×1.25", "blur", "JPEG 35", "⅓ res"]
grouped("stability.png", "Boxes that survive a small image change",
        "same-label persistence vs the original image · 8 scenes · higher is better", vl,
        [st["mirage"][v]["same_label_persist_%"] for v in vk], [st["yolo"][v]["same_label_persist_%"] for v in vk])

lat = json.load(open(R("results", "options_latency.json"), encoding="utf-8"))["modes"]
items = [(k2, v["median_ms"], C_M) for k2, v in lat.items()]
items.append(("YOLO26n (COCO run, end to end)", sy["latency_ms"]["median"], C_Y))
items.insert(len(items) - 1, ("Mirage (COCO run, end to end)", sm["latency_ms"]["median"], C_M))
fig, ax = plt.subplots(figsize=(8, 4.6))
yy = np.arange(len(items))[::-1]
bars = ax.barh(yy, [v for _, v, _ in items], 0.6, color=[c for *_, c in items], linewidth=0)
for b, (_, v, _) in zip(bars, items):
    ax.text(b.get_width() + 25, b.get_y() + b.get_height() / 2, f"{v:,.0f} ms", va="center", fontsize=9,
            color="#52514e")
ax.set_yticks(yy, [n for n, _, _ in items], fontsize=9)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.tick_params(axis="both", length=0)
ax.set_xlim(0, max(v for _, v, _ in items) * 1.15)
ax.set_xlabel("median ms per image (lower is better)")
fig.suptitle("Speed on an Intel i5-11320H laptop CPU", x=0.012, ha="left", fontsize=14, fontweight="bold")
ax.set_title("Mirage (blue) by option mode vs YOLO26n (orange) · warm medians", loc="left", fontsize=10,
             color="#52514e")
fig.tight_layout()
fig.savefig(os.path.join(MEDIA, "charts", "latency.png"))
plt.close(fig)

cal = sm["score_calibration (bin: n_preds, %matching_any_GT@0.5)"]
bins = [b for b in cal if b != "0.0-0.1"]
fig, ax = plt.subplots(figsize=(8, 3.8))
bars = ax.bar(bins, [cal[b][1] for b in bins], 0.6, color=C_M, linewidth=0)
for b, name in zip(bars, bins):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{cal[name][1]:.0f}%\n(n={cal[name][0]})",
            ha="center", va="bottom", fontsize=8, color="#52514e")
ax.set_ylim(0, 115)
ax.set_ylabel("% of boxes matching a GT object")
ax.set_xlabel("Mirage box score")
ax.tick_params(axis="both", length=0)
fig.suptitle("Is Mirage's score meaningful?", x=0.012, ha="left", fontsize=14, fontweight="bold")
ax.set_title("share of Mirage boxes in each score bin that overlap a real object (IoU ≥ 0.5)", loc="left",
             fontsize=10, color="#52514e")
fig.tight_layout()
fig.savefig(os.path.join(MEDIA, "charts", "score_calibration.png"))
plt.close(fig)
print("charts done", flush=True)
