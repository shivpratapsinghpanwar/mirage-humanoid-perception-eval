"""Score detector outputs against COCO val2017 GT (humanoid-relevant metrics, not mAP).

Usage: python 05_coco_score.py <instances_val2017.json> <pred_dir> [--names]
pred_dir holds <image_id>.json with {"objects":[{"name","bbox":[x1,y1,x2,y2] on 0-1000 per-axis
scale,"score"}]} and _sample_ids.json. Prints a JSON summary; --names dumps the names a detector
used for boxes that overlap GT persons (to audit the person-synonym list).
"""
import collections
import json
import os
import re
import sys

ann_path, pred_dir = sys.argv[1], sys.argv[2]
dump_names = "--names" in sys.argv

coco = json.load(open(ann_path, encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {i["id"]: i for i in coco["images"]}
gt_by_img = collections.defaultdict(list)
for a in coco["annotations"]:
    if not a.get("iscrowd"):
        gt_by_img[a["image_id"]].append(a)
sample = json.load(open(os.path.join(pred_dir, "_sample_ids.json")))

# Names treated as "person". Fixed list, audited with --names; reported alongside results.
PERSON = {"person", "man", "woman", "boy", "girl", "child", "kid", "baby", "people", "human",
          "player", "tennis player", "baseball player", "soccer player", "skier", "snowboarder",
          "surfer", "skateboarder", "rider", "cyclist", "batter", "catcher", "umpire", "pitcher",
          "pedestrian", "lady", "guy", "athlete", "spectator", "chef", "toddler", "worker", "officer",
          "police officer", "student", "bride", "groom", "biker", "motorcyclist", "referee", "goalkeeper"}
# Small alias table for lenient label agreement (COCO name -> accepted extra names).
ALIAS = {"tv": {"television", "monitor", "screen", "tv"}, "dining table": {"table", "dining table", "desk"},
         "couch": {"sofa", "couch"}, "cell phone": {"phone", "cell phone", "smartphone", "mobile phone"},
         "potted plant": {"plant", "potted plant", "houseplant", "flower pot"},
         "motorcycle": {"motorcycle", "motorbike"}, "airplane": {"airplane", "plane", "aircraft", "jet"},
         "laptop": {"laptop", "computer"}, "sports ball": {"ball", "sports ball", "tennis ball", "soccer ball",
                                                             "baseball", "football", "basketball"},
         "dog": {"dog", "puppy"}, "cat": {"cat", "kitten"}, "bicycle": {"bicycle", "bike"},
         "car": {"car", "taxi", "vehicle"}, "remote": {"remote", "remote control"},
         "handbag": {"handbag", "purse", "bag"}, "backpack": {"backpack", "bag"},
         "wine glass": {"wine glass", "glass"}, "cup": {"cup", "mug", "glass"},
         "hair drier": {"hair dryer", "hair drier"}, "teddy bear": {"teddy bear", "stuffed animal"}}
HOUSEHOLD = {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
             "orange", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
             "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
             "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
             "backpack", "handbag", "umbrella", "suitcase"}


def norm(s):
    s = re.sub(r"[^a-z ]", "", (s or "").lower()).strip()
    return s[:-1] if s.endswith("s") and not s.endswith("ss") and len(s) > 3 else s


PERSON |= {"men", "women", "children", "kids", "players", "girls", "boys"}


def is_person(name):
    """Person if the full name or its head noun (last word) is a person noun.
    e.g. 'young girl' -> girl, 'male tennis player' -> player. 'hand', 'crowd' are not persons."""
    n = re.sub(r"[^a-z ]", " ", (name or "").lower()).split()
    if not n:
        return False
    full = " ".join(n)
    return full in PERSON or n[-1] in PERSON or norm(n[-1]) in PERSON


def label_ok(pred_name, gt_name):
    p = (pred_name or "").lower().strip()
    if gt_name == "person":
        return is_person(p)
    ok = {gt_name} | ALIAS.get(gt_name, set())
    return p in ok or norm(p) in {norm(x) for x in ok}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(preds, gts, thr=0.5):
    """Greedy one-to-one matching by IoU. Returns list of (pi, gi, iou)."""
    pairs = [(iou(p["box"], g["box"]), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)]
    pairs = sorted((x for x in pairs if x[0] >= thr), reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append((pi, gi, v))
    return out


def size_bucket(area):
    return "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large")


S = collections.Counter()
calib = collections.defaultdict(lambda: [0, 0])
names_on_person = collections.Counter()
label_pairs = collections.Counter()
lat = []
n_imgs = 0
for iid in sample:
    pf = os.path.join(pred_dir, f"{iid}.json")
    if not os.path.exists(pf):
        continue
    n_imgs += 1
    r = json.load(open(pf, encoding="utf-8"))
    W, H = imgs[iid]["width"], imgs[iid]["height"]
    preds = []
    for o in r.get("objects", []):
        b = o.get("bbox")
        if not b or len(b) != 4:
            continue
        preds.append({"name": o.get("name", ""), "score": o.get("score") or 0.0,
                      "box": [b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H]})
    gts = []
    for a in gt_by_img[iid]:
        x, y, w, h = a["bbox"]
        gts.append({"name": cats[a["category_id"]], "box": [x, y, x + w, y + h], "area": a["area"]})
    S["preds"] += len(preds)
    S["gts"] += len(gts)

    # 1) class-agnostic recall, by size and by group
    m = match(preds, gts)
    matched_g = {gi: pi for pi, gi, _ in m}
    matched_p = {pi for pi, _, _ in m}
    for gi, g in enumerate(gts):
        sb = size_bucket(g["area"])
        grp = "person" if g["name"] == "person" else ("household" if g["name"] in HOUSEHOLD else "other")
        for key in ("all", sb, grp):
            S[f"gt_{key}"] += 1
            S[f"hit_{key}"] += gi in matched_g
        if gi in matched_g:
            pn = preds[matched_g[gi]]["name"]
            S["lab_n"] += 1
            S["lab_ok"] += label_ok(pn, g["name"])
            S[f"lab_n_{grp}"] += 1
            S[f"lab_ok_{grp}"] += label_ok(pn, g["name"])
            if not label_ok(pn, g["name"]):
                label_pairs[(g["name"], pn.lower())] += 1
            if g["name"] == "person":
                names_on_person[pn.lower()] += 1
    for pi, p in enumerate(preds):
        b = min(int((p["score"] or 0) * 10), 9)
        calib[b][0] += 1
        calib[b][1] += pi in matched_p

    # 2) person detection (label-aware)
    gp = [g for g in gts if g["name"] == "person"]
    pp = [p for p in preds if is_person(p["name"])]
    mp = match(pp, gp)
    S["person_gt"] += len(gp)
    S["person_pred"] += len(pp)
    S["person_tp"] += len(mp)
    hit_gp = {gi for _, gi, _ in mp}
    for gi, g in enumerate(gp):
        sb = size_bucket(g["area"])
        S[f"person_gt_{sb}"] += 1
        S[f"person_tp_{sb}"] += gi in hit_gp
    # image-level presence
    if gp:
        S["img_with_person"] += 1
        S["img_person_detected"] += bool(pp)
        big = [g for g in gp if g["area"] >= 96 ** 2]
        if big:
            S["img_with_large_person"] += 1
            S["img_large_person_detected"] += bool(pp)
    else:
        S["img_without_person"] += 1
        S["img_person_false_alarm"] += bool(pp)

idx = os.path.join(pred_dir, "_index.jsonl")
if os.path.exists(idx):
    rows = [json.loads(x) for x in open(idx, encoding="utf-8")]
    lat = sorted(r["ms"] for r in rows if r.get("ms") and not r.get("error"))
    S["errors"] = sum(1 for r in rows if r.get("error"))


def pct(a, b):
    return round(100.0 * S[a] / S[b], 1) if S[b] else None


summary = {
    "images": n_imgs, "gt_boxes": S["gts"], "pred_boxes": S["preds"],
    "preds_per_image": round(S["preds"] / max(n_imgs, 1), 2),
    "class_agnostic_recall@0.5": {k: pct(f"hit_{k}", f"gt_{k}") for k in
                                  ("all", "small", "medium", "large", "person", "household", "other")},
    "gt_counts": {k: S[f"gt_{k}"] for k in ("all", "small", "medium", "large", "person", "household", "other")},
    "label_agreement_on_matched_%": {"all": pct("lab_ok", "lab_n"), "person": pct("lab_ok_person", "lab_n_person"),
                                     "household": pct("lab_ok_household", "lab_n_household"),
                                     "other": pct("lab_ok_other", "lab_n_other")},
    "person": {
        "instance_recall@0.5_%": pct("person_tp", "person_gt"),
        "instance_precision@0.5_%": pct("person_tp", "person_pred"),
        "recall_by_size_%": {k: pct(f"person_tp_{k}", f"person_gt_{k}") for k in ("small", "medium", "large")},
        "gt_by_size": {k: S[f"person_gt_{k}"] for k in ("small", "medium", "large")},
        "image_level_recall_%": pct("img_person_detected", "img_with_person"),
        "image_level_recall_large_person_%": pct("img_large_person_detected", "img_with_large_person"),
        "image_false_alarm_rate_%": pct("img_person_false_alarm", "img_without_person"),
        "images_with_person": S["img_with_person"], "images_without_person": S["img_without_person"],
    },
    "score_calibration (bin: n_preds, %matching_any_GT@0.5)": {
        f"{b / 10:.1f}-{(b + 1) / 10:.1f}": [calib[b][0], round(100 * calib[b][1] / calib[b][0], 1)]
        for b in sorted(calib) if calib[b][0]},
    "latency_ms": {"median": lat[len(lat) // 2] if lat else None, "p90": lat[int(len(lat) * 0.9)] if lat else None},
    "run_errors": S["errors"],
    "top_label_confusions(gt->pred)": [[g, p, c] for (g, p), c in label_pairs.most_common(25)],
}
if dump_names:
    summary["names_used_on_gt_persons"] = names_on_person.most_common(40)
print(json.dumps(summary, indent=1))
