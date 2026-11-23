"""Score probe runs: hallucination on null frames, stability under perturbations, OCR.

Usage: python 06c_score_probes.py <probe_dir> <mirage_out_root> [yolo_out_root]
       python 06c_score_probes.py <probe_dir> <tag>=<out_root> [<tag>=<out_root> ...]
  out roots contain null/, stab/, ocr/ subfolders of <stem>.json. OCR is scored for the tag "mirage"
  (describe() text output) only. --min-score T ignores boxes scored below T (null and stability).
"""
import collections
import json
import os
import sys

probe_dir = sys.argv[1]
MIN = float(sys.argv[sys.argv.index("--min-score") + 1]) if "--min-score" in sys.argv else None
_args = [x for k, x in enumerate(sys.argv[2:], 2) if not x.startswith("--") and sys.argv[k - 1] != "--min-score"]
ROOTS = []
for i, x in enumerate(_args):
    tag, root = x.split("=", 1) if "=" in x else (("mirage", "yolo")[i] if i < 2 else f"model{i}", x)
    ROOTS.append((tag, root))
mroot = dict(ROOTS).get("mirage")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PN = {"person", "man", "woman", "boy", "girl", "child", "kid", "baby", "people", "player", "skier", "surfer",
       "skateboarder", "snowboarder", "rider", "cyclist", "batter", "catcher", "umpire", "pedestrian", "lady",
       "guy", "human", "athlete", "toddler", "chef", "worker", "men", "women", "children", "players"}


class _PersonSet:
    def __contains__(self, name):
        t = "".join(ch if ch.isalpha() or ch == " " else " " for ch in (name or "").lower()).split()
        return bool(t) and (" ".join(t) in _PN or t[-1] in _PN)


PERSONISH = _PersonSet()


def load(root, sub, stem):
    p = os.path.join(root, sub, stem + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(A, B, thr=0.5):
    pairs = sorted(((iou(a["box"], b["box"]), i, j) for i, a in enumerate(A) for j, b in enumerate(B)), reverse=True)
    ua, ub, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        out.append((i, j))
    return out


def keep(o):
    return MIN is None or not isinstance(o.get("score"), (int, float)) or o["score"] >= MIN


def boxes(r, w, h):
    out = []
    for o in (r or {}).get("objects", []):
        if not keep(o):
            continue
        b = o["bbox"]
        out.append({"name": o.get("name", "").lower(), "box": [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]})
    return out


report = {}

# ---- null frames ----
nulls = {}
for f in sorted(os.listdir(os.path.join(probe_dir, "null"))):
    stem = os.path.splitext(f)[0]
    row = {}
    for tag, root in ROOTS:
        r = load(root, "null", stem)
        if r is None:
            continue
        row[tag] = {"objects": [f"{o.get('name')} {o.get('score') or 0:.2f}" for o in r.get("objects", []) if keep(o)]}
        if "text" in r:
            row[tag]["text"] = [t.get("string") for t in r.get("text", [])]
            row[tag]["scene"] = r.get("scene")
            row[tag]["reasoning"] = r.get("reasoning")
    nulls[stem] = row
report["null_frames"] = nulls

# ---- stability ----
meta = json.load(open(os.path.join(probe_dir, "stab", "meta.json")))
dims = {str(i["id"]): (i["w"], i["h"]) for i in meta["images"]}
stab = {}
for tag, root in ROOTS:
    agg = collections.defaultdict(lambda: collections.Counter())
    for base, variants in meta["variants"].items():
        w, h = dims[base]
        orig = boxes(load(root, "stab", f"{base}__orig"), w, h)
        orig_person = any(o["name"] in PERSONISH for o in orig)
        for vname, vm in variants.items():
            dx, dy, _, _ = vm["crop"]
            sxf, syf = w / (w - dx), h / (h - dy)
            tr = []
            for o in orig:
                x1, y1, x2, y2 = o["box"]
                nb = [(x1 - dx) * sxf, (y1 - dy) * syf, (x2 - dx) * sxf, (y2 - dy) * syf]
                cb = [max(0, nb[0]), max(0, nb[1]), min(w, nb[2]), min(h, nb[3])]
                full = (nb[2] - nb[0]) * (nb[3] - nb[1])
                vis = max(0, cb[2] - cb[0]) * max(0, cb[3] - cb[1])
                if full > 0 and vis / full >= 0.7:
                    tr.append({"name": o["name"], "box": cb})
            var = boxes(load(root, "stab", f"{base}__{vname}"), w, h)
            mm = match(tr, var)
            c = agg[vname]
            c["orig_objs"] += len(tr)
            c["persist"] += len(mm)
            c["persist_same_label"] += sum(1 for i, j in mm if tr[i]["name"] == var[j]["name"])
            c["new_objs"] += len(var) - len(mm)
            c["var_objs"] += len(var)
            c["person_flip"] += orig_person != any(o["name"] in PERSONISH for o in var)
            c["images"] += 1
    stab[tag] = {v: {"box_persist_%": round(100 * c["persist"] / max(c["orig_objs"], 1), 1),
                     "same_label_persist_%": round(100 * c["persist_same_label"] / max(c["orig_objs"], 1), 1),
                     "new_boxes_%_of_variant": round(100 * c["new_objs"] / max(c["var_objs"], 1), 1),
                     "person_presence_flips": f"{c['person_flip']}/{c['images']}"} for v, c in agg.items()}
report["stability"] = stab


# ---- OCR ----
def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


ocr_meta = json.load(open(os.path.join(probe_dir, "ocr", "meta.json"))) if mroot else []
by = collections.defaultdict(lambda: collections.Counter())
ocr_rows = []
for e in ocr_meta:
    r = load(mroot, "ocr", os.path.splitext(e["file"])[0]) or {}
    lines = [(t.get("string") or "").upper().strip() for t in r.get("text", [])]
    cands = lines + [lines[i] + " " + lines[i + 1] for i in range(len(lines) - 1)]
    gt = e["text"].upper()
    best = min(cands, key=lambda s: lev(gt, s)) if cands else ""
    cer = lev(gt, best) / len(gt)
    key = e["size"] + ("_rot" if e["rot"] else "")
    by[key]["n"] += 1
    by[key]["exact"] += best == gt
    by[key]["cer_x1000"] += int(cer * 1000)
    ocr_rows.append({"file": e["file"], "gt": gt, "read": lines})
report["ocr"] = {k: {"n": c["n"], "exact_%": round(100 * c["exact"] / c["n"], 1),
                     "mean_CER": round(c["cer_x1000"] / 1000 / c["n"], 3)} for k, c in by.items()}
report["ocr_reads"] = ocr_rows
print(json.dumps(report, indent=1, ensure_ascii=False))
