"""extract() with a free-text field on the synthetic sign probes (compare with describe() OCR in 06c).

Usage: python 06f_extract_ocr.py <probes/ocr dir with meta.json> <out.json>
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


probe_dir, out_path = sys.argv[1:3]
meta = json.load(open(os.path.join(probe_dir, "meta.json"), encoding="utf-8"))
m = SafeMirage()
rows, by = [], collections.defaultdict(lambda: {"n": 0, "exact": 0, "cer": 0.0})
for e in meta:
    r = m.extract(os.path.join(probe_dir, e["file"]), {"sign_text": {"type": "string"}})["form"]["sign_text"]
    gt, got = e["text"].upper(), str(r.get("value") or "").upper().strip()
    cer = lev(gt, got) / len(gt)
    key = os.path.splitext(e["file"])[0].split("_", 1)[1]
    by[key]["n"] += 1
    by[key]["exact"] += got == gt
    by[key]["cer"] += cer
    rows.append({"file": e["file"], "gt": gt, "extract": r.get("value"), "cer": round(cer, 3)})
    print(e["file"], repr(gt), "->", repr(r.get("value")), flush=True)
summary = {k: {"n": v["n"], "exact_%": round(100 * v["exact"] / v["n"], 1), "mean_cer": round(v["cer"] / v["n"], 3)}
           for k, v in by.items()}
json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=1)
print(json.dumps(summary, indent=1))
