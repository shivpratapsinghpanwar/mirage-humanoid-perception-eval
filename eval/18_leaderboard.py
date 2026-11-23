"""Build the leaderboard from the raw outputs in results/ (detectors and VLM question answering).

Usage: python 18_leaderboard.py
Writes results/leaderboard.json and results/leaderboard.md. Every number is recomputed from raw
outputs with eval/05_coco_score.py, eval/06c_score_probes.py and eval/17_score_qa.py, each detector
at its library's documented default score threshold. Models without results are skipped.
"""
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EV = os.path.join(ROOT, "eval")
RES = os.path.join(ROOT, "results")
PY = sys.executable
R = lambda *p: os.path.join(RES, *p)  # noqa: E731

# key, name, params (M), licence, default threshold (None = model returns no usable score),
# {hardware: run root}. Accuracy comes from the most complete run; latency from every run (>= 20 images).
DETECTORS = [
    ("mirage", "Mirage (cactus-compute 4.2.7)", 51.3, "Apache-2.0", None,
     {"laptop CPU": R("runs", "mirage-4.2.7", "cpu")}),
    ("yolo26n", "YOLO26n", 2.4, "AGPL-3.0", 0.25,
     {"laptop CPU": R("runs", "yolo26n", "cpu"), "laptop GPU": R("runs", "yolo26n", "cuda")}),
    ("rfdetr-nano", "RF-DETR Nano", None, "Apache-2.0", 0.5,
     {"laptop CPU": R("runs", "rfdetr-nano", "cpu"), "laptop GPU": R("runs", "rfdetr-nano", "cuda")}),
    ("rfdetr-small", "RF-DETR Small", None, "Apache-2.0", 0.5,
     {"laptop CPU": R("runs", "rfdetr-small", "cpu"), "laptop GPU": R("runs", "rfdetr-small", "cuda")}),
    ("dfine-n", "D-FINE N", None, "Apache-2.0", 0.5,
     {"laptop CPU": R("runs", "dfine-n", "cpu"), "laptop GPU": R("runs", "dfine-n", "cuda")}),
    ("dfine-s", "D-FINE S", None, "Apache-2.0", 0.5,
     {"laptop CPU": R("runs", "dfine-s", "cpu"), "laptop GPU": R("runs", "dfine-s", "cuda")}),
    ("rtmdet-tiny", "RTMDet-tiny", None, "Apache-2.0", 0.3,
     {"laptop CPU": R("runs", "rtmdet-tiny", "cpu"), "laptop GPU": R("runs", "rtmdet-tiny", "cuda")}),
    ("rtmdet-s", "RTMDet-s", None, "Apache-2.0", 0.3,
     {"laptop CPU": R("runs", "rtmdet-s", "cpu"), "laptop GPU": R("runs", "rtmdet-s", "cuda")}),
    ("owlv2-base", "OWLv2 base (80 COCO prompts)", None, "Apache-2.0", 0.1,
     {"laptop CPU": R("runs", "owlv2-base", "cpu"), "laptop GPU": R("runs", "owlv2-base", "cuda")}),
    ("florence2-base", "Florence-2 base (<OD>)", None, "MIT", None,
     {"laptop CPU": R("runs", "florence2-base", "cpu"), "laptop GPU": R("runs", "florence2-base", "cuda")}),
    ("moondream2-detect-person", "Moondream 2 detect('person')", None, "Apache-2.0", None,
     {"Kaggle GPU": R("runs", "moondream2-detect-person", "kaggle-gpu")}),
]
VLMS = [
    ("mirage-extract", "Mirage extract() (4.2.7)", 51.3, "Apache-2.0", {"laptop CPU": R("runs", "mirage-4.2.7", "cpu", "qa")}),
    ("smolvlm2-256m", "SmolVLM2 256M", None, "Apache-2.0",
     {"laptop CPU": R("runs", "smolvlm2-256m", "cpu", "qa"), "laptop GPU": R("runs", "smolvlm2-256m", "cuda", "qa")}),
    ("smolvlm2-500m", "SmolVLM2 500M", None, "Apache-2.0",
     {"laptop CPU": R("runs", "smolvlm2-500m", "cpu", "qa"), "laptop GPU": R("runs", "smolvlm2-500m", "cuda", "qa")}),
    ("lfm2.5-vl-450m", "LFM2.5-VL 450M", None, "LFM Open License v1.0",
     {"laptop CPU": R("runs", "lfm2.5-vl-450m", "cpu", "qa"), "laptop GPU": R("runs", "lfm2.5-vl-450m", "cuda", "qa")}),
    ("lfm2.5-vl-1.6b", "LFM2.5-VL 1.6B", None, "LFM Open License v1.0",
     {"laptop GPU": R("runs", "lfm2.5-vl-1.6b", "cuda", "qa"), "Kaggle GPU": R("runs", "lfm2.5-vl-1.6b", "kaggle-gpu", "qa")}),
    ("moondream2", "Moondream 2 (2025-06-21)", None, "Apache-2.0", {"Kaggle GPU": R("runs", "moondream2", "kaggle-gpu", "qa")}),
    ("qwen3-vl-2b", "Qwen3-VL 2B Instruct", None, "Apache-2.0", {"Kaggle GPU": R("runs", "qwen3-vl-2b", "kaggle-gpu", "qa")}),
    ("smolvlm2-2.2b", "SmolVLM2 2.2B", None, "Apache-2.0", {"Kaggle GPU": R("runs", "smolvlm2-2.2b", "kaggle-gpu", "qa")}),
]


def run_json(args):
    r = subprocess.run([PY, "-I", *args], capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])
    return json.loads(r.stdout)


def median_ms(index_path):
    if not os.path.exists(index_path):
        return None
    ms = [json.loads(x)["ms"] for x in open(index_path, encoding="utf-8") if x.strip()]
    ms = [m for m in ms if m]
    return round(statistics.median(ms)) if len(ms) >= 20 else None


def meta(root):
    for p in (os.path.join(root, "run_meta.json"),):
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    return {}


board = {"detectors": [], "vlm_qa": []}
for key, name, params, lic, thr, roots in DETECTORS:
    # accuracy source: the run with the most COCO outputs (ties: the one that also ran the probes)
    cands = [(len([f for f in os.listdir(os.path.join(r, "coco")) if f[0] != "_"]),
              os.path.isdir(os.path.join(r, "stab")), r) for r in roots.values() if os.path.isdir(os.path.join(r, "coco"))]
    if not cands or max(cands)[0] < 300:
        print("incomplete or missing", key, flush=True)
        continue
    acc_root = max(cands)[2]
    args = [os.path.join(EV, "05_coco_score.py"), os.path.join(ROOT, "data", "coco_val2017_used.json"),
            os.path.join(acc_root, "coco")] + (["--min-score", str(thr)] if thr is not None else [])
    s = run_json(args)
    pargs = [os.path.join(EV, "06c_score_probes.py"), os.path.join(RES, "probes"), f"m={acc_root}"]
    p = run_json(pargs + (["--min-score", str(thr)] if thr is not None else [])) if os.path.isdir(
        os.path.join(acc_root, "stab")) else None
    m = meta(acc_root)
    row = {"key": key, "model": name, "params_M": params or (round(m["params"] / 1e6, 1) if m.get("params") else None),
           "licence": lic, "threshold": thr, "accuracy_run": os.path.relpath(acc_root, ROOT).replace("\\", "/"),
           "recall_all": s["class_agnostic_recall@0.5"]["all"], "recall_small": s["class_agnostic_recall@0.5"]["small"],
           "recall_medium": s["class_agnostic_recall@0.5"]["medium"], "recall_large": s["class_agnostic_recall@0.5"]["large"],
           "precision_all": s["class_agnostic_precision@0.5_%"], "boxes_per_image": s["preds_per_image"],
           "label_agreement": s["label_agreement_on_matched_%"]["all"],
           "person_recall": s["person"]["instance_recall@0.5_%"], "person_precision": s["person"]["instance_precision@0.5_%"],
           "person_image_false_alarm": s["person"]["image_false_alarm_rate_%"], "run_errors": s["run_errors"]}
    if key.endswith("detect-person"):  # queried for "person" only: all-object metrics are meaningless
        for f in ("recall_all", "recall_small", "recall_medium", "recall_large", "precision_all", "label_agreement"):
            row[f] = None
        row["note"] = "person-only queries"
    if p:
        st = p["stability"]["m"]
        row["same_label_persistence_mean"] = round(statistics.mean(v["same_label_persist_%"] for v in st.values()), 1)
        row["same_label_persistence_min"] = min(v["same_label_persist_%"] for v in st.values())
        row["blank_frame_boxes"] = sum(len(v.get("m", {}).get("objects", [])) for v in p["null_frames"].values())
    row["latency_ms_median"] = {hw: median_ms(os.path.join(r, "coco", "_index.jsonl")) for hw, r in roots.items()}
    row["peak_memory_mb"] = {hw: (meta(r).get("peak_gpu_mem_mb") or meta(r).get("peak_rss_mb")) for hw, r in roots.items()
                             if meta(r)}
    board["detectors"].append(row)
    print("scored", key, flush=True)

for key, name, params, lic, roots in VLMS:
    cands = [(sum(1 for _ in open(os.path.join(r, "answers.jsonl"), encoding="utf-8")), r) for r in roots.values()
             if os.path.exists(os.path.join(r, "answers.jsonl"))]
    if not cands:
        continue
    acc = max(cands)[1]
    s = run_json([os.path.join(EV, "17_score_qa.py"), os.path.join(acc, "answers.jsonl")])
    if s["answers"] < 254:
        print("incomplete", key, s["answers"], flush=True)
        continue
    m = meta(acc)
    t = s["tasks"]
    row = {"key": key, "model": name, "params_M": params or (round(m["params"] / 1e6, 1) if m.get("params") else None),
           "licence": lic, "revision": m.get("revision") or m.get("weights_rev"),
           "accuracy_run": os.path.relpath(acc, ROOT).replace("\\", "/"),
           "person_present_acc": t["person_present"]["accuracy_%"], "person_recall": t["person_present"].get("recall_%"),
           "person_specificity": t["person_present"].get("specificity_%"),
           "people_count_exact": t["people_count"]["accuracy_%"], "people_count_mae": t["people_count"].get("mae"),
           "chair_present_acc": t["chair_present"]["accuracy_%"],
           "dog_or_cat_acc": t["dog_or_cat"]["accuracy_%"],
           "cats_called_cat": t["dog_or_cat"]["answers_by_true_class"].get("cat", {}).get("cat", 0),
           "sign_exact": t["sign_text"]["accuracy_%"], "sign_cer": t["sign_text"].get("mean_cer"),
           "blank_no_person_acc": t["blank_person"]["accuracy_%"], "blank_no_text_acc": t["blank_text"]["accuracy_%"],
           "errors": s["errors"]}
    row["latency_ms_per_question"] = {}
    for hw, r in roots.items():
        ap = os.path.join(r, "answers.jsonl")
        if os.path.exists(ap):
            ms = sorted(json.loads(x)["ms"] for x in open(ap, encoding="utf-8") if x.strip())
            row["latency_ms_per_question"][hw] = round(statistics.median(ms)) if len(ms) >= 20 else None
    board["vlm_qa"].append(row)
    print("scored", key, flush=True)

with open(R("leaderboard.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(board, f, indent=1)


def fmt(v, suf=""):
    return "–" if v is None else f"{v}{suf}"


def lat(d, hw):
    return fmt(d.get(hw), " ms") if d else "–"


md = ["### Detection on COCO-300 (same 300 val2017 images, IoU 0.5)", "",
      "| Model | Params | Licence | Thr | Recall | Precision | Boxes/img | Person R / P | Label agree | "
      "Same-label persistence | Blank-frame boxes | Laptop CPU | Laptop GPU |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in board["detectors"]:
    md.append(f"| {r['model']} | {fmt(r['params_M'], 'M')} | {r['licence']} | {fmt(r['threshold'])} | "
              f"{fmt(r['recall_all'], '%')} | {fmt(r['precision_all'], '%')} | {r['boxes_per_image']} | "
              f"{fmt(r['person_recall'])} / {fmt(r['person_precision'])} | {fmt(r['label_agreement'], '%')} | "
              f"{fmt(r.get('same_label_persistence_mean'), '%')} | {fmt(r.get('blank_frame_boxes'))} | "
              f"{lat(r['latency_ms_median'], 'laptop CPU')} | {lat(r['latency_ms_median'], 'laptop GPU')} |")
md += ["", "### Visual question answering (254 frozen questions, greedy decoding)", "",
       "| Model | Params | Licence | Person? | People count (exact / MAE) | Chair? | Dog or cat | Cats called cat | "
       "Sign text exact / CER | Blank: no person | Blank: no text | Laptop CPU | Laptop GPU | Kaggle GPU |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in board["vlm_qa"]:
    L = r["latency_ms_per_question"]
    md.append(f"| {r['model']} | {fmt(r['params_M'], 'M')} | {r['licence']} | {fmt(r['person_present_acc'], '%')} | "
              f"{fmt(r['people_count_exact'], '%')} / {fmt(r['people_count_mae'])} | {fmt(r['chair_present_acc'], '%')} | "
              f"{fmt(r['dog_or_cat_acc'], '%')} | {r['cats_called_cat']}/15 | {fmt(r['sign_exact'], '%')} / {fmt(r['sign_cer'])} | "
              f"{fmt(r['blank_no_person_acc'], '%')} | {fmt(r['blank_no_text_acc'], '%')} | {lat(L, 'laptop CPU')} | "
              f"{lat(L, 'laptop GPU')} | {lat(L, 'Kaggle GPU')} |")
with open(R("leaderboard.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
