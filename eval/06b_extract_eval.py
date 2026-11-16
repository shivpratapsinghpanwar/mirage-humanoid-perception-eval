"""Known-answer test of Mirage.extract() on COCO images.

Usage: python 06b_extract_eval.py <val2017_dir> <instances_val2017.json> <mirage_coco_dir> <out.json> [n_each=30]
Picks n_each images with 1-10 people (no crowd) and n_each with none, from the main sample.
"""
import collections
import gc
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402

img_dir, ann_path, mdir, out_path = sys.argv[1:5]
n_each = int(sys.argv[5]) if len(sys.argv) > 5 else 30
coco = json.load(open(ann_path, encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
persons, crowd, chairs = collections.Counter(), set(), collections.Counter()
for a in coco["annotations"]:
    n = cats[a["category_id"]]
    if n == "person":
        if a.get("iscrowd"):
            crowd.add(a["image_id"])
        else:
            persons[a["image_id"]] += 1
    if n == "chair" and not a.get("iscrowd"):
        chairs[a["image_id"]] += 1
del coco  # free the parsed annotations before loading the model (low-RAM laptops)
gc.collect()
sample = json.load(open(os.path.join(mdir, "_sample_ids.json")))
with_p = [i for i in sample if 1 <= persons[i] <= 10 and i not in crowd]
without_p = [i for i in sample if persons[i] == 0 and i not in crowd]
rnd = random.Random(3)
chosen = rnd.sample(with_p, n_each) + rnd.sample(without_p, n_each)

YN = {"enum": ["yes", "no"]}
F_PERSON = {"person_present": YN}
F_CHAIR = {"chair_present": YN}
F_COUNT = {"n_people": {"type": "integer", "minimum": 0, "maximum": 10}}
F_MULTI = {"place": {"enum": ["kitchen", "living room", "bedroom", "bathroom", "office", "street", "park",
                              "sports field", "shop", "other"]},
           "animal_present": YN, "person_present": YN}

# Each finished image is appended to a checkpoint file, so a killed run resumes where it stopped.
ckpt = out_path + ".partial.jsonl"
done = {}
if os.path.exists(ckpt):
    for line in open(ckpt, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            done[r["id"]] = r
print(f"{len(chosen)} images, {len(done)} already done", flush=True)
m = SafeMirage()
with open(ckpt, "a", encoding="utf-8") as fck:
    for k, iid in enumerate(chosen):
        if iid in done:
            continue
        p = os.path.join(img_dir, f"{iid:012d}.jpg")
        r1 = m.extract(p, F_PERSON)["form"]["person_present"]
        r2 = m.extract(p, F_MULTI)["form"]["person_present"]
        r3 = m.extract(p, F_COUNT)["form"]["n_people"]
        r4 = m.extract(p, F_CHAIR)["form"]["chair_present"]
        done[iid] = {"id": iid, "gt_people": persons[iid], "gt_chair": chairs[iid] > 0,
                     "single": r1, "multi": r2, "count": r3, "chair": r4}
        fck.write(json.dumps(done[iid]) + "\n")
        fck.flush()
        print(f"{k + 1}/{len(chosen)} id={iid}", flush=True)
rows = [done[i] for i in chosen]


def yn_stats(key, gt_fn):
    ans, correct, abst, conf_ok, conf_bad = 0, 0, 0, [], []
    for r in rows:
        v = r[key]["value"]
        gt = "yes" if gt_fn(r) else "no"
        if v not in ("yes", "no"):
            abst += 1
            continue
        ans += 1
        c = r[key]["probabilities"][v]
        if v == gt:
            correct += 1
            conf_ok.append(c)
        else:
            conf_bad.append(c)
    n = len(rows)
    return {"n": n, "accuracy_all_%": round(100 * correct / n, 1), "abstained": abst,
            "accuracy_when_answered_%": round(100 * correct / ans, 1) if ans else None,
            "mean_conf_correct": round(statistics.mean(conf_ok), 3) if conf_ok else None,
            "mean_conf_wrong": round(statistics.mean(conf_bad), 3) if conf_bad else None}


summary = {
    "person_present_single": yn_stats("single", lambda r: r["gt_people"] > 0),
    "person_present_in_multi_form": yn_stats("multi", lambda r: r["gt_people"] > 0),
    "chair_present_single": yn_stats("chair", lambda r: r["gt_chair"]),
}
agree = sum(1 for r in rows if r["single"]["value"] == r["multi"]["value"])
summary["single_vs_multi_same_answer_%"] = round(100 * agree / len(rows), 1)
cnt = [r for r in rows]
exact = sum(1 for r in cnt if r["count"]["value"] == r["gt_people"])
mae = statistics.mean(abs((r["count"]["value"] if isinstance(r["count"]["value"], int) else 0) - r["gt_people"]) for r in cnt)
wp = [r for r in cnt if r["gt_people"] > 0]
summary["n_people_count"] = {
    "exact_%": round(100 * exact / len(cnt), 1),
    "mae": round(mae, 2),
    "images_with_people_predicted_zero_%": round(100 * sum(1 for r in wp if r["count"]["value"] == 0) / len(wp), 1),
    "mean_conf_of_chosen": round(statistics.mean(r["count"]["probabilities"].get(str(r["count"]["value"]), 0)
                                                 for r in cnt if r["count"]["value"] is not None), 3),
    "pred_distribution": dict(collections.Counter(str(r["count"]["value"]) for r in cnt)),
}
json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=1)
os.remove(ckpt)  # the final JSON holds every row; a later rerun starts fresh instead of replaying the cache
print(json.dumps(summary, indent=1))
