"""Follow-up to 06b: is extract() weak only on yes/no questions, or in general?

1. The vendor's own documented form, {"animal": {"enum": ["dog", "cat"]}}, on COCO val2017 images that
   contain a dog or a cat (not both) covering at least 10% of the image.
2. Person questions with other field names, on 20 of the 06b images (10 with people, 10 without).

Usage: python 06d_extract_fairness.py <val2017_dir> <instances_val2017.json> <extract_eval.json> <out.json> [n_each=15]
"""
import collections
import gc
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402

img_dir, ann_path, prev_path, out_path = sys.argv[1:5]
n_each = int(sys.argv[5]) if len(sys.argv) > 5 else 15
coco = json.load(open(ann_path, encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
img_area = {i["id"]: i["width"] * i["height"] for i in coco["images"]}
has = collections.defaultdict(set)
big = collections.defaultdict(set)
for a in coco["annotations"]:
    n = cats[a["category_id"]]
    if n in ("dog", "cat"):
        has[a["image_id"]].add(n)
        if a["area"] >= 0.10 * img_area[a["image_id"]]:
            big[a["image_id"]].add(n)
del coco
gc.collect()
dogs = sorted(i for i in has if has[i] == {"dog"} and "dog" in big[i])
catz = sorted(i for i in has if has[i] == {"cat"} and "cat" in big[i])
rnd = random.Random(5)
animal_imgs = [(i, "dog") for i in rnd.sample(dogs, n_each)] + [(i, "cat") for i in rnd.sample(catz, n_each)]

prev = json.load(open(prev_path, encoding="utf-8"))["rows"]
with_p = [r for r in prev if r["gt_people"] > 0][:10]
without_p = [r for r in prev if r["gt_people"] == 0][:10]
person_rows = with_p + without_p
YN = {"enum": ["yes", "no"]}
VARIANTS = {
    "person": {"person": YN},
    "is_there_a_person_in_the_image": {"is_there_a_person_in_the_image": YN},
    "contains_people": {"contains_people": YN},
    "subject": {"subject": {"enum": ["person", "animal", "vehicle", "food", "object"]}},
    "number_of_people": {"number_of_people": {"type": "integer", "minimum": 0, "maximum": 10}},
}

m = SafeMirage()
out = {"animal": [], "person": []}
for iid, gt in animal_imgs:
    r = m.extract(os.path.join(img_dir, f"{iid:012d}.jpg"), {"animal": {"enum": ["dog", "cat"]}})["form"]["animal"]
    out["animal"].append({"id": iid, "gt": gt, **r})
    print("animal", iid, gt, r["value"], flush=True)
for row in person_rows:
    p = os.path.join(img_dir, f"{row['id']:012d}.jpg")
    rec = {"id": row["id"], "gt_people": row["gt_people"]}
    for name, form in VARIANTS.items():
        rec[name] = m.extract(p, form)["form"][next(iter(form))]
    out["person"].append(rec)
    print("person", row["id"], row["gt_people"], {k: rec[k]["value"] for k in VARIANTS}, flush=True)

an = out["animal"]
summary = {"animal_dog_vs_cat": {
    "n": len(an),
    "correct": sum(r["value"] == r["gt"] for r in an),
    "abstained": sum(r["value"] not in ("dog", "cat") for r in an),
    "answers": dict(collections.Counter(f"{r['gt']}->{r['value']}" for r in an))}}
for name in VARIANTS:
    rows = out["person"]
    if name == "number_of_people":
        summary[name] = {"exact": sum(r[name]["value"] == r["gt_people"] for r in rows), "n": len(rows),
                         "answers": dict(collections.Counter(f"gt{r['gt_people']}->{r[name]['value']}" for r in rows))}
        continue
    yes_val = "person" if name == "subject" else "yes"
    summary[name] = {"n": len(rows),
                     "answers": dict(collections.Counter(
                         f"{'people' if r['gt_people'] else 'none'}->{r[name]['value']}" for r in rows))}
    if name != "subject":
        summary[name]["correct"] = sum((r[name]["value"] == "yes") == (r["gt_people"] > 0)
                                       and r[name]["value"] in ("yes", "no") for r in rows)
    else:
        summary[name]["says_person_when_people"] = sum(r[name]["value"] == yes_val
                                                       for r in rows if r["gt_people"] > 0)
json.dump({"summary": summary, **out}, open(out_path, "w"), indent=1)
print(json.dumps(summary, indent=1))
