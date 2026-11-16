"""Control for 06d: same dog/cat images with the enum order swapped, plus describe() on them.

If extract() answers by option position, swapping the order flips the answers. If describe() names the
animal correctly, the vision backbone sees it and the weakness is in the form-filling head.

Usage: python 06e_extract_order.py <val2017_dir> <extract_fairness.json> <out.json>
"""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402

img_dir, prev_path, out_path = sys.argv[1:4]
prev = json.load(open(prev_path, encoding="utf-8"))["animal"]
m = SafeMirage()
rows = []
for r in prev:
    p = os.path.join(img_dir, f"{r['id']:012d}.jpg")
    swapped = m.extract(p, {"animal": {"enum": ["cat", "dog"]}})["form"]["animal"]
    names = [o.get("name", "") for o in m.describe(p, sections=["objects"]).get("objects", [])]
    said = "dog" if any("dog" in n or "puppy" in n for n in names) else ""
    said += "cat" if any("cat" in n or "kitten" in n for n in names) else ""
    rows.append({"id": r["id"], "gt": r["gt"], "dog_first": r["value"], "p_dog_dog_first": r["probabilities"].get("dog"),
                 "cat_first": swapped["value"], "p_dog_cat_first": swapped["probabilities"].get("dog"),
                 "describe_animals": said or "neither", "describe_names": names[:8]})
    print(rows[-1]["id"], r["gt"], "dog-first:", r["value"], "cat-first:", swapped["value"],
          "describe:", rows[-1]["describe_animals"], flush=True)


def mean_p(key, gt):
    v = [x[key] for x in rows if x["gt"] == gt and x[key] is not None]
    return round(statistics.mean(v), 3) if v else None


summary = {
    "dog_first_answers": dict(collections.Counter(f"{x['gt']}->{x['dog_first']}" for x in rows)),
    "cat_first_answers": dict(collections.Counter(f"{x['gt']}->{x['cat_first']}" for x in rows)),
    "mean_P(dog)_dog_first": {"dog_imgs": mean_p("p_dog_dog_first", "dog"), "cat_imgs": mean_p("p_dog_dog_first", "cat")},
    "mean_P(dog)_cat_first": {"dog_imgs": mean_p("p_dog_cat_first", "dog"), "cat_imgs": mean_p("p_dog_cat_first", "cat")},
    "describe_names_correct_animal": sum(x["describe_animals"] == x["gt"] for x in rows),
    "describe_answers": dict(collections.Counter(f"{x['gt']}->{x['describe_animals']}" for x in rows)),
    "n": len(rows),
}
json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=1)
print(json.dumps(summary, indent=1))
