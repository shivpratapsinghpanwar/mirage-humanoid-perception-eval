"""Freeze the question set every VLM (and Mirage's extract()) answers: results/qa_set.json.

Built from the images and ground truth of the published extract() tests, so the questions are the
same ones Mirage was asked:
  person_present  60 COCO images (30 with 1-10 people, 30 without)      yes/no
  people_count    same 60 images                                        integer 0-10
  chair_present   same 60 images (8 contain a chair)                    yes/no
  dog_or_cat      30 COCO photos with only a dog or only a cat          dog/cat
  sign_text       32 synthetic signs (8 texts x large/medium/small/rot15) free text
  blank_person    6 blank/noise frames: is there a person?              yes/no (answer: no)
  blank_text      6 blank/noise frames: is there any text?              yes/no (answer: no)
Usage: python 15_build_qa_set.py   (run from anywhere; paths are repo-relative)
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
coco = lambda i: f"data/coco_val2017/{int(i):012d}.jpg"  # noqa: E731

Q = {
    "person_present": "Is there a person in this image? Answer yes or no.",
    "people_count": "How many people are in this image? Answer with a single number.",
    "chair_present": "Is there a chair in this image? Answer yes or no.",
    "dog_or_cat": "Is the animal in this image a dog or a cat? Answer with one word: dog or cat.",
    "sign_text": "What does the sign in this image say? Answer with the text only.",
    "blank_person": "Is there a person in this image? Answer yes or no.",
    "blank_text": "Is there any text in this image? Answer yes or no.",
}
# Mirage extract() forms for the same questions (field names as in the published tests)
FORM = {
    "person_present": {"person_present": {"enum": ["yes", "no"]}},
    "people_count": {"n_people": {"type": "integer", "minimum": 0, "maximum": 10}},
    "chair_present": {"chair_present": {"enum": ["yes", "no"]}},
    "dog_or_cat": {"animal": {"enum": ["dog", "cat"]}},
    "sign_text": {"sign_text": {"type": "string"}},
    "blank_person": {"person_present": {"enum": ["yes", "no"]}},
    "blank_text": {"text_present": {"enum": ["yes", "no"]}},
}
items = []
for r in json.load(open(R("results", "extract_eval.json"), encoding="utf-8"))["rows"]:
    items.append({"task": "person_present", "image": coco(r["id"]), "image_id": r["id"],
                  "answer": "yes" if r["gt_people"] > 0 else "no"})
    items.append({"task": "people_count", "image": coco(r["id"]), "image_id": r["id"], "answer": int(r["gt_people"])})
    items.append({"task": "chair_present", "image": coco(r["id"]), "image_id": r["id"],
                  "answer": "yes" if r["gt_chair"] else "no"})
for r in json.load(open(R("results", "extract_fairness.json"), encoding="utf-8"))["animal"]:
    items.append({"task": "dog_or_cat", "image": coco(r["id"]), "image_id": r["id"], "answer": r["gt"]})
for e in json.load(open(R("results", "probes", "ocr", "meta.json"), encoding="utf-8")):
    items.append({"task": "sign_text", "image": f"results/probes/ocr/{e['file']}", "answer": e["text"].upper()})
for f in sorted(os.listdir(R("results", "probes", "null"))):
    if f.endswith(".png"):
        for t in ("blank_person", "blank_text"):
            items.append({"task": t, "image": f"results/probes/null/{f}", "answer": "no"})
for k, it in enumerate(items):
    it["qid"] = k
    it["question"] = Q[it["task"]]
    it["mirage_form"] = FORM[it["task"]]
doc = {"version": 1, "note": "Frozen 2026-11-24. Same images/ground truth as the published extract() tests.",
       "parsing": {"yes_no": "lower-case; first standalone 'yes' or 'no' in the answer; otherwise abstain",
                   "count": "first integer (digits or the words zero..ten); otherwise abstain",
                   "dog_or_cat": "whichever of 'dog'/'cat' appears first; neither = abstain",
                   "sign_text": "upper-case, keep letters/digits/spaces, collapse spaces; exact match and CER"},
       "items": items}
with open(R("results", "qa_set.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(doc, f, indent=1)
from collections import Counter  # noqa: E402
print(len(items), "questions:", dict(Counter(i["task"] for i in items)))
