"""Score answers.jsonl from 16_run_qa.py against results/qa_set.json with the frozen parsing rules.

Usage: python 17_score_qa.py <answers.jsonl> [--out summary.json]
Per task: accuracy (abstentions count as wrong), abstain rate; dog_or_cat also reports the answer
split per true class (a model that always says "dog" scores 50% but answers "dog" for every cat);
people_count also exact-match and mean absolute error; sign_text exact match and character error rate.
"""
import collections
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ans_path = sys.argv[1]
out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
qa = {q["qid"]: q for q in json.load(open(os.path.join(ROOT, "results", "qa_set.json"), encoding="utf-8"))["items"]}
rows = [json.loads(x) for x in open(ans_path, encoding="utf-8") if x.strip()]
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def parse(task, raw):
    s = (raw or "").lower()
    if task in ("person_present", "chair_present", "blank_person", "blank_text"):
        m = re.search(r"\b(yes|no)\b", s)
        return m.group(1) if m else None
    if task == "people_count":
        m = re.search(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", s)
        return None if not m else (int(m.group(1)) if m.group(1).isdigit() else WORDS[m.group(1)])
    if task == "dog_or_cat":
        hits = [(m.start(), m.group(1)) for m in re.finditer(r"\b(dog|cat|puppy|kitten)s?\b", s)]
        return None if not hits else {"puppy": "dog", "kitten": "cat"}.get(min(hits)[1], min(hits)[1])
    if task == "sign_text":
        return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", (raw or "").upper())).strip() or None
    raise KeyError(task)


def cer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(hyp)] / max(len(ref), 1)


T = collections.defaultdict(lambda: collections.Counter())
split = collections.defaultdict(collections.Counter)
abs_err, cers, lat, errors = [], [], [], 0
for r in rows:
    q = qa[r["qid"]]
    t = q["task"]
    if r.get("error"):
        errors += 1
    if r.get("ms") and not r.get("error"):
        lat.append(r["ms"])
    p = parse(t, r.get("raw"))
    T[t]["n"] += 1
    T[t]["abstain"] += p is None
    if t == "sign_text":
        T[t]["ok"] += p == q["answer"]
        cers.append(min(1.0, cer(q["answer"], p or "")))
    else:
        T[t]["ok"] += p == q["answer"]
    if t == "people_count" and p is not None:
        abs_err.append(abs(p - q["answer"]))
    if t == "dog_or_cat":
        split[q["answer"]][p or "abstain"] += 1
    if t == "person_present":
        T[t][f"gt_{q['answer']}"] += 1
        T[t][f"ok_{q['answer']}"] += p == q["answer"]

lat.sort()
pc = lambda a, b: round(100 * a / b, 1) if b else None  # noqa: E731
summary = {"answers": len(rows), "errors": errors,
           "latency_ms_per_question": {"median": round(lat[len(lat) // 2]) if lat else None,
                                       "p90": round(lat[int(len(lat) * 0.9)]) if lat else None},
           "tasks": {t: {"n": c["n"], "accuracy_%": pc(c["ok"], c["n"]), "abstain_%": pc(c["abstain"], c["n"])}
                     for t, c in sorted(T.items())}}
if "person_present" in T:
    c = T["person_present"]
    summary["tasks"]["person_present"].update({"recall_%": pc(c["ok_yes"], c["gt_yes"]),
                                               "specificity_%": pc(c["ok_no"], c["gt_no"])})
if abs_err:
    summary["tasks"]["people_count"]["mae"] = round(sum(abs_err) / len(abs_err), 2)
if cers:
    summary["tasks"]["sign_text"]["mean_cer"] = round(sum(cers) / len(cers), 3)
if split:
    summary["tasks"]["dog_or_cat"]["answers_by_true_class"] = {k: dict(v) for k, v in split.items()}
txt = json.dumps(summary, indent=1)
if out_path:
    open(out_path, "w", encoding="utf-8", newline="\n").write(txt + "\n")
print(txt)
