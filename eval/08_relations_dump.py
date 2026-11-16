"""Print Mirage relations/scene/reasoning with resolved object names, for manual review.

Usage: python 08_relations_dump.py <mirage_coco_dir> <id> [<id> ...]
"""
import json
import os
import sys

d = sys.argv[1]
for iid in sys.argv[2:]:
    r = json.load(open(os.path.join(d, f"{iid}.json"), encoding="utf-8"))
    names = {}
    for o in r.get("objects", []):
        names.setdefault(o["id"], o["name"])
    for g in r.get("regions", []):
        names.setdefault(g.get("id"), g.get("name"))
    rels = [f"{names.get(x['subject'], x['subject'])} --{x['predicate']}--> {names.get(x['object'], x['object'])}"
            for x in r.get("relations", [])]
    print(f"== {iid}: scene={r.get('scene')}")
    print("   objects:", sorted({o['name'] for o in r.get('objects', [])}))
    print("   relations:", rels)
    print("   events:", r.get("events"))
    print("   text:", [t.get("string") for t in r.get("text", [])])
    print("   reasoning:", r.get("reasoning"))
