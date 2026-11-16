"""Summarize a Mirage result JSON: keys, objects, and every non-object section."""
import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
print("TOP KEYS:", list(r.keys()))
objs = r.get("objects", [])
print("n objects:", len(objs))
core = {"id", "name", "bbox", "score", "mask", "position", "description"}
for o in objs:
    extra = [k for k in o if k not in core]
    print(f"  id={o.get('id'):>2} {o.get('name', ''):<18} score={o.get('score', 0):.2f} "
          f"bbox={o.get('bbox')} pos={o.get('position')} desc={o.get('description')} extra={extra}")
for k, v in r.items():
    if k != "objects":
        print(f"\n== {k} ==")
        print(json.dumps(v, indent=1, ensure_ascii=False)[:3000])
