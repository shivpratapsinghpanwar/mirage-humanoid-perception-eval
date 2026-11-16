"""Try candidate syntaxes for a box-reference field in Mirage forms."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402

m = SafeMirage()
img = sys.argv[1]
cands = [{"box": True}, {"type": "box"}, {"type": "boxes"}, {"box": "object"}, {"type": "reference"},
         {"ref": "objects"}, {"$ref": "#/objects"}, {"type": "object"}, {"format": "box"},
         {"type": "string", "format": "box"}, {"reference": "object"}, {"refers": "objects"}, {"kind": "box"},
         {"box": {}}, {"type": "array", "items": {"type": "box"}}]
for c in cands:
    try:
        r = m.extract(img, {"thing_to_pick_up": c})
        print("OK ", json.dumps(c), "->", json.dumps(r)[:500])
    except Exception as e:  # noqa: BLE001
        print("ERR", json.dumps(c), "->", str(e)[:120])
