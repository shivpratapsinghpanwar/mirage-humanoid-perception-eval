"""Probe extract() form syntax and output format on one image."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402

img = sys.argv[1]
m = SafeMirage()
forms = {
    "enum": {"person_present": {"enum": ["yes", "no"]}},
    "int_minmax": {"n_people": {"type": "integer", "minimum": 0, "maximum": 10}},
    "int_bounds": {"n_people": {"integer": [0, 10]}},
    "string": {"main_object": {"type": "string"}},
    "box_ref": {"target": {"type": "box"}},
    "multi": {"room": {"enum": ["kitchen", "living room", "bedroom", "bathroom", "office", "outdoors"]},
              "person_present": {"enum": ["yes", "no"]},
              "path_clear": {"enum": ["yes", "no"]}},
}
for k, f in forms.items():
    try:
        r = m.extract(img, f)
        print(f"== {k}: {json.dumps(r, ensure_ascii=False)[:900]}")
    except Exception as e:  # noqa: BLE001
        print(f"== {k}: ERROR {e}")
