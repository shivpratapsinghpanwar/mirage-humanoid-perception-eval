"""First contact with Mirage: load, warm up, dump raw output for one image.

Usage: python 01_first_look.py <image> [out.json]
"""
import json
import os
import sys
import time

os.environ.setdefault("CACTUS_TELEMETRY", "0")

import cactus  # noqa: E402

img = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None

t0 = time.perf_counter()
m = cactus.Mirage()
t1 = time.perf_counter()
print(f"load (incl. first-time engine/weights fetch): {t1 - t0:.2f}s", file=sys.stderr)
print("weights:", m.weights, file=sys.stderr)

r = m.describe(img)  # default options
t2 = time.perf_counter()
print(f"first describe (cold): {(t2 - t1) * 1000:.0f} ms", file=sys.stderr)

times = []
for _ in range(3):
    a = time.perf_counter()
    r = m.describe(img)
    times.append((time.perf_counter() - a) * 1000)
print("warm describe ms:", [round(t) for t in times], file=sys.stderr)

txt = json.dumps(r, indent=1, ensure_ascii=False)
if out:
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
print(txt[:6000])
