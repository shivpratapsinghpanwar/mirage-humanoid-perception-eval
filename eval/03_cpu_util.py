"""Measure how many cores Mirage actually uses (process CPU time / wall time).

Also checks engine library details (file name/size) and CPU ISA flags of this host.
"""
import os
import sys
import time

os.environ.setdefault("CACTUS_TELEMETRY", "0")
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from PIL import Image  # noqa: E402

import cactus  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_safe import SafeMirage  # noqa: E402

img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
m = SafeMirage()
m.describe(img)
p = psutil.Process()
for t in [0, 1, 4, 8]:
    cactus.set_threads(t)
    m.describe(img)
    c0 = sum(p.cpu_times()[:2])
    w0 = time.perf_counter()
    for _ in range(4):
        m.describe(img)
    wall = time.perf_counter() - w0
    cpu = sum(p.cpu_times()[:2]) - c0
    print(f"set_threads({t}): wall/call {wall / 4 * 1000:.0f} ms, cores busy ~{cpu / wall:.2f}, "
          f"threads in process {p.num_threads()}")

from cactus.agent import fetch  # noqa: E402
lib = fetch.lib_name() if hasattr(fetch, "lib_name") else "?"
print("engine lib name:", lib, "| platform tag:", fetch._platform_tag() if hasattr(fetch, "_platform_tag") else "?")
for root in [fetch.engine_cache_dir()] if hasattr(fetch, "engine_cache_dir") else []:
    for dp, _, fs in os.walk(root):
        for f in fs:
            fp = os.path.join(dp, f)
            print("  ", fp.replace(os.path.expanduser("~"), "~"), os.path.getsize(fp))
