"""Call the engine directly and show raw output for images whose JSON fails to parse.

Usage: python debug_raw.py <image> [<image> ...]
"""
import json
import os
import sys

os.environ.setdefault("CACTUS_TELEMETRY", "0")

import cactus  # noqa: E402
from cactus.agent import mirage as mm  # noqa: E402
from cactus.agent.audio import _options  # noqa: E402

m = cactus.Mirage()


def raw(image, **opts):
    rgb, w, h = mm._image(image)
    lang, kw = _options(None, None)
    lib = m._bind()
    code = lib.cactus_describe(rgb, w, h, json.dumps(opts).encode(), lang, kw, m._buffer, len(m._buffer))
    return code, m._buffer.value.decode("utf-8", "replace")


for p in sys.argv[1:]:
    code, s = raw(p)
    try:
        json.loads(s)
        print(f"OK   {os.path.basename(p)} code={code} len={len(s)}")
    except json.JSONDecodeError as e:
        print(f"BAD  {os.path.basename(p)} code={code} len={len(s)} err={e}")
        lo = max(0, e.pos - 300)
        print("     ...context:", repr(s[lo:e.pos + 200]))
