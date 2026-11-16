"""Thin robust wrapper around cactus.Mirage.

Works around an engine bug (cactus-compute 4.2.2 / engine for mirage@76c1e9e): the
OCR head can emit a NaN score, which the C engine serialises as `-nan(ind)` (MSVC)
or `nan`, producing invalid JSON and crashing Mirage.describe(). We call the engine
directly, replace non-finite numbers with null, and count how often that happened.
"""
from __future__ import annotations

import json
import os
import re

os.environ.setdefault("CACTUS_TELEMETRY", "0")

import cactus  # noqa: E402
from cactus.agent import mirage as _mm  # noqa: E402
from cactus.agent.audio import _options  # noqa: E402

_NONFINITE = re.compile(r'(?<=[:\[,])\s*-?(?:nan(?:\([a-z]*\))?|inf(?:inity)?)(?=\s*[,}\]])', re.IGNORECASE)


class SafeMirage:
    def __init__(self):
        self.m = cactus.Mirage()
        self.sanitized = 0
        self.calls = 0

    def describe(self, image, *, language=None, keywords=None, **opts):
        rgb, w, h = _mm._image(image)
        lang, kw = _options(language, keywords)
        lib = self.m._bind()
        enc = json.dumps(opts, ensure_ascii=False, allow_nan=False).encode("utf-8")
        code = lib.cactus_describe(rgb, w, h, enc, lang, kw, self.m._buffer, len(self.m._buffer))
        if code < 0:
            raise RuntimeError(lib.cactus_last_error().decode("utf-8", "replace"))
        s = self.m._buffer.value.decode("utf-8", "replace")
        self.calls += 1
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            fixed, n = _NONFINITE.subn(" null", s)
            if n:
                self.sanitized += 1
            return json.loads(fixed)

    def extract(self, image, form, **opts):
        return self.describe(image, form=form, **opts)
