"""Ask one model every question in results/qa_set.json (greedy decoding), one record per question.

Usage: python 16_run_qa.py <model> [--device cpu|cuda] [--tasks dog_or_cat,...] [--limit N] [--out DIR]
Writes <out>/answers.jsonl ({"qid", "task", "raw", "ms", "error"[, "probabilities"]}) and <out>/run_meta.json.
Resumable. Score with 17_score_qa.py. Prompts are the frozen questions; no per-model prompt tuning.
Latency = wall time per question from image path to answer, after 2 warm-up questions.
"""
import argparse
import datetime
import importlib.metadata as md
import json
import os
import platform
import sys
import time

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MAX_NEW = {"sign_text": 24}  # every other task needs a word or a number


def _ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None


class MirageExtract:
    """Stock cactus.Mirage().extract with the form from qa_set.json (the published extract() tests)."""

    def __init__(self, device="cpu", **_):
        import cactus
        self.m = cactus.Mirage()

    def info(self):
        from cactus.agent import fetch
        return {"package": "cactus-compute", "version": _ver("cactus-compute"), "engine": fetch.ENGINE_VERSION,
                "weights_rev": fetch.MODEL_REVISIONS[fetch.MIRAGE], "device": "cpu", "api": "extract()"}

    def __call__(self, path, item):
        form = item["mirage_form"]
        r = self.m.extract(path, form)["form"][next(iter(form))]
        return {"raw": "" if r.get("value") is None else str(r.get("value")),
                "probabilities": r.get("probabilities")}


class HFVLM:
    """Any transformers image-text-to-text model with a chat template (SmolVLM2, LFM2-VL, Qwen-VL, Gemma)."""

    def __init__(self, repo, device="cpu", dtype="auto", **_):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.repo, self.device = torch, repo, device
        dt = {"auto": (torch.float16 if device.startswith("cuda") else torch.float32),
              "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dt).to(device).eval()
        self.dtype = dt

    def info(self):
        return {"package": "transformers", "version": _ver("transformers"), "repo": self.repo,
                "revision": getattr(self.model.config, "_commit_hash", None), "dtype": str(self.dtype),
                "torch": self.torch.__version__, "device": self.device,
                **({"gpu": self.torch.cuda.get_device_name(0)} if self.device.startswith("cuda") else {})}

    def __call__(self, path, item):
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": item["question"]}]}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW.get(item["task"], 8), do_sample=False)
        return {"raw": self.proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()}


class Moondream2:
    def __init__(self, device="cpu", repo="vikhyatk/moondream2", revision="2025-06-21", **_):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch, self.repo, self.rev, self.device = torch, repo, revision, device
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, trust_remote_code=True,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map={"": device}).eval()

    def info(self):
        return {"package": "transformers", "version": _ver("transformers"), "repo": self.repo, "revision": self.rev,
                "torch": self.torch.__version__, "device": self.device}

    def __call__(self, path, item):
        img = Image.open(path).convert("RGB")
        with self.torch.inference_mode():
            return {"raw": self.model.query(img, item["question"])["answer"].strip()}


MODELS = {
    "mirage-extract": (MirageExtract, {}),
    "smolvlm2-256m": (HFVLM, {"repo": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"}),
    "smolvlm2-500m": (HFVLM, {"repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"}),
    "smolvlm2-2.2b": (HFVLM, {"repo": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}),
    "lfm2-vl-450m": (HFVLM, {"repo": "LiquidAI/LFM2-VL-450M"}),
    "lfm2.5-vl-1.6b": (HFVLM, {"repo": "LiquidAI/LFM2.5-VL-1.6B"}),
    "qwen3-vl-2b": (HFVLM, {"repo": "Qwen/Qwen3-VL-2B-Instruct"}),
    "qwen2.5-vl-3b": (HFVLM, {"repo": "Qwen/Qwen2.5-VL-3B-Instruct"}),
    "gemma-3n-e2b": (HFVLM, {"repo": "google/gemma-3n-E2B-it"}),
    "moondream2": (Moondream2, {}),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--qa", default=os.path.join(ROOT, "results", "qa_set.json"))
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", "runs", a.model, a.device.split(":")[0], "qa")
    os.makedirs(out, exist_ok=True)
    qa = json.load(open(a.qa, encoding="utf-8"))["items"]
    if a.tasks:
        qa = [q for q in qa if q["task"] in a.tasks.split(",")]
    qa = qa[: a.limit or None]
    ans_path = os.path.join(out, "answers.jsonl")
    done = set()
    if os.path.exists(ans_path):
        done = {json.loads(x)["qid"] for x in open(ans_path, encoding="utf-8") if x.strip()}

    cls, kw = MODELS[a.model]
    t0 = time.perf_counter()
    model = cls(device=a.device, dtype=a.dtype, **kw)
    load_s = time.perf_counter() - t0
    for q in qa[:2]:  # warm-up
        model(os.path.join(ROOT, q["image"]), q)
    n = 0
    with open(ans_path, "a", encoding="utf-8") as f:
        for q in qa:
            if q["qid"] in done:
                continue
            t = time.perf_counter()
            try:
                rec, err = model(os.path.join(ROOT, q["image"]), q), None
            except Exception as e:  # noqa: BLE001
                rec, err = {"raw": None}, repr(e)[:300]
            f.write(json.dumps({"qid": q["qid"], "task": q["task"], **rec,
                                "ms": (time.perf_counter() - t) * 1000, "error": err}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0:
                print(f"{a.model}: {n} answered", flush=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adapters import count_params
    meta = {"model": a.model, **model.info(), "params": count_params(model), "load_s": round(load_s, 2), "answered_this_run": n,
            "finished": datetime.datetime.now().isoformat(timespec="seconds"), "cpu": platform.processor(),
            "os": platform.platform()}
    try:
        import psutil
        mi = psutil.Process().memory_info()
        meta["peak_rss_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)
    except ImportError:
        pass
    if a.device.startswith("cuda"):
        import torch
        meta["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    with open(os.path.join(out, "run_meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
