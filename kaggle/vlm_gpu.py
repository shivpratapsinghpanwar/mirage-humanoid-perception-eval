# Kaggle GPU run for the VLMs too large for the 4 GB laptop GPU.
# Clones the public benchmark repo at a pinned commit, runs the same frozen question set
# (results/qa_set.json) and Moondream's person detect() on COCO-300, and leaves the outputs in
# /kaggle/working/out. Accuracy only: Kaggle latencies are recorded but never compared with laptop ones.
import json
import os
import subprocess
import sys
import time

REPO = "https://github.com/shivpratapsinghpanwar/mirage-humanoid-perception-eval.git"
COMMIT = os.environ.get("BENCH_COMMIT", "__COMMIT__")
BUDGET_S = 3 * 3600  # hard stop for the whole notebook
T0 = time.time()
W = "/kaggle/working"
OUT = f"{W}/out"
os.makedirs(OUT, exist_ok=True)


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


sh(f"git clone -q {REPO} {W}/bench && cd {W}/bench && git checkout -q {COMMIT}")
sh(f"{sys.executable} -m pip install -q accelerate psutil 2>&1 | tail -2")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"{sys.executable} -c 'import transformers, torch; print(transformers.__version__, torch.__version__)'")

# Moondream's remote code may need an older transformers than the other models, so it runs first with
# Kaggle's preinstalled version; on failure it retries once with transformers<5. Then the remaining
# models run on transformers 5.24.0, the version used on the laptop.
# fp32 everywhere (T4/P100 have no native bf16; fp16 can overflow in some VLMs); same as the laptop runs.
MOONDREAM = [("qa", "moondream2", "--dtype fp32"), ("det", "moondream2-detect-person", "--tasks coco,null")]
REST = [("qa", "qwen3-vl-2b", "--dtype fp32"), ("qa", "smolvlm2-2.2b", "--dtype fp32"), ("qa", "lfm2.5-vl-1.6b", "--dtype fp32")]
status = []


def run_jobs(jobs, tag=""):
    ok = True
    for kind, key, extra in jobs:
        left = BUDGET_S - (time.time() - T0)
        if left < 600:
            status.append({"model": key, "status": "skipped: time budget"})
            ok = False
            continue
        script = "16_run_qa.py" if kind == "qa" else "12_run_model.py"
        od = f"{OUT}/{key}/kaggle-gpu" + ("/qa" if kind == "qa" else "")
        t = time.time()
        r = sh(f"cd {W}/bench && timeout {int(left - 300)} {sys.executable} eval/{script} {key} --device cuda {extra} "
               f"--out {od} >> {OUT}/{key}.log 2>&1")
        status.append({"model": key, "kind": kind, "exit": r.returncode, "minutes": round((time.time() - t) / 60, 1),
                       "attempt": tag})
        print(status[-1], flush=True)
        sh(f"tail -5 {OUT}/{key}.log")
        ok &= r.returncode == 0
    return ok


if not run_jobs(MOONDREAM, "preinstalled transformers"):
    sh(f"{sys.executable} -m pip install -q 'transformers>=4.56,<5' 2>&1 | tail -2")
    run_jobs(MOONDREAM, "transformers<5")
sh(f"{sys.executable} -m pip install -q 'transformers==5.24.0' 2>&1 | tail -2")
run_jobs(REST, "transformers 5.24.0")
json.dump({"commit": COMMIT, "jobs": status, "total_minutes": round((time.time() - T0) / 60, 1)},
          open(f"{OUT}/status.json", "w"), indent=1)
sh(f"rm -rf {W}/bench")
print(open(f"{OUT}/status.json").read())
