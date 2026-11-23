# Benchmark expansion status

Started 2026-11-24. One model at a time; push after each. Kaggle: only one new notebook of our own; never touch existing sessions, notebooks or datasets.

## Protocol (fixed before any new model)
- COCO-300 = the same 300 val2017 images (`results/coco_mirage/_sample_ids.json`), scored by `eval/05_coco_score.py` against `data/coco_val2017_used.json`. The scorer reproduces the published Mirage/YOLO numbers exactly (checked 2026-11-24).
- Detectors save raw outputs at a low threshold (0.05 where configurable) and are scored at their documented default threshold with `--min-score`; boxes per image and class-agnostic precision are reported beside recall.
- Open-vocabulary detectors are prompted with the 80 COCO class names.
- VLMs answer the frozen question set `results/qa_set.json` (the same images and questions as Mirage's `extract()` tests) with greedy decoding and pinned revisions; prompts and parsing rules are fixed in the runner before running.
- Latency: one model per process, warm median after 2 warm-up runs, end to end from image path; one column per hardware. Kaggle GPU numbers are never compared with laptop numbers.

## Hardware
- Laptop: Intel i5-11320H (4C/8T), 8 GB RAM, NVIDIA RTX 2050 4 GB (driver 529.04), Windows 11.
- Kaggle: GPU notebook (accuracy for models too large for the laptop).
- Laptop GPU needs torch built for CUDA 11.8 (driver 529.04 rejects the cu126 build: cudaErrorDevicesUnavailable). Envs: .venv-mirage (cactus 4.2.7), .venv-gpu (torch 2.7.1+cu118, transformers 5.24.0, rfdetr 1.12.1, ultralytics 8.4.216), .venv-mmdet (py3.11, torch 2.1.0+cu118, mmcv 2.1.0, mmdet 3.3.0).
- Default thresholds (library docs/source): YOLO 0.25, RF-DETR 0.5, D-FINE/RT-DETR (HF) 0.5, RTMDet (DetInferencer) 0.3, OWLv2 (HF) 0.1.
- Kaggle notebook slug for this work: sspanwar/mirage-bench-vlm-gpu (new; script kaggle/vlm_gpu.py). Existing notebooks and running sessions are never touched.

## Models
| Model | Track | Where | Status |
|---|---|---|---|
| Mirage, cactus-compute 4.2.2 (engine 4.1.1) | detector + extract | laptop CPU | done (published 2026-11-23) |
| Mirage, cactus-compute 4.2.7 (engine 4.1.4, same weights) | re-check | laptop CPU | COCO + probes done: 0 errors in 402 stock describe() calls (NaN fixed); recall 34.5% (4.2.2: 34.2%), median 1.27 s (4.2.2: 1.75 s). extract() still broken (dog 30/30, count always 0, string echoes field name). Threads test running |
| YOLO26n | detector | laptop CPU | done |
| RF-DETR | detector | laptop CPU/GPU | pending |
| D-FINE | detector | laptop CPU/GPU | pending |
| RTMDet | detector | laptop / Kaggle | pending |
| OWLv2 | open-vocab detector | laptop GPU | pending |
| Florence-2 | VLM | laptop | pending |
| SmolVLM2 | VLM | laptop | pending |
| LFM2.5-VL-450M | VLM | laptop | pending |
| Moondream 2 | VLM | laptop GPU / Kaggle | pending |
| Qwen2.5-VL-3B (or newer small Qwen-VL) | VLM | Kaggle | pending |

## Then
- Leaderboard, CITATION.cff, results-submission issue template, social preview image.
- Update GitHub README, HF dataset, Kaggle dataset (new version).
- Public posts: HF discussion on the Mirage page; GitHub issue for `extract()` only if it reproduces on 4.2.7 (NaN crash is #201, fixed in 4.2.6); HF post; social drafts.
