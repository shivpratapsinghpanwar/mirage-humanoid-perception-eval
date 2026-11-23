"""Model adapters for eval/12_run_model.py. Each adapter loads once and maps an image path to
{"objects": [{"name", "score", "bbox": [x1, y1, x2, y2] on a 0-1000 per-axis scale}], ...}.

Imports are lazy so each model only needs its own packages. `info()` returns what run_meta.json
records (versions, weights, threshold, device).
"""
from __future__ import annotations

import importlib.metadata as md

from PIL import Image

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
          "scissors", "teddy bear", "hair drier", "toothbrush"]


def _ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _norm(boxes_xyxy, W, H):
    return [[x1 / W * 1000, y1 / H * 1000, x2 / W * 1000, y2 / H * 1000] for x1, y1, x2, y2 in boxes_xyxy]


def _torch_info(device):
    import torch
    d = {"torch": torch.__version__, "device": device, "torch_threads": torch.get_num_threads()}
    if device.startswith("cuda"):
        d["gpu"] = torch.cuda.get_device_name(0)
    return d


class Mirage:
    """Stock cactus.Mirage().describe (no wrapper), so engine errors surface as run errors."""

    def __init__(self, device="cpu", threshold=None, threads=None, **_):
        import cactus
        if threads:
            cactus.set_threads(int(threads))  # must precede model load (cactus-compute issue #188)
        self.threads = threads
        self.m = cactus.Mirage()

    def info(self):
        from cactus.agent import fetch
        return {"package": "cactus-compute", "version": _ver("cactus-compute"), "engine": fetch.ENGINE_VERSION,
                "weights_rev": fetch.MODEL_REVISIONS[fetch.MIRAGE], "device": "cpu", "threads": self.threads,
                "threshold": "model default"}

    def __call__(self, path):
        return self.m.describe(path)


class Ultralytics:
    def __init__(self, device="cpu", threshold=0.05, weights="yolo26n.pt", **_):
        from ultralytics import YOLO
        self.model, self.device, self.thr, self.weights = YOLO(weights), device, threshold, weights

    def info(self):
        return {"package": "ultralytics", "version": _ver("ultralytics"), "weights": self.weights,
                "threshold": self.thr, **_torch_info(self.device)}

    def __call__(self, path):
        r = self.model.predict(path, verbose=False, device=self.device, conf=self.thr)[0]
        H, W = r.orig_shape
        return {"objects": [{"name": r.names[int(c)], "score": float(s), "bbox": b}
                            for b, c, s in zip(_norm(r.boxes.xyxy.tolist(), W, H), r.boxes.cls.tolist(),
                                               r.boxes.conf.tolist())]}


class RFDETR:
    def __init__(self, device="cpu", threshold=0.05, size="nano", **_):
        import rfdetr
        cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase"}[size]
        self.model = getattr(rfdetr, cls)(device=device)
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES
        except ImportError:
            from rfdetr.assets.coco_classes import COCO_CLASSES
        self.names, self.size, self.device, self.thr = COCO_CLASSES, size, device, threshold

    def info(self):
        return {"package": "rfdetr", "version": _ver("rfdetr"), "size": self.size, "threshold": self.thr,
                **_torch_info(self.device)}

    def __call__(self, path):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        d = self.model.predict(img, threshold=self.thr)
        return {"objects": [{"name": self.names.get(int(c), str(int(c))) if isinstance(self.names, dict)
                             else self.names[int(c)], "score": float(s), "bbox": b}
                            for b, c, s in zip(_norm(d.xyxy.tolist(), W, H), d.class_id.tolist(),
                                               d.confidence.tolist())]}


class HFDetector:
    """transformers object detectors with post_process_object_detection (D-FINE, RT-DETRv2, ...)."""

    def __init__(self, device="cpu", threshold=0.05, repo="ustc-community/dfine-nano-coco", **_):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForObjectDetection.from_pretrained(repo).to(device).eval()
        self.torch, self.repo, self.device, self.thr = torch, repo, device, threshold

    def info(self):
        return {"package": "transformers", "version": _ver("transformers"), "repo": self.repo,
                "threshold": self.thr, **_torch_info(self.device)}

    def __call__(self, path):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        r = self.proc.post_process_object_detection(out, target_sizes=[(H, W)], threshold=self.thr)[0]
        lab = self.model.config.id2label
        return {"objects": [{"name": lab[int(c)], "score": float(s), "bbox": b}
                            for b, c, s in zip(_norm(r["boxes"].tolist(), W, H), r["labels"].tolist(),
                                               r["scores"].tolist())]}


class OWLv2:
    """Open-vocabulary detector prompted with the 80 COCO class names."""

    def __init__(self, device="cpu", threshold=0.05, repo="google/owlv2-base-patch16-ensemble", **_):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(repo)
        self.model = Owlv2ForObjectDetection.from_pretrained(repo).to(device).eval()
        self.torch, self.repo, self.device, self.thr = torch, repo, device, threshold
        self.queries = [f"a photo of a {n}" for n in COCO80]

    def info(self):
        return {"package": "transformers", "version": _ver("transformers"), "repo": self.repo,
                "threshold": self.thr, "prompts": "a photo of a <COCO class name> (80 classes)",
                **_torch_info(self.device)}

    def __call__(self, path):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        inputs = self.proc(text=[self.queries], images=img, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        side = max(W, H)  # OWLv2 pads to a square; boxes come back in padded coordinates
        r = self.proc.image_processor.post_process_object_detection(out, threshold=self.thr,
                                                                    target_sizes=[(side, side)])[0]
        objs = []
        for (x1, y1, x2, y2), c, s in zip(r["boxes"].tolist(), r["labels"].tolist(), r["scores"].tolist()):
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                objs.append({"name": COCO80[int(c)], "score": float(s), "bbox": _norm([(x1, y1, x2, y2)], W, H)[0]})
        return {"objects": objs}


class RTMDet:
    """Official OpenMMLab RTMDet (mmdet 3.3.0, mmcv 2.1.0) with the weights listed in mmdet's metafile."""
    URLS = {"rtmdet_tiny_8xb32-300e_coco": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
            "rtmdet_s_8xb32-300e_coco": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"}

    def __init__(self, device="cpu", threshold=0.05, cfg="rtmdet_tiny_8xb32-300e_coco", **_):
        import os

        import mmdet
        import numpy as np
        from mmdet.apis import inference_detector, init_detector
        path = os.path.join(os.path.dirname(mmdet.__file__), ".mim", "configs", "rtmdet", cfg + ".py")
        self.model = init_detector(path, self.URLS[cfg], device=device)
        self.names = self.model.dataset_meta["classes"]
        self.infer, self.np, self.cfg, self.device, self.thr = inference_detector, np, cfg, device, threshold

    def info(self):
        return {"package": "mmdet", "version": _ver("mmdet"), "mmcv": _ver("mmcv"), "config": self.cfg,
                "weights": self.URLS[self.cfg], "threshold": self.thr, **_torch_info(self.device)}

    def __call__(self, path):
        img = self.np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()  # mmdet expects BGR
        H, W = img.shape[:2]
        p = self.infer(self.model, img).pred_instances
        keep = p.scores >= self.thr
        return {"objects": [{"name": self.names[int(c)], "score": float(s), "bbox": b}
                            for b, c, s in zip(_norm(p.bboxes[keep].tolist(), W, H), p.labels[keep].tolist(),
                                               p.scores[keep].tolist())]}


class Florence2:
    """Florence-2 <OD> task (boxes and labels, no scores) via transformers' native Florence2 classes."""

    def __init__(self, device="cpu", threshold=None, repo="florence-community/Florence-2-base", **_):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.dtype = torch.float32  # fp32 on CPU and GPU, so both passes give the same boxes
        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = Florence2ForConditionalGeneration.from_pretrained(repo, dtype=self.dtype).to(device).eval()
        self.torch, self.repo, self.device = torch, repo, device

    def info(self):
        return {"package": "transformers", "version": _ver("transformers"), "repo": self.repo,
                "revision": getattr(self.model.config, "_commit_hash", None), "task": "<OD>",
                "threshold": "none (Florence-2 returns no scores)", **_torch_info(self.device)}

    def __call__(self, path):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        inputs = self.proc(text="<OD>", images=img, return_tensors="pt").to(self.device, self.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
        txt = self.proc.batch_decode(ids, skip_special_tokens=False)[0]
        r = self.proc.post_process_generation(txt, task="<OD>", image_size=(W, H))["<OD>"]
        return {"objects": [{"name": n, "score": None, "bbox": b}
                            for n, b in zip(r["labels"], _norm(r["bboxes"], W, H))]}


class MoondreamDetect:
    """Moondream 2 open-vocabulary detect(), queried with "person" only (scored on person metrics only)."""

    def __init__(self, device="cpu", threshold=None, repo="vikhyatk/moondream2", revision="2025-06-21",
                 queries=("person",), **_):
        import torch
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, trust_remote_code=True,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32, device_map={"": device}).eval()
        self.torch, self.repo, self.rev, self.device, self.queries = torch, repo, revision, device, queries

    def info(self):
        return {"package": "transformers", "version": _ver("transformers"), "repo": self.repo, "revision": self.rev,
                "queries": list(self.queries), "threshold": "none (detect() returns no scores)",
                **_torch_info(self.device)}

    def __call__(self, path):
        img = Image.open(path).convert("RGB")
        objs = []
        with self.torch.inference_mode():
            for q in self.queries:
                for o in self.model.detect(img, q)["objects"]:
                    objs.append({"name": q, "score": None, "bbox": [o["x_min"] * 1000, o["y_min"] * 1000,
                                                                     o["x_max"] * 1000, o["y_max"] * 1000]})
        return {"objects": objs}


REGISTRY = {
    "moondream2-detect-person": (MoondreamDetect, {}),
    "rtmdet-tiny": (RTMDet, {"cfg": "rtmdet_tiny_8xb32-300e_coco"}),
    "rtmdet-s": (RTMDet, {"cfg": "rtmdet_s_8xb32-300e_coco"}),
    "florence2-base": (Florence2, {"repo": "florence-community/Florence-2-base"}),
    "mirage": (Mirage, {}),
    "yolo26n": (Ultralytics, {"weights": "yolo26n.pt"}),
    "yoloe-26s": (Ultralytics, {"weights": "yoloe-26s-seg.pt"}),
    "rfdetr-nano": (RFDETR, {"size": "nano"}),
    "rfdetr-small": (RFDETR, {"size": "small"}),
    "rfdetr-medium": (RFDETR, {"size": "medium"}),
    "dfine-n": (HFDetector, {"repo": "ustc-community/dfine-nano-coco"}),
    "dfine-s": (HFDetector, {"repo": "ustc-community/dfine-small-coco"}),
    "rtdetrv2-r18": (HFDetector, {"repo": "PekingU/rtdetr_v2_r18vd"}),
    "owlv2-base": (OWLv2, {"repo": "google/owlv2-base-patch16-ensemble"}),
}


def count_params(obj):
    """Parameter count of the first torch module found at obj.model[.model[.model]], else None."""
    for path in ("model", "model.model", "model.model.model"):
        o = obj
        try:
            for a in path.split("."):
                o = getattr(o, a)
            if hasattr(o, "parameters") and callable(o.parameters):
                return int(sum(p.numel() for p in o.parameters()))
        except Exception:  # noqa: BLE001
            continue
    return None


def load(key, device="cpu", threshold=0.05, **kw):
    cls, defaults = REGISTRY[key]
    return cls(device=device, threshold=threshold, **{**defaults, **kw})
