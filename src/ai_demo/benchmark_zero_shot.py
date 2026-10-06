#!/usr/bin/env python3
# Zero-shot benchmark: RT-DETRv2-S vs YOLO26m vs Faster R-CNN (all COCO-pretrained, no finetune).
#   conda activate vd3d-ai
#   conda run -n vd3d-ai python src/ai_demo/benchmark_zero_shot.py --models yolo26m rtdetrv2-s fasterrcnn
# Metrics: mAP50, mAP50-95 (pycocotools, car-only), FPS, params, FLOPs.
"""Zero-shot object-detection benchmark on the VisionDrive3D test split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from src.ai_demo.model_zoo import ensure_ckpt_local, load_yaml, project_root, resolve_ultralytics_weight

MODEL_CONFIGS = {
    "yolo26n": "src/ai_demo/config/model/yolo26n.yaml",
    "yolo26s": "src/ai_demo/config/model/yolo26s.yaml",
    "yolo26m": "src/ai_demo/config/model/yolo26m.yaml",
    "yolo26l": "src/ai_demo/config/model/yolo26l.yaml",
    "yolo26x": "src/ai_demo/config/model/yolo26x.yaml",
    "rtdetrv2-s": "src/ai_demo/config/model/rtdetrv2-s.yaml",
    "fasterrcnn": "src/ai_demo/config/model/fasterrcnn.yaml",
}


def read_split_ids(imagesets_path: Path) -> List[str]:
    return [ln.strip() for ln in imagesets_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def yolo_to_abs_xywh(line: str, w: int, h: int) -> List[float]:
    _, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
    aw, ah = bw * w, bh * h
    return [(cx * w) - aw / 2.0, (cy * h) - ah / 2.0, aw, ah]


def build_coco_gt(dataset_root: Path, ids: List[str]) -> tuple[dict, dict]:
    images, annotations = [], []
    ann_id = 1
    for i, sid in enumerate(ids):
        img_path = dataset_root / "images" / f"scene_{sid}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        images.append({"id": i, "file_name": img_path.name, "width": w, "height": h})
        lbl = dataset_root / "labels" / "yolo" / f"scene_{sid}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    x, y, bw, bh = yolo_to_abs_xywh(line, w, h)
                    annotations.append({"id": ann_id, "image_id": i, "category_id": 1,
                                        "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
                    ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "car"}]}, {m["file_name"]: m["id"] for m in images}


def count_params_torch(model) -> int:
    import torch  # noqa: F401
    return sum(p.numel() for p in model.parameters())


def measure_flops(model_desc: str, model, imgsz: int, hf_pixel_mask: bool = False) -> float | None:
    try:
        import torch
        from thop import profile
        target = model.model if hasattr(model, "model") else model
        dev = next(target.parameters()).device
        dummy = torch.randn(1, 3, imgsz, imgsz, device=dev)
        inputs = (dummy, torch.ones(1, imgsz, imgsz, device=dev)) if hf_pixel_mask else (dummy,)
        macs, _ = profile(target, inputs=inputs, verbose=False)
        return float(macs * 2.0)
    except Exception as exc:
        print(f"[FLOPs WARN] {model_desc}: {exc}")
        return None


def predict_ultralytics(model, img_path: Path, conf: float, car_id: int) -> List[dict]:
    res = model.predict(str(img_path), conf=conf, verbose=False)[0]
    out = []
    if res.boxes is None:
        return out
    for b in res.boxes:
        if int(b.cls) != car_id:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        out.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(b.conf)})
    return out


def predict_transformers(bundle, img_path: Path, conf: float, car_id: int) -> List[dict]:
    import torch
    from PIL import Image
    processor, model, dev = bundle
    img_bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = model(**inputs)
    h, w = img_bgr.shape[:2]
    results = processor.post_process_object_detection(outputs, threshold=conf,
                                                      target_sizes=[(h, w)])[0]
    out = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if int(label) != car_id:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        out.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    return out


def predict_torchvision(model, dev, img_path: Path, conf: float, car_id: int) -> List[dict]:
    import torch
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float().to(dev) / 255.0
    with torch.no_grad():
        r = model([t])[0]
    out = []
    for b, l, s in zip(r["boxes"].cpu(), r["labels"].cpu().tolist(), r["scores"].cpu().tolist()):
        if int(l) != car_id or s < conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.tolist())
        out.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return out


def coco_map(gt: dict, preds: List[dict]) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile, json as js
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        js.dump(gt, f)
        gt_path = f.name
    coco = COCO(gt_path)
    if not preds:
        return {"mAP50": 0.0, "mAP50_95": 0.0}
    dt = coco.loadRes(preds)
    ev = COCOeval(coco, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"mAP50": float(ev.stats[1]), "mAP50_95": float(ev.stats[0])}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Zero-shot benchmark (RT-DETRv2-S / YOLO26m / FasterRCNN)")
    ap.add_argument("--dataset-config", default="src/ai_demo/config/dataset/dataset.yaml")
    ap.add_argument("--models", nargs="+", default=["yolo26m", "rtdetrv2-s", "fasterrcnn"],
                    choices=list(MODEL_CONFIGS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--save-dir", default="output_dataset/benchmark_zeroshot")
    ap.add_argument("--qualitative-k", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    ds_cfg = load_yaml(args.dataset_config)
    dataset_root = (root / str(ds_cfg.get("dataset_root", "output_dataset"))).resolve()
    split_file = root / str(ds_cfg.get("imagesets", {}).get(args.split, f"output_dataset/ImageSets/{args.split}.txt"))
    ids = read_split_ids(split_file)
    gt, name_to_id = build_coco_gt(dataset_root, ids)
    print(f"GT: {len(gt['images'])} images, {len(gt['annotations'])} car boxes")

    save_dir = (root / args.save_dir).resolve() if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (root / "output_dataset" / "ckpt").mkdir(parents=True, exist_ok=True)

    results: Dict[str, dict] = {}
    for tag in args.models:
        cfg = load_yaml(MODEL_CONFIGS[tag])
        conf = float(cfg.get("conf", 0.25))
        fw = str(cfg.get("framework", "ultralytics"))
        print(f"\n=== {tag} ({fw}) ===")

        if fw == "transformers":
            import torch
            from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
            from src.ai_demo.model_zoo import ensure_hf_snapshot, hf_car_label_id
            snap = ensure_hf_snapshot(cfg, root)
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            processor = RTDetrImageProcessor.from_pretrained(str(snap))
            hf_model = RTDetrV2ForObjectDetection.from_pretrained(str(snap)).to(dev).eval()
            bundle = (processor, hf_model, dev)
            car_id = hf_car_label_id(hf_model, int(cfg.get("coco_car_id", 2)))
            n_params = count_params_torch(hf_model)
            flops = measure_flops(tag, hf_model, int(cfg.get("imgsz", 640)), hf_pixel_mask=True)
            test_paths = [dataset_root / "images" / f"scene_{s}.png" for s in ids]
            for p in test_paths[:5]:
                predict_transformers(bundle, p, conf, car_id)
            t0 = time.perf_counter()
            preds = []
            for p in test_paths:
                for d in predict_transformers(bundle, p, conf, car_id):
                    preds.append({"image_id": name_to_id[p.name], "category_id": 1,
                                  "bbox": d["bbox"], "score": d["score"]})
            dt = time.perf_counter() - t0
            fps = len(test_paths) / max(dt, 1e-6)
        elif fw == "torchvision":
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            ckpt = ensure_ckpt_local(cfg, root)
            if ckpt.exists():
                model = fasterrcnn_resnet50_fpn_v2(weights=None)
                model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
            else:
                from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
                model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
                torch.save(model.state_dict(), str(ckpt))
                print(f"[ckpt] cached torchvision weights -> {ckpt}")
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(dev).eval()
            car_id, n_params = int(cfg.get("coco_car_id", 3)), count_params_torch(model)
            flops = measure_flops(tag, model, 800)
            # warmup + timed run
            test_paths = [dataset_root / "images" / f"scene_{s}.png" for s in ids]
            for p in test_paths[:5]:
                predict_torchvision(model, dev, p, conf, car_id)
            t0 = time.perf_counter()
            preds = []
            for p in test_paths:
                for d in predict_torchvision(model, dev, p, conf, car_id):
                    preds.append({"image_id": name_to_id[p.name], "category_id": 1,
                                  "bbox": d["bbox"], "score": d["score"]})
            dt = time.perf_counter() - t0
            fps = len(test_paths) / max(dt, 1e-6)
            qual_fn = lambda p, **k: (lambda r: (cv2.imread(str(p)), r))(predict_torchvision(model, dev, p, conf, car_id))
            _ = qual_fn
        else:
            from ultralytics import YOLO
            weight = resolve_ultralytics_weight(cfg, root)
            model = YOLO(weight)
            car_id = int(cfg.get("coco_car_id", 2))
            try:
                n_params = int(model.info(verbose=False)[1]) if isinstance(model.info(verbose=False), (list, tuple)) else count_params_torch(model.model)
            except Exception:
                n_params = count_params_torch(model.model)
            flops = measure_flops(tag, model, int(cfg.get("imgsz", 640)))
            test_paths = [dataset_root / "images" / f"scene_{s}.png" for s in ids]
            for p in test_paths[:5]:
                predict_ultralytics(model, p, conf, car_id)
            t0 = time.perf_counter()
            preds = []
            for p in test_paths:
                for d in predict_ultralytics(model, p, conf, car_id):
                    preds.append({"image_id": name_to_id[p.name], "category_id": 1,
                                  "bbox": d["bbox"], "score": d["score"]})
            dt = time.perf_counter() - t0
            fps = len(test_paths) / max(dt, 1e-6)

        metrics = coco_map(gt, preds)
        results[tag] = {**metrics, "FPS": round(fps, 2),
                        "inference_ms": round(1000.0 / max(fps, 1e-6), 2),
                        "params": n_params, "FLOPs": flops,
                        "n_pred": len(preds), "ckpt": str(cfg.get("ckpt_local", ""))}
        print(f"{tag}: mAP50={metrics['mAP50']:.3f} mAP50-95={metrics['mAP50_95']:.3f} "
              f"FPS={fps:.1f} params={n_params} FLOPs={flops}")

    out = save_dir / "results.json"
    out.write_text(json.dumps({"models": results,
                               "meta": {"split": args.split, "n_images": len(gt["images"]),
                                        "n_gt_boxes": len(gt["annotations"]) }}, indent=2))
    print(f"\nSaved: {out}")
    print("Model | mAP50 | mAP50-95 | FPS | params | FLOPs")
    for tag, m in results.items():
        print(f"{tag} | {m['mAP50']:.3f} | {m['mAP50_95']:.3f} | {m['FPS']} | {m['params']} | {m['FLOPs']}")


if __name__ == "__main__":
    main()
