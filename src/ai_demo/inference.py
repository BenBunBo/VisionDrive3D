#!/usr/bin/env python3
# Run single-model inference on one scene / folder / video.
#   conda activate vd3d-ai
#   conda run -n vd3d-ai python src/ai_demo/inference.py --model-config src/ai_demo/config/model/yolo26m.yaml --scene 42
#   conda run -n vd3d-ai python src/ai_demo/inference.py --model-config src/ai_demo/config/model/yolo26m.yaml --source output_dataset/images/scene_00042.png
#   conda run -n vd3d-ai python src/ai_demo/inference.py --model-config src/ai_demo/config/model/fasterrcnn.yaml --source ./my_clip.mp4 --save-video
"""Inference for one user-chosen seq; outputs to output_dataset/vid_visualize/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from src.ai_demo.model_zoo import load_yaml, project_root, resolve_ultralytics_weight

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def resolve_sources(source: str | None, scene: int | None, dataset_root: Path) -> tuple[List[Path], bool]:
    """Return (paths, is_video). Video path returns single entry with flag True."""
    if scene is not None:
        p = dataset_root / "images" / f"scene_{scene:05d}.png"
        if not p.exists():
            raise FileNotFoundError(f"Scene image not found: {p}")
        return [p], False
    if not source:
        raise ValueError("Provide --source or --scene.")
    p = Path(source).expanduser()
    if not p.is_absolute():
        p = (project_root() / p).resolve()
    if p.is_file() and p.suffix.lower() in VID_EXTS:
        return [p], True
    if p.is_file():
        return [p], False
    if p.is_dir():
        imgs = sorted([q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS and q.is_file()])
        if not imgs:
            raise RuntimeError(f"No images in dir: {p}")
        return imgs, False
    raise FileNotFoundError(f"Source not found: {p}")


def load_model(model_cfg: dict, device: str):
    framework = str(model_cfg.get("framework", "ultralytics"))
    if framework == "transformers":
        raise ValueError("Use load_transformers() for framework='transformers'.")
    if framework == "torchvision":
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        ckpt = Path(str(model_cfg.get("ckpt_local", "")))
        if not ckpt.is_absolute():
            ckpt = project_root() / ckpt
        if ckpt.exists():
            model = fasterrcnn_resnet50_fpn_v2(weights=None)
            model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
            print(f"[model] FasterRCNN loaded from {ckpt}")
        else:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
            model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
            print("[model] FasterRCNN loaded via torchvision COCO weights (cached by torch hub).")
        dev = torch.device("cuda" if device not in ("cpu",) and torch.cuda.is_available() else "cpu")
        return ("torchvision", model.to(dev).eval(), dev)
    from ultralytics import YOLO
    weight = resolve_ultralytics_weight(model_cfg, project_root())
    model = YOLO(weight)
    return ("ultralytics", model, device)


def load_transformers(model_cfg: dict, device: str):
    import torch
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
    from src.ai_demo.model_zoo import ensure_hf_snapshot, hf_car_label_id
    snap = ensure_hf_snapshot(model_cfg, project_root())
    dev = torch.device("cuda:0" if torch.cuda.is_available() and device != "cpu" else "cpu")
    processor = RTDetrImageProcessor.from_pretrained(str(snap))
    model = RTDetrV2ForObjectDetection.from_pretrained(str(snap)).to(dev).eval()
    model_cfg["coco_car_id"] = hf_car_label_id(model, int(model_cfg.get("coco_car_id", 2)))
    print(f"[model] RT-DETRv2 loaded from {snap} on {dev}")
    return ("transformers", (processor, model, dev), str(dev))


def predict_transformers(bundle, img_bgr, conf_thr: float, car_id: int):
    import torch
    from PIL import Image
    processor, model, dev = bundle
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = model(**inputs)
    h, w = img_bgr.shape[:2]
    results = processor.post_process_object_detection(outputs, threshold=conf_thr,
                                                      target_sizes=[(h, w)])[0]
    dets = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if int(label) != car_id:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.tolist())
        dets.append({"class": "car", "confidence": float(score), "bbox": [x1, y1, x2, y2]})
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_bgr, f"car {float(score):.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img_bgr, dets


def draw_torchvision(img_bgr, out: dict, conf_thr: float, car_id: int):
    import torch
    boxes = out["boxes"].detach().cpu()
    labels = out["labels"].detach().cpu().tolist()
    scores = out["scores"].detach().cpu().tolist()
    dets = []
    for b, l, s in zip(boxes, labels, scores):
        if s < conf_thr or int(l) != car_id:
            continue
        x1, y1, x2, y2 = (int(v) for v in b.tolist())
        dets.append({"class": "car", "confidence": float(s), "bbox": [x1, y1, x2, y2]})
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"car {s:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img_bgr, dets


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inference on 1 user-chosen seq")
    ap.add_argument("--model-config", default="src/ai_demo/config/model/yolo26m.yaml")
    ap.add_argument("--dataset-config", default="src/ai_demo/config/dataset/dataset.yaml")
    ap.add_argument("--source", default=None, help="image file | image dir | video file")
    ap.add_argument("--scene", type=int, default=None, help="scene id, e.g. --scene 42")
    ap.add_argument("--save-dir", default="output_dataset/vid_visualize")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-video", action="store_true", help="also pack dir frames into mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    model_cfg = load_yaml(args.model_config)
    ds_cfg = load_yaml(args.dataset_config)
    dataset_root = (root / str(ds_cfg.get("dataset_root", "output_dataset"))).resolve()
    conf = float(args.conf if args.conf is not None else model_cfg.get("conf", 0.25))
    device = str(args.device or model_cfg.get("device", "0"))

    if str(model_cfg.get("framework", "ultralytics")) == "transformers":
        kind, model, dev = load_transformers(model_cfg, device)
    else:
        kind, model, dev = load_model(model_cfg, device)
    model_tag = Path(args.model_config).stem
    save_dir = Path(args.save_dir).expanduser()
    if not save_dir.is_absolute():
        save_dir = (root / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    paths, is_video = resolve_sources(args.source, args.scene, dataset_root)
    if is_video:
        cap = cv2.VideoCapture(str(paths[0]))
        fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        w, h = int(cap.get(3)), int(cap.get(4))
        out_mp4 = save_dir / f"{paths[0].stem}_{model_tag}.mp4"
        writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if kind == "ultralytics":
                res = model.predict(frame, conf=conf, verbose=False)[0]
                writer.write(res.plot())
            elif kind == "transformers":
                frame, _ = predict_transformers(model, frame, conf, int(model_cfg.get("coco_car_id", 3)))
                writer.write(frame)
            else:
                import torch
                import numpy as np
                t = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
                with torch.no_grad():
                    out = model([t.to(dev)])[0]
                frame, _ = draw_torchvision(frame, out, conf, int(model_cfg.get("coco_car_id", 3)))
                writer.write(frame)
            n += 1
        cap.release()
        writer.release()
        print(f"Wrote {n} frames -> {out_mp4}")
        return

    frames_for_video = []
    for img_path in paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] cannot read {img_path}")
            continue
        if kind == "ultralytics":
            res = model.predict(str(img_path), conf=conf, verbose=False)[0]
            cv2.imwrite(str(save_dir / f"{img_path.stem}_{model_tag}.png"), res.plot())
            dets = [{"class": model.names[int(b.cls)], "confidence": float(b.conf),
                     "bbox": [float(v) for v in b.xyxy[0].tolist()]} for b in res.boxes] if res.boxes is not None else []
        elif kind == "transformers":
            img_bgr, dets = predict_transformers(model, img_bgr, conf, int(model_cfg.get("coco_car_id", 3)))
            cv2.imwrite(str(save_dir / f"{img_path.stem}_{model_tag}.png"), img_bgr)
        else:
            import torch
            t = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                out = model([t.to(dev)])[0]
            img_bgr, dets = draw_torchvision(img_bgr, out, conf, int(model_cfg.get("coco_car_id", 3)))
            cv2.imwrite(str(save_dir / f"{img_path.stem}_{model_tag}.png"), img_bgr)
        (save_dir / f"{img_path.stem}_{model_tag}.json").write_text(json.dumps(dets, indent=2))
        frames_for_video.append(save_dir / f"{img_path.stem}_{model_tag}.png")
        print(f"{img_path.name}: {len(dets)} cars")

    if args.save_video and len(frames_for_video) > 1:
        f0 = cv2.imread(str(frames_for_video[0]))
        h, w = f0.shape[:2]
        out_mp4 = save_dir / f"{Path(paths[0]).parent.name or 'seq'}_{model_tag}.mp4"
        writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for f in frames_for_video:
            writer.write(cv2.imread(str(f)))
        writer.release()
        print(f"Wrote video -> {out_mp4}")
    print(f"Done. Outputs in: {save_dir}")


if __name__ == "__main__":
    main()
