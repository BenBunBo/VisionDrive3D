#!/usr/bin/env python3
"""Fine-tune YOLO26n-sem on the repaired VisionDrive3D semantic dataset.

Run: /home/ml4u/anaconda3/envs/vd3d-ai/bin/python src/ai_demo/finetune_yolo26_semantic.py
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".mesh_cache/semantic_dataset_v1"
RUNS = ROOT / ".mesh_cache/yolo26_semantic_finetune/runs"
BASE = ROOT / ".mesh_cache/benchmark_yolo26_semantic/checkpoints/yolo26n-sem.pt"
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for YOLO26 semantic fine-tuning")
    for path in (WORK / "dataset.yaml", WORK / "manifest.json", WORK / "masks/train").__iter__():
        if not path.exists():
            raise FileNotFoundError(f"Semantic dataset is incomplete: {path}")
    base_weight = BASE
    if not base_weight.is_file():
        base_weight.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(attempt_download_asset(str(base_weight)))
        if downloaded != base_weight and downloaded.is_file():
            base_weight = downloaded
    if not base_weight.is_file():
        raise FileNotFoundError(f"Could not download pretrained checkpoint {base_weight}")

    batch = args.batch
    while True:
        model = YOLO(str(base_weight))
        try:
            model.train(
                task="semantic", data=str(WORK / "dataset.yaml"),
                epochs=args.epochs, imgsz=640, batch=batch,
                device=args.device, project=str(RUNS), name=f"yolo26n_sem_{args.epochs}ep_b{batch}",
                exist_ok=True, seed=SEED, deterministic=True, val=True,
                optimizer="AdamW", lr0=0.002, warmup_epochs=3,
                rect=False, cache=False, workers=4, amp=True,
                plots=True, patience=100, verbose=True,
            )
            best = Path(model.trainer.best).resolve()
            if not best.is_file():
                raise FileNotFoundError(f"Training did not create best checkpoint: {best}")
            payload = {
                "task": "semantic", "dataset": str((WORK / "dataset.yaml").relative_to(ROOT)),
                "checkpoint": str(best.relative_to(ROOT)), "pretrained": str(base_weight.relative_to(ROOT)),
                "epochs_requested": args.epochs, "epochs_best": int(model.trainer.stopper.best_epoch + 1),
                "seed": SEED, "batch_requested": args.batch, "batch_used": batch,
                "imgsz": 640, "device": torch.cuda.get_device_name(0),
                "metrics": model.trainer.metrics,
            }
            output = best.parent.parent / "semantic_manifest.json"
            output.write_text(json.dumps(payload, indent=2, default=float) + "\n")
            print(json.dumps(payload, indent=2, default=float), flush=True)
            return
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if "out of memory" not in str(exc).lower() or batch <= 1:
                raise
            batch = max(1, batch // 2)
            print(f"CUDA OOM; retrying the run with batch={batch}", flush=True)
        finally:
            if "model" in locals():
                del model
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
