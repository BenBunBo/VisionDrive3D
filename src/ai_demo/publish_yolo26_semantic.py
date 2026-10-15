#!/usr/bin/env python3
"""Publish completed semantic benchmarks, comparison data, and qualitative images.

Run: /home/ml4u/anaconda3/envs/vd3d-ai/bin/python src/ai_demo/publish_yolo26_semantic.py --finetuned .mesh_cache/yolo26_semantic_finetune/runs/yolo26n_sem_50ep_b8/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / ".mesh_cache/benchmark_yolo26_semantic/results.json"
SEM_DATA = ROOT / "docs/assets/data/benchmark_semantic_segmentation.json"
COMPARISON = ROOT / "docs/assets/data/benchmark_yolo26_finetune.json"
QUAL = ROOT / "docs/assets/data/qualitative_yolo26_semantic.json"
RESULTS = ROOT / "docs/assets/data/results.json"
SCENES = ("scene_00025", "scene_00046", "scene_00733")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(BENCH.read_text())
    models = benchmark.get("models", {})
    family = [f"yolo26{x}-sem" for x in "nsmlx"]
    missing = [key for key in family if models.get(key, {}).get("status") != "complete"]
    fine = models.get("yolo26n-sem-finetuned")
    if missing or not fine or fine.get("status") != "complete":
        raise RuntimeError(f"Refusing partial publication; missing semantic results: {missing or ['finetuned']}")
    if any(not (ROOT / "docs/assets/images/qualitative" / f"{prefix}_{scene}.png").is_file()
           for prefix in ("detection_baseline", "detection_finetuned") for scene in SCENES):
        raise FileNotFoundError("Existing detection qualitative images are incomplete")
    source_dir = ROOT / ".mesh_cache/benchmark_yolo26_semantic/qualitative"
    for prefix, model_stem in (("semantic_baseline", "yolo26n-sem"),
                               ("semantic_finetuned", args.finetuned.stem)):
        for scene in SCENES:
            source = source_dir / f"{model_stem}_{scene}.png"
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Missing qualitative image {source}")

    # Preserve the old instance comparison as a local history artifact before
    # replacing its semantic row. Its original image-derived scores are not
    # overwritten by the new semantic task metrics.
    old = json.loads(COMPARISON.read_text())
    archive = ROOT / ".mesh_cache/benchmark_yolo26_semantic/benchmark_yolo26_finetune_instance_archive.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(json.dumps(old, indent=2) + "\n")
    comparison = {"models": {}, "meta": {**old.get("meta", {}), "task": "detection + semantic segmentation"}}
    if "yolo26n" in old.get("models", {}):
        comparison["models"]["yolo26n"] = old["models"]["yolo26n"]

    baseline = models["yolo26n-sem"]
    comparison["models"]["yolo26n-sem"] = {
        "baseline": {
            "task": "semantic", "weights": "yolo26n-sem.pt",
            "mAPbox_50": None, "mAPbox_50_95": None,
            "car_iou": baseline["car_iou"], "car_dice": baseline["car_dice"],
            "miou": baseline["miou"], "checkpoint": baseline["checkpoint"],
        },
        "finetuned": {
            "task": "semantic", "weights": "VisionDrive3D fine-tuned",
            "mAPbox_50": None, "mAPbox_50_95": None,
            "car_iou": fine["car_iou"], "car_dice": fine["car_dice"],
            "miou": fine["miou"], "checkpoint": fine["checkpoint"],
        },
    }

    qualitative = {
        "task": "semantic", "scenes": list(SCENES), "confidence": None,
        "image_size": 640, "nms": None,
        "baseline": baseline["checkpoint"], "finetuned": fine["checkpoint"],
        "images": [f"{kind}_{scene}.png" for kind in ("semantic_baseline", "semantic_finetuned") for scene in SCENES],
    }
    results = json.loads(RESULTS.read_text())
    results.setdefault("meta", {})["qualitative_scenes"] = list(SCENES)
    models_meta = results["meta"].setdefault("qualitative_models", {})
    models_meta["segmentation_baseline"] = baseline["checkpoint"]
    models_meta["segmentation_finetuned"] = fine["checkpoint"]
    results["meta"]["qualitative_semantic_models"] = qualitative

    # Write the data and images only once every dependency has passed checks.
    SEM_DATA.write_text(json.dumps(benchmark, indent=2) + "\n")
    COMPARISON.write_text(json.dumps(comparison, indent=2) + "\n")
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    QUAL.write_text(json.dumps(qualitative, indent=2) + "\n")
    destination = ROOT / "docs/assets/images/qualitative"
    for prefix, model_stem in (("semantic_baseline", "yolo26n-sem"),
                               ("semantic_finetuned", args.finetuned.stem)):
        for scene in SCENES:
            shutil.copy2(source_dir / f"{model_stem}_{scene}.png", destination / f"{prefix}_{scene}.png")
    print(json.dumps({"published_models": family, "comparison_rows": 4,
                      "qualitative_images": len(qualitative["images"])}, indent=2))


if __name__ == "__main__":
    main()
