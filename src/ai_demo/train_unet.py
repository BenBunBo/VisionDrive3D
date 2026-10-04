#!/usr/bin/env python3
"""Train a transfer-learned U-Net model for car mask segmentation."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34

SEED = 42
SCENE_IMAGE_RE = re.compile(r"scene_(\d+)\.png$")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SampleItem:
    image_path: Path
    mask_path: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_root(dataset_arg: str, root: Path) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def scene_index_from_name(name: str) -> int:
    match = SCENE_IMAGE_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Unsupported scene filename format: {name}")
    return int(match.group(1))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_runtime_device(requested_device: str) -> str:
    requested = str(requested_device).strip()
    if requested.lower() == "cpu":
        return "cpu"

    wants_cuda = requested.isdigit() or requested.lower().startswith("cuda")
    if not wants_cuda:
        return requested

    if not torch.cuda.is_available():
        print(
            f"[WARN] Requested CUDA device '{requested}' but CUDA is unavailable. "
            "Falling back to CPU."
        )
        return "cpu"

    return requested


def to_torch_device_string(runtime_device: str) -> str:
    if runtime_device == "cpu":
        return "cpu"
    if runtime_device.isdigit():
        return f"cuda:{runtime_device}"
    return runtime_device


def collect_split_samples(dataset_root: Path, split_name: str) -> List[SampleItem]:
    image_dir = dataset_root / "images" / split_name
    masks_dir = dataset_root / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {image_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    samples: List[SampleItem] = []
    for image_path in image_dir.glob("scene_*.png"):
        if not image_path.is_file() or SCENE_IMAGE_RE.fullmatch(image_path.name) is None:
            continue

        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            # Support split symlink names if they ever differ from canonical stem.
            resolved_name = image_path.resolve().name
            mask_path = masks_dir / resolved_name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for image '{image_path.name}' in split '{split_name}'"
            )

        samples.append(SampleItem(image_path=image_path, mask_path=mask_path))

    samples.sort(key=lambda s: scene_index_from_name(s.image_path.name))
    if not samples:
        raise RuntimeError(f"No split images found in: {image_dir}")
    return samples


def rgb_instance_mask_to_binary(mask_bgr: np.ndarray) -> np.ndarray:
    if mask_bgr.ndim != 3 or mask_bgr.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB/BGR mask, got shape {mask_bgr.shape}")

    # Each non-black color is a car instance; convert to a semantic foreground mask.
    fg = np.any(mask_bgr != 0, axis=2).astype(np.float32)
    return fg


class CarMaskDataset(Dataset):
    def __init__(self, samples: Sequence[SampleItem], imgsz: int, augment: bool) -> None:
        self.samples = list(samples)
        self.imgsz = int(imgsz)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

    def _load_mask(self, path: Path) -> np.ndarray:
        mask_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        return rgb_instance_mask_to_binary(mask_bgr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = self._load_image(sample.image_path)
        mask = self._load_mask(sample.mask_path)

        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask[None, ...]).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "stem": sample.image_path.stem,
        }


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = ResNet34_Weights.DEFAULT
            except Exception:
                weights = None

        backbone = None
        if pretrained:
            try:
                backbone = resnet34(weights=weights)
                print("[INFO] Loaded ResNet34 ImageNet pretrained encoder.")
            except Exception as exc:
                print(
                    "[WARN] Could not load pretrained ResNet34 weights. "
                    f"Falling back to random init. ({exc})"
                )
                backbone = resnet34(weights=None)
        else:
            backbone = resnet34(weights=None)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.center = nn.Sequential(
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
        )

        self.dec4 = DecoderBlock(512 + 256, 256)
        self.dec3 = DecoderBlock(256 + 128, 128)
        self.dec2 = DecoderBlock(128 + 64, 64)
        self.dec1 = DecoderBlock(64 + 64, 64)
        self.dec0 = DecoderBlock(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def encoder_parameters(self) -> List[nn.Parameter]:
        groups = [self.stem, self.enc1, self.enc2, self.enc3, self.enc4]
        params: List[nn.Parameter] = []
        for module in groups:
            params.extend(module.parameters())
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)      # /2
        x1 = self.pool(x0)     # /4
        x2 = self.enc1(x1)     # /4
        x3 = self.enc2(x2)     # /8
        x4 = self.enc3(x3)     # /16
        x5 = self.enc4(x4)     # /32

        c = self.center(x5)
        d4 = self.dec4(c, x4)   # /16
        d3 = self.dec3(d4, x3)  # /8
        d2 = self.dec2(d3, x2)  # /4
        d1 = self.dec1(d2, x0)  # /2
        d0 = self.dec0(d1)      # /1
        return self.head(d0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def compute_batch_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_area = pred.sum(dim=(1, 2, 3))
    target_area = target.sum(dim=(1, 2, 3))
    union = pred_area + target_area - intersection

    dice = (2.0 * intersection + eps) / (pred_area + target_area + eps)
    iou = (intersection + eps) / (union + eps)

    return float(dice.mean().item()), float(iou.mean().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    bce_weight: float,
    dice_weight: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    bce_criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            bce = bce_criterion(logits, target)
            d_loss = dice_loss(logits, target)
            loss = bce_weight * bce + dice_weight * d_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dice, iou = compute_batch_metrics(logits.detach(), target)
        running_loss += float(loss.item())
        running_dice += dice
        running_iou += iou
        n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss": running_loss / denom,
        "dice": running_dice / denom,
        "iou": running_iou / denom,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    bce_criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            bce = bce_criterion(logits, target)
            d_loss = dice_loss(logits, target)
            loss = bce_weight * bce + dice_weight * d_loss

        dice, iou = compute_batch_metrics(logits, target)
        running_loss += float(loss.item())
        running_dice += dice
        running_iou += iou
        n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss": running_loss / denom,
        "dice": running_dice / denom,
        "iou": running_iou / denom,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_dice: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "args": vars(args),
    }
    torch.save(payload, path)


def set_encoder_frozen(model: ResNet34UNet, frozen: bool) -> None:
    for param in model.encoder_parameters():
        param.requires_grad = not frozen


@torch.no_grad()
def save_test_visualizations(
    model: nn.Module,
    test_samples: Sequence[SampleItem],
    out_dir: Path,
    imgsz: int,
    device: torch.device,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(test_samples[:6])
    if not selected:
        return

    for sample in selected:
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw_h, raw_w = image_rgb.shape[:2]

        resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (probs > 0.5).astype(np.uint8) * 255

        pred_raw = cv2.resize(pred, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
        overlay = image_rgb.copy()
        overlay[pred_raw > 0] = (0.6 * overlay[pred_raw > 0] + 0.4 * np.array([255, 0, 0])).astype(
            np.uint8
        )

        canvas = np.hstack([image_rgb, overlay])
        out_path = out_dir / f"{sample.image_path.stem}_overlay.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transfer-learned U-Net on VisionDrive3D masks")
    parser.add_argument("--dataset", type=str, default="./output_dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/unet")
    parser.add_argument("--name", type=str, default="unet_resnet34_finetuned")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pretrained-encoder", action="store_true", default=True)
    parser.add_argument("--no-pretrained-encoder", dest="pretrained_encoder", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = project_root()
    dataset_root = resolve_dataset_root(args.dataset, root)

    runtime_device = resolve_runtime_device(args.device)
    device = torch.device(to_torch_device_string(runtime_device))
    use_amp = device.type == "cuda"

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Device: {device}")

    train_samples = collect_split_samples(dataset_root, "train")
    val_samples = collect_split_samples(dataset_root, "val")
    test_samples = collect_split_samples(dataset_root, "test")

    print(
        f"[INFO] Split sizes: {len(train_samples)} train | "
        f"{len(val_samples)} val | {len(test_samples)} test"
    )

    train_dataset = CarMaskDataset(train_samples, imgsz=args.imgsz, augment=True)
    val_dataset = CarMaskDataset(val_samples, imgsz=args.imgsz, augment=False)
    test_dataset = CarMaskDataset(test_samples, imgsz=args.imgsz, augment=False)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = ResNet34UNet(pretrained=args.pretrained_encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = root / args.project / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_val_dice = -1.0
    best_epoch = -1

    start_time = time.time()
    encoder_frozen_prev = None

    for epoch in range(1, args.epochs + 1):
        freeze_encoder = epoch <= args.freeze_encoder_epochs
        if encoder_frozen_prev is None or encoder_frozen_prev != freeze_encoder:
            set_encoder_frozen(model, freeze_encoder)
            state = "frozen" if freeze_encoder else "trainable"
            print(f"[INFO] Encoder is now {state} (epoch {epoch}).")
            encoder_frozen_prev = freeze_encoder

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            use_amp=use_amp,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            use_amp=use_amp,
        )

        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} | "
            f"train_dice={row['train_dice']:.4f} val_dice={row['val_dice']:.4f} | "
            f"train_iou={row['train_iou']:.4f} val_iou={row['val_iou']:.4f} | "
            f"lr={lr:.6f}"
        )

        save_checkpoint(out_dir / "weights" / "last.pt", epoch, model, optimizer, best_val_dice, args)

        if row["val_dice"] > best_val_dice:
            best_val_dice = row["val_dice"]
            best_epoch = epoch
            save_checkpoint(out_dir / "weights" / "best.pt", epoch, model, optimizer, best_val_dice, args)
            print(f"[INFO] New best checkpoint at epoch {epoch} (val_dice={best_val_dice:.4f}).")

    elapsed = time.time() - start_time

    best_ckpt = out_dir / "weights" / "best.pt"
    best_payload = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        use_amp=use_amp,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "device": str(device),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_loss": test_metrics["loss"],
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "elapsed_sec": elapsed,
        "history": history,
    }

    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_test_visualizations(
        model=model,
        test_samples=test_samples,
        out_dir=out_dir / "qualitative",
        imgsz=args.imgsz,
        device=device,
    )

    print(f"[INFO] Training completed in {elapsed / 60.0:.1f} minutes.")
    print(f"[INFO] Best epoch: {best_epoch} | best val dice: {best_val_dice:.4f}")
    print(
        f"[INFO] Test metrics: loss={test_metrics['loss']:.4f}, "
        f"dice={test_metrics['dice']:.4f}, iou={test_metrics['iou']:.4f}"
    )
    print(f"[INFO] Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
