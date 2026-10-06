# Run: conda run -n vd3d-ai python -c "import src.ai_demo.model_zoo"
"""Shared helpers: yaml configs + ckpt resolve (local-first, download fallback)."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is not installed. Run: conda run -n vd3d-ai pip install pyyaml")
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    return dict(yaml.safe_load(p.read_text(encoding="utf-8")))


def ensure_ckpt_local(model_cfg: Dict[str, Any], root: Path | None = None) -> Path:
    """Return local ckpt path; download from ckpt_url if missing.

    Never raises on download failure for ultralytics assets — caller falls back
    to letting ultralytics auto-fetch by asset name.
    """
    root = root or project_root()
    local = Path(str(model_cfg.get("ckpt_local", "")))
    if not local.is_absolute():
        local = root / local
    if local.exists():
        return local
    url = str(model_cfg.get("ckpt_url", "")).strip()
    if url:
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"[ckpt] downloading {url} -> {local}")
            urllib.request.urlretrieve(url, str(local))
            return local
        except Exception as exc:
            print(f"[ckpt WARN] download failed ({exc}); will try framework auto-fetch.")
    return local


def resolve_ultralytics_weight(model_cfg: Dict[str, Any], root: Path | None = None) -> str:
    """Prefer local ckpt; else downloaded file; else asset name for auto-fetch."""
    local = ensure_ckpt_local(model_cfg, root)
    if local.exists():
        return str(local)
    asset = str(model_cfg.get("ultralytics_asset", "")).strip()
    if asset:
        return asset
    return str(local)


def hf_car_label_id(model, fallback: int = 2) -> int:
    """Resolve the 'car' label id from an HF detection model's id2label map."""
    try:
        for key, name in dict(getattr(model.config, "id2label", {}) or {}).items():
            if str(name).lower() == "car":
                return int(key)
    except Exception:
        pass
    return fallback


def ensure_hf_snapshot(model_cfg: Dict[str, Any], root: Path | None = None) -> Path:
    """Return local HF snapshot dir; snapshot_download(hf_repo) into it if missing."""
    from huggingface_hub import snapshot_download
    root = root or project_root()
    local = Path(str(model_cfg.get("ckpt_local", "")))
    if not local.is_absolute():
        local = root / local
    needed = ("config.json",)
    if local.exists() and all((local / n).exists() for n in needed):
        return local
    repo = str(model_cfg.get("hf_repo", "")).strip()
    if not repo:
        raise ValueError("transformers model config missing 'hf_repo'.")
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] snapshot_download {repo} -> {local}")
    snapshot_download(repo_id=repo, local_dir=str(local))
    return local
