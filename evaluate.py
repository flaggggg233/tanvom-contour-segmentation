"""Recompute pixel-level validation metrics from a trusted model checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from infer_oringen_contours import load_checkpoint
from train_contour import ContourDataset, cfg_load, read_rows, validate, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    cfg = cfg_load(args.config)
    cfg["mixed_precision"] = bool(cfg["mixed_precision"] and name == "cuda")
    _, rows = read_rows(cfg)
    val_rows = [row for row in rows if row["split"] == "val"]
    if not val_rows:
        parser.error("metadata contains no validation samples")
    val_loader = DataLoader(
        ContourDataset(val_rows),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"] and name == "cuda"),
    )
    model, _ = load_checkpoint(args.checkpoint, torch.device(name))
    result, per_tile, _ = validate(model, val_loader, torch.device(name), cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.output / "per_tile_metrics.csv", per_tile)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
