"""Run the trained contour model on one RGB map image without comparison data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from infer_oringen_contours import infer_image, load_checkpoint, save_pngs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.window < 32 or args.window % 32 or not 0 < args.stride <= args.window:
        parser.error("--window must be a positive multiple of 32; 0 < stride <= window")
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    model, _ = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    probability, metadata = infer_image(
        model, image, device, device.type == "cuda", args.window, args.stride
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stats = save_pngs(args.output, image, probability, args.threshold)
    report = {
        "device": str(device),
        "threshold": args.threshold,
        **metadata,
        **stats,
        "ground_truth_available": False,
    }
    (args.output / "inference.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
