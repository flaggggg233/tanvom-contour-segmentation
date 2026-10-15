# Publication validation

Validated on 2026-10-15 using the original local Python environment and RTX 3060 Laptop GPU.

## Completed checks

- Python source compilation passed for all published scripts.
- `pip check` reported no broken requirements in the existing experiment environment. A clean dependency installation was not repeated.
- `train_contour.py --help` and the original comparison script's `--help` passed.
- Exported inference-only checkpoint loaded with `torch.load(..., weights_only=True)`.
- `infer.py` ran on a real 512 × 512 validation RGB tile using CUDA; probability, binary mask and overlay outputs retained the input dimensions.
- CPU single-image inference also completed on the same validation tile.
- `evaluate.py` ran on four validation tiles using the exported checkpoint. Each tile's Dice, IoU, precision and recall matched the historical `reports/per_tile_metrics.csv` values within `1e-10`.
- `.gitignore` excludes datasets, generated outputs, local test configurations and checkpoint binaries from Git. Inference weights are distributed as a Release asset instead.

## Scope

The reported 18,880 / 2,664 experiment results come from the completed 20-epoch run and its saved logs. Publication validation did not rerun full training or the entire validation set. The four-tile check verifies the exported checkpoint and evaluation entry point, not an additional benchmark.

The independent O-Ringen experiment has no pixel-level ground truth and provides qualitative evidence only.
