# O-Ringen Etapp 1 TANVOM contour inference

- Checkpoint: checkpoints/best_model.pth
- Model: U-Net + ResNet18; task: ISOM 101/102 merged contours
- Test tiles: 20; GPT-6 fixed tile manifest reused: yes
- RGB hash verification: passed for all 20 tiles
- Input sizes: original tile pixels preserved; all tiles were smaller than 512x512 and padded white on the right/bottom to 512x512 for inference
- Window size / stride: 512 / 384; overlap average used only for larger tiles
- Threshold: 0.5
- Device: CUDA (NVIDIA GeForce RTX 3060 Laptop GPU), AMP enabled
- Total inference time: 907.97 ms
- Mean inference time per tile: 45.40 ms
- Post-processing: none; raw probability -> fixed 0.5 threshold -> binary mask
- GPT-6 outputs read during inference: no; used only after inference for display
- Pixel-level ground truth: unavailable; no Dice/IoU/precision/recall reported for O-Ringen
- Normal review: review.html
- Blind review: review_blind.html
- Manual review template: review_template.csv

This is a blinded/manual comparative review, not a ground-truth accuracy benchmark.

The HTML pages and external map images listed above are local experiment outputs and are not distributed in this repository. No quantitative accuracy or measured domain gap is claimed without ground truth.
