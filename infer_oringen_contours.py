from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = smp.Unet(
        encoder_name=cfg["encoder"],
        encoder_weights=None,
        in_channels=int(cfg["in_channels"]),
        classes=int(cfg["classes"]),
        activation=None,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, {"checkpoint": ckpt, "config": cfg}


def starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    values = list(range(0, length - window + 1, stride))
    last = length - window
    if values[-1] != last:
        values.append(last)
    return values


def normalize(arr: np.ndarray) -> torch.Tensor:
    x = arr.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).float()


@torch.no_grad()
def infer_image(model: torch.nn.Module, image: Image.Image, device: torch.device, amp: bool, window: int, stride: int) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = arr.shape[:2]
    accumulator = np.zeros((height, width), dtype=np.float32)
    visits = np.zeros((height, width), dtype=np.float32)
    padded = width <= window and height <= window
    x_starts = starts(width, window, stride)
    y_starts = starts(height, window, stride)
    for y in y_starts:
        for x in x_starts:
            patch = np.full((window, window, 3), 255, dtype=np.uint8)
            patch_h = min(window, height - y)
            patch_w = min(window, width - x)
            patch[:patch_h, :patch_w] = arr[y:y + patch_h, x:x + patch_w]
            tensor = normalize(patch).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
            accumulator[y:y + patch_h, x:x + patch_w] += probability[:patch_h, :patch_w]
            visits[y:y + patch_h, x:x + patch_w] += 1.0
    return accumulator / np.maximum(visits, 1.0), {
        "window_size": window,
        "stride": stride,
        "overlap_probability_averaging": bool(not padded),
        "padding_mode": f"white_right_bottom_to_{window}" if padded else None,
        "original_width": width,
        "original_height": height,
    }


def make_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    red = Image.new("RGBA", base.size, (255, 40, 40, 0))
    alpha = Image.fromarray(np.where(mask, 155, 0).astype(np.uint8), mode="L")
    red.putalpha(alpha)
    return Image.alpha_composite(base, red).convert("RGB")


def save_pngs(tile_out: Path, image: Image.Image, probability: np.ndarray, threshold: float) -> dict[str, float]:
    mask = probability >= threshold
    Image.fromarray(np.clip(np.round(probability * 255.0), 0, 255).astype(np.uint8), mode="L").save(tile_out / "contour_probability.png")
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(tile_out / "contour_mask.png")
    make_overlay(image, mask).save(tile_out / "contour_overlay.png")
    return {
        "predicted_contour_pixel_count": int(mask.sum()),
        "predicted_contour_fraction": float(mask.mean()),
        "mean_probability": float(probability.mean()),
        "max_probability": float(probability.max()),
    }


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_manifest(path: Path, gpt_root: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    for row in rows:
        tile_dir = gpt_root / row["tile_path"]
        rgb = tile_dir / "rgb.png"
        gpt = tile_dir / "contours_mask.png"
        if not rgb.exists():
            raise FileNotFoundError(f"Fixed GPT-6 RGB tile missing: {rgb}")
        if sha256(rgb) != row["source_sha256"]:
            raise RuntimeError(f"RGB hash mismatch for {row['tile_id']}; refusing to change the comparison set")
        row["rgb_abs"] = str(rgb)
        row["gpt_abs"] = str(gpt) if gpt.exists() else ""
    return rows


def make_review_html(root: Path, rows: list[dict[str, str]], summary: dict[str, Any], blind: bool, blind_a: str) -> str:
    difficulties = sorted({r.get("difficulty", "") for r in rows if r.get("difficulty")})
    cards = []
    for row in rows:
        tile_id = row["tile_id"]
        tile_dir = root / "tiles" / tile_id
        rgb = rel(tile_dir / "rgb.png", root)
        tan_prob = rel(tile_dir / "contour_probability.png", root)
        tan_mask = rel(tile_dir / "contour_mask.png", root)
        tan_overlay = rel(tile_dir / "contour_overlay.png", root)
        gpt = rel(tile_dir / "gpt6_contour.png", root) if (tile_dir / "gpt6_contour.png").exists() else ""
        metrics = json.loads((tile_dir / "metrics.json").read_text(encoding="utf-8"))
        frac, mean_p, ms = metrics["predicted_contour_fraction"], metrics["mean_probability"], metrics["inference_time_ms"]
        diff = html.escape(row.get("difficulty", ""))
        if blind:
            a_path = rel(tile_dir / "method_a.png", root)
            b_path = rel(tile_dir / "method_b.png", root)
            method_html = f"""
              <figure><figcaption>Method A</figcaption><img class="zoomable" src="{a_path}" alt="Method A"></figure>
              <figure><figcaption>Method B</figcaption><img class="zoomable" src="{b_path}" alt="Method B"></figure>
            """
        else:
            gpt_html = f'<figure><figcaption>GPT-6 contour</figcaption><img class="zoomable" src="{gpt}" alt="GPT-6 contour"></figure>' if gpt else '<figure><figcaption>GPT-6 contour</figcaption><div class="unavailable">GPT-6 result unavailable</div></figure>'
            method_html = f"""
              {gpt_html}
              <figure><figcaption>TANVOM probability</figcaption><img class="zoomable" src="{tan_prob}" alt="TANVOM probability"></figure>
              <figure><figcaption>TANVOM binary mask</figcaption><img class="zoomable" src="{tan_mask}" alt="TANVOM binary mask"></figure>
              <figure><figcaption>TANVOM overlay</figcaption><img class="zoomable" src="{tan_overlay}" alt="TANVOM overlay"></figure>
            """
        checks = "" if blind else """
          <div class="checks">
            <label><input type="checkbox"> Missing contour?</label>
            <label><input type="checkbox"> False contour?</label>
            <label><input type="checkbox"> Broken contour?</label>
            <label><input type="checkbox"> Wrong connection?</label>
            <label><input type="checkbox"> Position/width error?</label>
            <label><input type="checkbox"> GPT-6 visibly better?</label>
            <label><input type="checkbox"> TANVOM visibly better?</label>
            <label><input type="checkbox"> Similar?</label>
            <label>Notes <input class="notes" type="text"></label>
          </div>
        """
        cards.append(f"""
        <article class="card" data-tile="{html.escape(tile_id)}" data-difficulty="{diff}">
          <h2>{html.escape(tile_id)}</h2>
          <div class="meta">Difficulty: {diff or 'n/a'} · Predicted contour fraction: {frac:.4f} · Mean probability: {mean_p:.4f} · Inference: {ms:.2f} ms</div>
          <div class="grid"><figure><figcaption>Original RGB</figcaption><img class="zoomable" src="{rgb}" alt="Original RGB"></figure>{method_html}</div>
          {checks}
        </article>
        """)
    title = "O-Ringen Etapp 1 — GPT-6 vs TANVOM Contour Review" if not blind else "O-Ringen Etapp 1 — Blind Contour Review"
    method_note = "TANVOM model: U-Net + ResNet18 · TANVOM task: ISOM 101/102 merged contours · threshold: 0.5" if not blind else "Blind comparison: Method A / Method B labels are intentionally hidden."
    filters = '<button data-filter="all">Show all</button>' + ''.join(f'<button data-filter="{html.escape(d)}">{html.escape(d)}</button>' for d in difficulties)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f4f5f7;color:#17202a}} header{{position:sticky;top:0;z-index:3;background:#17202a;color:white;padding:18px 24px;box-shadow:0 2px 8px #0003}} h1{{margin:0 0 8px;font-size:22px}} .sub{{opacity:.85}} .controls{{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}} button,input[type=text]{{border:1px solid #ccd2d9;border-radius:6px;padding:7px 10px;background:white}} button{{cursor:pointer}} .summary{{padding:18px 24px;background:white;border-bottom:1px solid #ddd}} .cards{{padding:20px 24px}} .card{{background:white;margin:0 0 22px;padding:16px;border-radius:10px;box-shadow:0 1px 5px #0001}} .card h2{{margin:0 0 6px;font-size:18px}} .meta{{color:#566573;font-size:13px;margin-bottom:14px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}} figure{{margin:0;border:1px solid #dfe3e8;border-radius:6px;overflow:hidden;background:#fafafa}} figcaption{{font-weight:600;padding:7px 9px;font-size:13px;background:#eef1f4}} figure img{{display:block;width:100%;height:auto;cursor:zoom-in;image-rendering:auto}} .unavailable{{padding:50px 10px;text-align:center;color:#777}} .checks{{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px;padding-top:12px;border-top:1px solid #e6e9ed;font-size:13px}} .notes{{min-width:220px}} .hidden{{display:none}} #modal{{display:none;position:fixed;inset:0;background:#000d;z-index:10;align-items:center;justify-content:center;padding:20px}} #modal img{{max-width:96vw;max-height:94vh;object-fit:contain;cursor:zoom-out}} #modal .close{{position:absolute;right:22px;top:12px;color:white;font-size:30px;cursor:pointer}}
</style></head><body>
<header><h1>{title}</h1><div class="sub">{method_note}</div><div class="controls"><input id="search" type="text" placeholder="Search tile ID">{filters}</div></header>
<section class="summary"><b>Summary</b> · Total tiles: {summary['total_tiles']} · Mean predicted contour fraction: {summary['mean_predicted_contour_fraction']:.4f} · Mean inference time: {summary['mean_inference_time_ms']:.2f} ms<br><span>No pixel-level ground truth is available for O-Ringen Etapp 1; this page is for blinded/manual comparative review.</span></section>
<main class="cards">{''.join(cards)}</main>
<div id="modal"><span class="close">×</span><img alt="expanded image"></div>
<script>
const cards=[...document.querySelectorAll('.card')]; let active='all';
function apply(){{const q=document.querySelector('#search').value.toLowerCase();cards.forEach(c=>c.classList.toggle('hidden',!(active==='all'||c.dataset.difficulty===active)||!c.dataset.tile.toLowerCase().includes(q)));}}
document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{{active=b.dataset.filter;apply();}}));document.querySelector('#search').addEventListener('input',apply);
const modal=document.querySelector('#modal');const mi=modal.querySelector('img');document.querySelectorAll('.zoomable').forEach(i=>i.addEventListener('click',()=>{{mi.src=i.src;modal.style.display='flex';}}));modal.addEventListener('click',()=>modal.style.display='none');
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpt-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    args = parser.parse_args()
    if args.threshold != 0.5:
        raise RuntimeError("This frozen comparison requires threshold=0.5")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest, args.gpt_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this inference experiment")
    model, loaded = load_checkpoint(args.checkpoint, device)
    ckpt, cfg = loaded["checkpoint"], loaded["config"]
    model_info = {
        "checkpoint_path": str(args.checkpoint),
        "model_architecture": "U-Net",
        "encoder": "ResNet18",
        "encoder_weights": "ImageNet during training; not re-downloaded during inference",
        "training_task": "ISOM 101 + 102 merged contour geometry binary segmentation",
        "best_epoch": int(ckpt["epoch"]),
        "validation_dice": float(ckpt["validation"]["dice"]),
        "validation_iou": float(ckpt["validation"]["iou"]),
        "threshold": args.threshold,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "mixed_precision": True,
        "postprocessing": "none",
        "gpt6_used_during_inference": False,
        "input_preprocessing": "RGB, float conversion, ImageNet normalization; no resize; smaller tiles padded white right/bottom to 512x512",
        "window_size": args.window,
        "stride": args.stride,
    }
    (args.output / "model_info.json").write_text(json.dumps(model_info, indent=2, ensure_ascii=False), encoding="utf-8")
    blind_a = "GPT6" if (int(hashlib.sha256(b"oringen-etapp1-blind-v1").hexdigest()[:8], 16) % 2 == 0) else "TANVOM"
    out_rows = []
    for row in rows:
        tile_id = row["tile_id"]
        tile_out = args.output / "tiles" / tile_id
        tile_out.mkdir(parents=True, exist_ok=True)
        source = Path(row["rgb_abs"])
        image = Image.open(source).convert("RGB")
        shutil.copy2(source, tile_out / "rgb.png")
        if row["gpt_abs"]:
            shutil.copy2(row["gpt_abs"], tile_out / "gpt6_contour.png")
        start = time.perf_counter()
        probability, infer_meta = infer_image(model, image, device, True, args.window, args.stride)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stats = save_pngs(tile_out, image, probability, args.threshold)
        blind_a_source = tile_out / ("gpt6_contour.png" if blind_a == "GPT6" else "contour_mask.png")
        blind_b_source = tile_out / ("contour_mask.png" if blind_a == "GPT6" else "gpt6_contour.png")
        shutil.copy2(blind_a_source, tile_out / "method_a.png")
        shutil.copy2(blind_b_source, tile_out / "method_b.png")
        metrics = {"tile_id": tile_id, "difficulty": row.get("difficulty", ""), "threshold": args.threshold, **stats, "inference_time_ms": float(elapsed_ms), "input_width": image.width, "input_height": image.height, "gpt6_result_available": bool(row["gpt_abs"]), **infer_meta}
        (tile_out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        out_rows.append({**row, **metrics, "rgb_relative": rel(tile_out / "rgb.png", args.output)})
    fieldnames = list(out_rows[0].keys())
    with (args.output / "test_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    summary = {
        "total_tiles": len(out_rows),
        "mean_predicted_contour_fraction": float(np.mean([r["predicted_contour_fraction"] for r in out_rows])),
        "mean_inference_time_ms": float(np.mean([r["inference_time_ms"] for r in out_rows])),
        "total_inference_time_ms": float(np.sum([r["inference_time_ms"] for r in out_rows])),
        "difficulty_counts": {d: sum(r.get("difficulty", "") == d for r in out_rows) for d in sorted({r.get("difficulty", "") for r in out_rows})},
        "ground_truth_available": False,
        "gpt6_results_available": sum(bool(r["gpt_abs"]) for r in out_rows),
        "postprocessing": "none",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["tile_id", "difficulty", "missing_contour", "false_contour", "broken_contour", "wrong_connection", "position_error", "preferred_method", "notes"]
    with (args.output / "review_template.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({"tile_id": r["tile_id"], "difficulty": r.get("difficulty", ""), **{k: "" for k in fields[2:]}})
    (args.output / "blind_mapping.json").write_text(json.dumps({"A": blind_a, "B": "TANVOM" if blind_a == "GPT6" else "GPT6", "fixed_for_entire_experiment": True}, indent=2), encoding="utf-8")
    (args.output / "review.html").write_text(make_review_html(args.output, rows, summary, False, blind_a), encoding="utf-8")
    (args.output / "review_blind.html").write_text(make_review_html(args.output, rows, summary, True, blind_a), encoding="utf-8")
    report = f"""# O-Ringen Etapp 1 TANVOM contour inference

- Checkpoint: {args.checkpoint}
- Model: U-Net + ResNet18; task: ISOM 101/102 merged contours
- Test tiles: {len(rows)}; GPT-6 fixed tile manifest reused: yes
- RGB hash verification: passed for all {len(rows)} tiles
- Input sizes: original tile pixels preserved; all tiles were smaller than 512x512 and padded white on the right/bottom to 512x512 for inference
- Window size / stride: {args.window} / {args.stride}; overlap average used only for larger tiles
- Threshold: {args.threshold}
- Device: CUDA ({torch.cuda.get_device_name(0)}), AMP enabled
- Total inference time: {summary['total_inference_time_ms']:.2f} ms
- Mean inference time per tile: {summary['mean_inference_time_ms']:.2f} ms
- Post-processing: none; raw probability -> fixed 0.5 threshold -> binary mask
- GPT-6 outputs read during inference: no; used only after inference for display
- Pixel-level ground truth: unavailable; no Dice/IoU/precision/recall reported for O-Ringen
- Normal review: review.html
- Blind review: review_blind.html
- Manual review template: review_template.csv

This is a blinded/manual comparative review, not a ground-truth accuracy benchmark.
"""
    (args.output / "test_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary, "model_info": model_info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
