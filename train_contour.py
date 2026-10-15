from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cfg_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def outdir(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["output_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_all(cfg: dict[str, Any]) -> None:
    seed = int(cfg["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cfg["deterministic"])
    torch.backends.cudnn.benchmark = bool(cfg["benchmark"] and not cfg["deterministic"])


def read_rows(cfg: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(cfg["dataset_root"])
    metadata = root / cfg["metadata_relpath"]
    with metadata.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"split", "map_id", "image_path", "mask_path"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Invalid TANVOM metadata: {metadata}")
    for row in rows:
        row["image_abs"] = str(root / Path(row["image_path"]))
        row["mask_abs"] = str(root / Path(row["mask_path"]))
        row["tile_id"] = Path(row["image_path"]).stem
    return metadata, rows


def audit(cfg: dict[str, Any]) -> dict[str, Any]:
    metadata, rows = read_rows(cfg)
    missing = []
    image_sizes = Counter()
    mask_sizes = Counter()
    mask_values = Counter()
    ratios = []
    for row in rows:
        image = Path(row["image_abs"])
        mask = Path(row["mask_abs"])
        if not image.exists() or not mask.exists():
            missing.append(row["tile_id"])
            continue
        with Image.open(image) as im:
            image_sizes[f"{im.width}x{im.height}"] += 1
            image_size = im.size
        with Image.open(mask) as mm:
            arr = np.asarray(mm)
            if arr.ndim != 2 or arr.size == 0:
                raise RuntimeError(f"Invalid mask: {mask}")
            if image_size != mm.size:
                raise RuntimeError(f"Image/mask size mismatch: {row['tile_id']}")
            mask_sizes[f"{mm.width}x{mm.height}"] += 1
            ratios.append(float(np.count_nonzero(arr > 0) / arr.size))
            for value in np.unique(arr):
                mask_values[str(int(value))] += 1
    image_paths = [r["image_abs"] for r in rows]
    mask_paths = [r["mask_abs"] for r in rows]
    duplicate_images = [p for p, n in Counter(image_paths).items() if n > 1]
    duplicate_masks = [p for p, n in Counter(mask_paths).items() if n > 1]
    maps = defaultdict(set)
    for row in rows:
        maps[row["split"]].add(row["map_id"])
    leakage = sorted(maps.get("train", set()) & maps.get("val", set()))
    if missing or duplicate_images or duplicate_masks or leakage:
        raise RuntimeError(f"Data audit failed: missing={len(missing)}, duplicate images={len(duplicate_images)}, duplicate masks={len(duplicate_masks)}, map leakage={leakage}")
    a = np.asarray(ratios, dtype=np.float64)
    audit_result = {
        "dataset_root": cfg["dataset_root"],
        "metadata_path": str(metadata),
        "detected_dataset_directory_structure": [str(p) for p in sorted(Path(cfg["dataset_root"]).rglob("*")) if p.is_dir() and p.name in {"train", "val", "images", "masks", "contours", "region_landcover", "region_overlay", "terrain_rare"}],
        "split_source": cfg["train_split_source"],
        "train_sample_count": sum(r["split"] == "train" for r in rows),
        "validation_sample_count": sum(r["split"] == "val" for r in rows),
        "map_ids_by_split": {k: sorted(v) for k, v in maps.items()},
        "missing_image_mask_pairs": missing,
        "duplicated_image_paths": duplicate_images,
        "duplicated_mask_paths": duplicate_masks,
        "image_size_distribution": dict(image_sizes),
        "mask_size_distribution": dict(mask_sizes),
        "mask_foreground_pixel_ratio_statistics": {"min": float(a.min()), "q25": float(np.quantile(a, .25)), "median": float(np.median(a)), "q75": float(np.quantile(a, .75)), "max": float(a.max()), "mean": float(a.mean())},
        "mask_values_seen": dict(mask_values),
        "checks": {"all_pairs_exist": not missing, "all_sizes_match": True, "native_512x512": dict(image_sizes) == {"512x512": len(rows)}, "official_split_used": True, "map_sheet_leakage": leakage},
    }
    (outdir(cfg) / "data_audit.json").write_text(json.dumps(audit_result, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_result


def environment(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    import cv2
    import segmentation_models_pytorch as smp
    result = {
        "python_version": os.sys.version,
        "python_executable": os.sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_mb": int(torch.cuda.get_device_properties(0).total_memory / 1024**2),
        "device": str(device),
        "segmentation_models_pytorch_version": smp.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "mixed_precision": bool(cfg["mixed_precision"]),
    }
    (outdir(cfg) / "environment.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


class ContourDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_abs"]) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(row["mask_abs"]) as mm:
            mask = np.asarray(mm, dtype=np.uint8)
        image = (image - MEAN) / STD
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float(), torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0), row["tile_id"]


def loaders(cfg, rows, batch_size):
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    args = {"num_workers": int(cfg["num_workers"]), "pin_memory": bool(cfg["pin_memory"]), "persistent_workers": False}
    return DataLoader(ContourDataset(train), batch_size=batch_size, shuffle=True, **args), DataLoader(ContourDataset(val), batch_size=batch_size, shuffle=False, **args)


def make_model(cfg):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=cfg["encoder"], encoder_weights=cfg["encoder_weights"], in_channels=int(cfg["in_channels"]), classes=int(cfg["classes"]), activation=None)


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - ((2 * inter + 1e-7) / (denom + 1e-7))).mean()


def counts(logits, target, threshold):
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    return {"tp": int((pred & truth).sum()), "fp": int((pred & ~truth).sum()), "fn": int((~pred & truth).sum()), "tn": int((~pred & ~truth).sum())}


def metrics(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"dice": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1), "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def amp_context(enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def scaler(enabled):
    return torch.amp.GradScaler("cuda", enabled=enabled)


def train_epoch(model, loader, optimizer, grad_scaler, device, cfg):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total = 0.0
    seen = 0
    for image, mask, _ in loader:
        image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(bool(cfg["mixed_precision"])):
            logits = model(image)
            loss = bce(logits, mask) + dice_loss(logits, mask)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        total += float(loss.detach()) * image.shape[0]
        seen += image.shape[0]
    return total / max(seen, 1)


@torch.no_grad()
def validate(model, loader, device, cfg, keep=False):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total = 0.0
    seen = 0
    c = Counter()
    per_tile = []
    kept = {}
    for image, mask, ids in loader:
        image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        with amp_context(bool(cfg["mixed_precision"])):
            logits = model(image)
            loss = bce(logits, mask) + dice_loss(logits, mask)
        total += float(loss) * image.shape[0]
        seen += image.shape[0]
        c.update(counts(logits, mask, float(cfg["threshold"])))
        prob = torch.sigmoid(logits)
        pred = prob >= float(cfg["threshold"])
        for i, tile_id in enumerate(ids):
            one = counts(logits[i:i + 1], mask[i:i + 1], float(cfg["threshold"]))
            mm = metrics(one)
            per_tile.append({"tile_id": tile_id, **mm, "foreground_ratio_gt": float(mask[i].mean()), "foreground_ratio_pred": float(pred[i].float().mean())})
            if keep:
                kept[tile_id] = (image[i].cpu(), mask[i].cpu(), prob[i].cpu())
    result = metrics(c)
    result["loss"] = total / max(seen, 1)
    return result, per_tile, kept


def model_device(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; experiment rules require stopping before full training.")
    device = torch.device("cuda")
    environment(cfg, device)
    return make_model(cfg).to(device), device


def denorm(image):
    arr = image.numpy().transpose(1, 2, 0)
    return (np.clip(arr * STD + MEAN, 0, 1) * 255).astype(np.uint8)


def panel(image, mask, prob, threshold):
    rgb = Image.fromarray(denorm(image), "RGB")
    gt = mask.squeeze().numpy() > .5
    pr = prob.squeeze().numpy()
    binary = pr >= threshold
    overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
    overlay[gt & binary] = [0, 180, 0]
    overlay[~gt & binary] = [220, 30, 30]
    overlay[gt & ~binary] = [30, 80, 220]
    gt_im = Image.fromarray(np.repeat((gt * 255).astype(np.uint8)[..., None], 3, 2))
    prob_im = Image.fromarray(np.repeat((np.clip(pr * 255, 0, 255)).astype(np.uint8)[..., None], 3, 2))
    bin_im = Image.fromarray(np.repeat((binary * 255).astype(np.uint8)[..., None], 3, 2))
    images = [rgb, gt_im, prob_im, bin_im, Image.fromarray(overlay)]
    out = Image.new("RGB", (rgb.width * len(images), rgb.height), "white")
    for i, im in enumerate(images):
        out.paste(im, (i * rgb.width, 0))
    return out


@torch.no_grad()
def save_panels(cfg, model, rows, device, folder, indices):
    folder.mkdir(parents=True, exist_ok=True)
    data = ContourDataset(rows)
    model.eval()
    for index in indices:
        image, mask, tile_id = data[index]
        with amp_context(bool(cfg["mixed_precision"])):
            prob = torch.sigmoid(model(image.unsqueeze(0).to(device))).squeeze(0).cpu()
        panel(image, mask, prob, float(cfg["threshold"])).save(folder / f"{tile_id}.png")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_ckpt(path, model, optimizer, epoch, val, cfg, batch_size, oom):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "validation": val, "config": cfg, "batch_size": batch_size, "oom_occurred": oom}, path)


def smoke(cfg, rows):
    model, device = model_device(cfg)
    train_rows = [r for r in rows if r["split"] == "train"][:int(cfg["smoke_train_samples"])]
    val_rows = [r for r in rows if r["split"] == "val"][:int(cfg["smoke_validation_samples"])]
    train_loader = DataLoader(ContourDataset(train_rows), batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(ContourDataset(val_rows), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    sc = scaler(True)
    oom = False
    started = time.time()
    try:
        train_loss = train_epoch(model, train_loader, optimizer, sc, device, cfg)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        oom = True
        torch.cuda.empty_cache()
        train_loader = DataLoader(ContourDataset(train_rows), batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        train_loss = train_epoch(model, train_loader, optimizer, sc, device, cfg)
    val, _, _ = validate(model, val_loader, device, cfg)
    report = {"status": "passed", "train_samples": len(train_rows), "validation_samples": len(val_rows), "epochs": 1, "train_loss": train_loss, "validation": val, "loss_is_finite": math.isfinite(train_loss) and math.isfinite(val["loss"]), "gpu_used": True, "gpu_name": torch.cuda.get_device_name(0), "oom_occurred": oom, "elapsed_seconds": time.time() - started}
    if not report["loss_is_finite"]:
        raise RuntimeError("Smoke test loss was not finite")
    ckpt = outdir(cfg) / "checkpoints" / "smoke_test_model.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "validation": val, "config": cfg}, ckpt)
    report["checkpoint"] = str(ckpt)
    (outdir(cfg) / "visualizations" / "smoke_test").mkdir(parents=True, exist_ok=True)
    save_panels(cfg, model, val_rows, device, outdir(cfg) / "visualizations" / "smoke_test", list(range(min(5, len(val_rows)))))
    (outdir(cfg) / "smoke_test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def error_report(path, per_tile):
    ordered = sorted(per_tile, key=lambda x: x["dice"])
    low, high = ordered[:20], ordered[-10:][::-1]
    gt = np.asarray([x["foreground_ratio_gt"] for x in per_tile])
    thin = sum(x["foreground_ratio_gt"] < np.quantile(gt, .25) and x["recall"] < .5 for x in per_tile)
    dense = sum(x["foreground_ratio_gt"] >= np.quantile(gt, .75) and x["dice"] < .5 for x in per_tile)
    lines = ["# TANVOM contour baseline-v1 error analysis", "", "本报告仅作像素层面的描述性分析，不把错误自动解释成具体地图符号。", "", f"- validation tiles: {len(per_tile)}", f"- recall < 0.5（疑似漏线）: {sum(x['recall'] < .5 for x in per_tile)}", f"- precision < 0.5（疑似误检）: {sum(x['precision'] < .5 for x in per_tile)}", f"- low-foreground + low-recall（疑似极细线漏检）: {thin}", f"- high-foreground + Dice < 0.5（疑似高密度区域困难）: {dense}", "", "## Best 10", "", "|tile_id|Dice|IoU|Precision|Recall|", "|---|---:|---:|---:|---:|"]
    lines += [f"|{x['tile_id']}|{x['dice']:.4f}|{x['iou']:.4f}|{x['precision']:.4f}|{x['recall']:.4f}|" for x in high]
    lines += ["", "## Worst 20", "", "|tile_id|Dice|IoU|Precision|Recall|", "|---|---:|---:|---:|---:|"]
    lines += [f"|{x['tile_id']}|{x['dice']:.4f}|{x['iou']:.4f}|{x['precision']:.4f}|{x['recall']:.4f}|" for x in low]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return low, high


def train(cfg, rows):
    model, device = model_device(cfg)
    batch_size = int(cfg["batch_size"])
    train_loader, val_loader = loaders(cfg, rows, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    sc = scaler(True)
    logs = []
    best_dice = -1.0
    best_epoch = 0
    oom = False
    started = time.time()
    ckpt_dir = outdir(cfg) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_rows = [r for r in rows if r["split"] == "val"]
    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        while True:
            try:
                torch.cuda.reset_peak_memory_stats()
                train_loss = train_epoch(model, train_loader, optimizer, sc, device, cfg)
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or batch_size == 1:
                    raise
                oom = True
                torch.cuda.empty_cache()
                batch_size = 1
                train_loader, val_loader = loaders(cfg, rows, batch_size)
        val, _, _ = validate(model, val_loader, device, cfg)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "val_dice": val["dice"], "val_iou": val["iou"], "val_precision": val["precision"], "val_recall": val["recall"], "val_f1": val["f1"], "learning_rate": optimizer.param_groups[0]["lr"], "epoch_time_seconds": time.time() - t0, "gpu_memory_peak_mb": int(torch.cuda.max_memory_allocated() / 1024**2)}
        logs.append(row)
        write_csv(outdir(cfg) / "training_log.csv", logs)
        if val["dice"] > best_dice:
            best_dice, best_epoch = val["dice"], epoch
            save_ckpt(ckpt_dir / "best_model.pth", model, optimizer, epoch, val, cfg, batch_size, oom)
        fixed = list(range(min(int(cfg["fixed_visualization_samples"]), len(val_rows))))
        save_panels(cfg, model, val_rows, device, outdir(cfg) / "visualizations" / f"epoch_{epoch:02d}", fixed)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    save_ckpt(ckpt_dir / "last_model.pth", model, optimizer, int(cfg["epochs"]), logs[-1], cfg, batch_size, oom)
    best = torch.load(ckpt_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final_val, per_tile, _ = validate(model, val_loader, device, cfg)
    write_csv(outdir(cfg) / "per_tile_metrics.csv", per_tile)
    low, high = error_report(outdir(cfg) / "error_analysis.md", per_tile)
    by_id = {r["tile_id"]: i for i, r in enumerate(val_rows)}
    save_panels(cfg, model, val_rows, device, outdir(cfg) / "visualizations" / "best_tiles", [by_id[x["tile_id"]] for x in high])
    save_panels(cfg, model, val_rows, device, outdir(cfg) / "visualizations" / "worst_tiles", [by_id[x["tile_id"]] for x in low])
    duration = time.time() - started
    report = {"best_epoch": best_epoch, "best_validation": final_val, "train_samples": len([r for r in rows if r["split"] == "train"]), "validation_samples": len(val_rows), "parameter_count": sum(p.numel() for p in model.parameters()), "batch_size_requested": int(cfg["batch_size"]), "batch_size_final": batch_size, "training_time_seconds": duration, "oom_occurred": oom, "gpu_memory_peak_mb": max(x["gpu_memory_peak_mb"] for x in logs), "hyperparameter_tuning": False}
    md = ["# TANVOM contour segmentation baseline-v1", "", "|指标|结果|", "|---|---:|", f"|训练样本|{report['train_samples']}|", f"|验证样本|{report['validation_samples']}|", f"|模型参数量|{report['parameter_count']:,}|", f"|最佳 epoch|{best_epoch}|", f"|最佳 validation Dice|{final_val['dice']:.6f}|", f"|最佳 validation IoU|{final_val['iou']:.6f}|", f"|Precision|{final_val['precision']:.6f}|", f"|Recall|{final_val['recall']:.6f}|", f"|F1|{final_val['f1']:.6f}|", f"|最终 validation loss|{final_val['loss']:.6f}|", f"|训练时间（秒）|{duration:.1f}|", f"|最终 batch size|{batch_size}|", f"|OOM|{oom}|", f"|GPU 峰值显存（MB）|{report['gpu_memory_peak_mb']}|", "", "本次仅训练合并的 ISOM 101/102 contour geometry。没有进行超参数调优、测试时增强、拓扑后处理或矢量化。", "", "需要先人工检查 visualizations/best_tiles 和 visualizations/worst_tiles，才能决定是否进入 topology/vectorization 阶段。"]
    (outdir(cfg) / "baseline_v1_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (outdir(cfg) / "baseline_v1_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["audit", "smoke", "train", "all"], required=True)
    args = parser.parse_args()
    cfg = cfg_load(Path(args.config))
    seed_all(cfg)
    if args.phase in {"audit", "all"}:
        result = audit(cfg)
        print(json.dumps({"audit": "passed", "train": result["train_sample_count"], "val": result["validation_sample_count"]}, ensure_ascii=False), flush=True)
    _, rows = read_rows(cfg)
    if args.phase in {"smoke", "all"}:
        if not (outdir(cfg) / "data_audit.json").exists():
            audit(cfg)
        result = smoke(cfg, rows)
        print(json.dumps({"smoke": result["status"], "val_dice": result["validation"]["dice"]}, ensure_ascii=False), flush=True)
    if args.phase in {"train", "all"}:
        if not (outdir(cfg) / "smoke_test_report.json").exists():
            raise RuntimeError("Smoke test report missing; run --phase smoke first.")
        result = train(cfg, rows)
        print(json.dumps({"training": "completed", "best_epoch": result["best_epoch"], "val_dice": result["best_validation"]["dice"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
