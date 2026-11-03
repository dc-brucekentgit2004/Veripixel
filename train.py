"""Train the AI-image detector.

Examples
    python train.py --data-dir data/demo --backbone small --img-size 64 --epochs 3      # quick CPU test
    python train.py --data-dir data/cifake --backbone efficientnet_b0 --img-size 128 --epochs 8
    python train.py --data-dir data/genimage/sdv14 --max-per-class 20000 --mode fusion
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from veripixel.config import BACKBONES, DEFAULT_CHECKPOINT, MODES, RUNS_DIR, default_workers, get_device, set_seed
from veripixel.data import build_splits, describe, make_loader
from veripixel.metrics import compute_metrics, markdown_table, plot_confusion, plot_history, plot_roc, save_json
from veripixel.models import Detector, count_params, load_detector, save_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the VeriPixel AI-image detector")
    p.add_argument("--data-dir", default="data/demo", help="dataset root (see README -> Datasets)")
    p.add_argument("--mode", default="fusion", choices=MODES, help="rgb, freq or fusion (both)")
    p.add_argument("--backbone", default="efficientnet_b0", choices=BACKBONES)
    p.add_argument("--no-pretrained", action="store_true", help="train RGB backbone from scratch")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=4, help="early stopping patience (epochs)")
    p.add_argument("--max-per-class", type=int, default=None, help="cap training images per class")
    p.add_argument("--max-eval-per-class", type=int, default=None, help="cap val/test images per class")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--no-augment", action="store_true", help="disable JPEG/blur augmentation")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if not set)")
    p.add_argument("--no-set-default", action="store_true",
                   help="don't copy the best model to models/detector.pt (used by the app)")
    return p.parse_args(argv)


def run_epoch(model, loader, device, optimizer=None, scaler=None, desc=""):
    train = optimizer is not None
    model.train(train)
    criterion = nn.BCEWithLogitsLoss()
    losses, ys, ps = [], [], []
    use_amp = device.type == "cuda"
    for x, y in tqdm(loader, desc=desc, leave=False, ncols=90):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        amp = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
        with torch.set_grad_enabled(train), amp:
            logits = model(x)
            loss = criterion(logits.float(), y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(loss.item() * len(y))
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    return sum(losses) / len(y_all), y_all, p_all


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    run_name = args.run_name or f"{Path(args.data_dir).name}_{args.backbone}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== VeriPixel training: {run_name} ===")
    print(f"device: {device} | mode: {args.mode} | backbone: {args.backbone} | img: {args.img_size}px")
    splits = build_splits(args.data_dir, max_per_class=args.max_per_class,
                          max_eval_per_class=args.max_eval_per_class, seed=args.seed)
    print(describe(splits))

    loaders = {
        "train": make_loader(splits["train"], args.img_size, args.batch_size, True, args.workers, not args.no_augment),
        "val": make_loader(splits["val"], args.img_size, args.batch_size, False, args.workers),
    }
    test_samples = splits["test"] or splits["val"]
    loaders["test"] = make_loader(test_samples, args.img_size, args.batch_size, False, args.workers)

    pretrained = not args.no_pretrained
    model = Detector(mode=args.mode, backbone=args.backbone, pretrained=pretrained).to(device)
    print(f"parameters: {count_params(model) / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.param_groups(args.lr, pretrained), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history, best_score, bad_epochs = [], -1.0, 0
    best_path = run_dir / "best.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_y, tr_p = run_epoch(model, loaders["train"], device, optimizer, scaler, f"epoch {epoch} train")
        va_loss, va_y, va_p = run_epoch(model, loaders["val"], device, desc=f"epoch {epoch} val")
        scheduler.step()
        tr_m, va_m = compute_metrics(tr_y, tr_p), compute_metrics(va_y, va_p)
        score = va_m["auc"] if not np.isnan(va_m["auc"]) else va_m["accuracy"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_m["accuracy"], "val_acc": va_m["accuracy"], "val_auc": va_m["auc"]})
        flag = ""
        if score > best_score:
            best_score, bad_epochs, flag = score, 0, "  * best"
            save_checkpoint(best_path, model, args.img_size, epoch=epoch, val_metrics=va_m, args=vars(args))
        else:
            bad_epochs += 1
        print(f"epoch {epoch:>2}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_m['accuracy']:.4f} | "
              f"val loss {va_loss:.4f} acc {va_m['accuracy']:.4f} auc {va_m['auc']:.4f}{flag}")
        if bad_epochs >= args.patience:
            print(f"early stopping (no val improvement for {args.patience} epochs)")
            break

    minutes = (time.time() - t0) / 60
    save_json(history, run_dir / "history.json")
    plot_history(history, run_dir / "training_curves.png")

    # ---- final test with the best checkpoint
    model, ckpt = load_detector(best_path, device)
    _, te_y, te_p = run_epoch(model, loaders["test"], device, desc="test")
    te_m = compute_metrics(te_y, te_p)
    ckpt["test_metrics"] = te_m
    torch.save(ckpt, best_path)
    save_json({"test": te_m, "best_val": ckpt["val_metrics"], "best_epoch": ckpt["epoch"],
               "train_minutes": round(minutes, 2), "args": vars(args)}, run_dir / "metrics.json")
    plot_confusion(te_m["confusion_matrix"], run_dir / "confusion_matrix.png", f"Test - {args.mode}")
    plot_roc(te_y, te_p, run_dir / "roc_curve.png", f"Test ROC - {args.mode}")

    print("\nTest results (best checkpoint, epoch %d):" % ckpt["epoch"])
    print(markdown_table([te_m], ["accuracy", "precision", "recall", "f1", "auc", "real_accuracy", "fake_accuracy", "n"]))
    if not args.no_set_default:
        DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, DEFAULT_CHECKPOINT)
        print(f"\nSaved app model -> {DEFAULT_CHECKPOINT.relative_to(DEFAULT_CHECKPOINT.parent.parent)}")
    print(f"Run folder: runs/{run_name}  (curves, confusion matrix, ROC, metrics.json)  [{minutes:.1f} min]")
    return {"run_dir": str(run_dir), "test": te_m, "mode": args.mode, "backbone": args.backbone, "minutes": minutes}


if __name__ == "__main__":
    main()
