"""Evaluate a trained detector: test metrics, cross-generator generalisation and robustness.

Examples
    python evaluate.py --data-dir data/demo --robustness
    python evaluate.py --data-dir data/cifake
    # cross-generator: model trained on SD v1.4, tested on other generators
    python evaluate.py --data-dir SD14=data/genimage/sdv14 MJ=data/genimage/midjourney GLIDE=data/genimage/glide
    # your own real-world test set
    python evaluate.py --data-dir data/custom --split all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from veripixel.attacks import DETECTOR_ATTACKS
from veripixel.config import OUTPUTS_DIR, default_workers, get_device
from veripixel.data import eval_samples, make_loader
from veripixel.inference import find_checkpoint
from veripixel.metrics import compute_metrics, markdown_table, plot_confusion, plot_roc, save_json
from veripixel.models import load_detector


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the VeriPixel detector")
    p.add_argument("--checkpoint", default=None, help="default: models/detector.pt or newest run")
    p.add_argument("--data-dir", nargs="+", default=["data/demo"],
                   help="one or more dataset folders, optionally NAME=PATH")
    p.add_argument("--split", default="auto", choices=["auto", "test", "val", "all"],
                   help="auto = test/ (or val/) folder if present, else every image")
    p.add_argument("--robustness", action="store_true", help="also test JPEG / resize / blur / noise")
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--out", default=str(OUTPUTS_DIR / "evaluation"))
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def predict_loader(model, loader, device, desc):
    ys, ps = [], []
    for x, y in tqdm(loader, desc=desc, leave=False, ncols=90):
        ps.append(torch.sigmoid(model(x.to(device)).float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    device = get_device(args.device)
    ckpt_path = find_checkpoint(args.checkpoint)
    if ckpt_path is None:
        raise SystemExit("No trained model found. Train one first:  python train.py --data-dir data/demo")
    model, ckpt = load_detector(ckpt_path, device)
    img_size = ckpt["config"]["img_size"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"model: {ckpt_path}  ({ckpt['config']})")

    attacks = DETECTOR_ATTACKS if args.robustness else {"clean": DETECTOR_ATTACKS["clean"]}
    rows = []
    for spec in args.data_dir:
        name, path = spec.split("=", 1) if "=" in spec else (Path(spec).name, spec)
        samples = eval_samples(path, args.split, args.max_per_class)
        print(f"\n[{name}] {len(samples)} images from {path}")
        for attack_name, fn in attacks.items():
            pre = None if attack_name == "clean" else fn
            loader = make_loader(samples, img_size, args.batch_size, False, args.workers, pre_transform=pre)
            y, p = predict_loader(model, loader, device, f"{name}/{attack_name}")
            m = compute_metrics(y, p)
            rows.append({"dataset": name, "attack": attack_name, **{k: v for k, v in m.items() if k != "confusion_matrix"}})
            print(f"  {attack_name:<12} acc {m['accuracy']:.4f}  auc {m['auc']:.4f}  f1 {m['f1']:.4f}")
            if attack_name == "clean":
                plot_confusion(m["confusion_matrix"], out / f"confusion_{name}.png", f"{name}")
                plot_roc(y, p, out / f"roc_{name}.png", f"ROC - {name}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    cols = ["dataset", "attack", "accuracy", "auc", "f1", "real_accuracy", "fake_accuracy", "n"]
    (out / "results.md").write_text(markdown_table(rows, cols) + "\n", encoding="utf-8")
    save_json({"checkpoint": str(ckpt_path), "config": ckpt["config"], "results": rows}, out / "results.json")

    if args.robustness:
        pivot = df.pivot(index="attack", columns="dataset", values="accuracy").reindex(list(attacks))
        ax = pivot.plot(kind="bar", figsize=(8, 4), rot=30)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("Detector robustness to common image distortions")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "robustness.png", dpi=150)
        plt.close()
    if len(args.data_dir) > 1:
        clean = df[df.attack == "clean"]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.bar(clean.dataset, clean.accuracy, color="#4C72B0")
        for i, v in enumerate(clean.accuracy):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Accuracy")
        ax.set_title("Cross-generator generalisation")
        plt.tight_layout()
        plt.savefig(out / "cross_generator.png", dpi=150)
        plt.close()

    print(f"\nSaved results to {out}  (results.csv, results.md, plots)")
    return df


if __name__ == "__main__":
    main()
