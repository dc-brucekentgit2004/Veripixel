"""Research experiments for the report.

    python experiments.py spectrum --data-dir data/cifake     # average FFT spectrum: real vs AI (key figure)
    python experiments.py ablation --data-dir data/cifake --backbone efficientnet_b0 --epochs 5
        -> trains RGB-only, frequency-only and fusion models and compares them
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import train as train_script
from veripixel.config import BACKBONES, OUTPUTS_DIR, default_workers
from veripixel.data import scan
from veripixel.frequency import radial_profile, spectrum_np, to_gray_array
from veripixel.metrics import markdown_table, save_json


def spectrum_analysis(args):
    out = Path(args.out) / "spectrum"
    out.mkdir(parents=True, exist_ok=True)
    samples = scan(args.data_dir)
    if not samples:
        raise SystemExit(f"no labelled images under {args.data_dir}")
    rng = random.Random(0)
    mean_spec = {}
    for label, name in ((0, "Real"), (1, "AI-generated")):
        group = [p for p, y in samples if y == label]
        rng.shuffle(group)
        specs = [spectrum_np(to_gray_array(Image.open(p), args.size))
                 for p in tqdm(group[: args.n], desc=f"spectrum {name}", ncols=90)]
        mean_spec[name] = np.mean(specs, axis=0)
    diff = mean_spec["AI-generated"] - mean_spec["Real"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    vmin = min(s.min() for s in mean_spec.values())
    vmax = max(s.max() for s in mean_spec.values())
    for ax, (name, spec) in zip(axes[:2], mean_spec.items()):
        im = ax.imshow(spec, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"Mean log-spectrum: {name}")
        ax.axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    lim = np.abs(diff).max()
    im = axes[2].imshow(diff, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[2].set_title("Difference (AI - Real)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for name, spec in mean_spec.items():
        axes[3].plot(radial_profile(spec), label=name)
    axes[3].set_xlabel("Frequency (radius)  low -> high")
    axes[3].set_ylabel("Mean log-magnitude")
    axes[3].set_title("Radial energy profile")
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "spectrum_analysis.png", dpi=150)
    plt.close()
    print(f"saved {out / 'spectrum_analysis.png'}")


def ablation(args):
    out = Path(args.out) / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("rgb", "freq", "fusion"):
        argv = ["--data-dir", args.data_dir, "--mode", mode, "--backbone", args.backbone,
                "--epochs", str(args.epochs), "--img-size", str(args.img_size),
                "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                "--run-name", f"ablation_{Path(args.data_dir).name}_{args.backbone}_{mode}", "--no-set-default"]
        if args.max_per_class:
            argv += ["--max-per-class", str(args.max_per_class)]
        if args.no_pretrained:
            argv.append("--no-pretrained")
        res = train_script.main(argv)
        t = res["test"]
        rows.append({"model": {"rgb": "RGB only", "freq": "Frequency only", "fusion": "Fusion (RGB + Freq)"}[mode],
                     "accuracy": t["accuracy"], "f1": t["f1"], "auc": t["auc"],
                     "train_min": round(res["minutes"], 1)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_results.csv", index=False)
    md = markdown_table(rows, ["model", "accuracy", "f1", "auc", "train_min"])
    (out / "ablation_results.md").write_text(md + "\n", encoding="utf-8")
    save_json(rows, out / "ablation_results.json")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    for i, metric in enumerate(["accuracy", "f1", "auc"]):
        ax.bar(x + (i - 1) * 0.25, df[metric], 0.25, label=metric.upper() if metric == "auc" else metric.title())
    ax.set_xticks(x, df.model)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: which branch matters?")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "ablation.png", dpi=150)
    plt.close()
    print("\n" + md)
    print(f"saved to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description="VeriPixel experiments")
    p.add_argument("--out", default=str(OUTPUTS_DIR))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spectrum")
    s.add_argument("--data-dir", default="data/demo")
    s.add_argument("--n", type=int, default=500, help="images per class")
    s.add_argument("--size", type=int, default=128)
    a = sub.add_parser("ablation")
    a.add_argument("--data-dir", default="data/demo")
    a.add_argument("--backbone", default="efficientnet_b0", choices=BACKBONES)
    a.add_argument("--no-pretrained", action="store_true")
    a.add_argument("--epochs", type=int, default=5)
    a.add_argument("--img-size", type=int, default=128)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--max-per-class", type=int, default=None)
    a.add_argument("--workers", type=int, default=default_workers())
    args = p.parse_args(argv)
    {"spectrum": spectrum_analysis, "ablation": ablation}[args.cmd](args)


if __name__ == "__main__":
    main()
