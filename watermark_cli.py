"""Invisible watermark tool.

Examples
    python watermark_cli.py embed   --input photo.jpg --output photo_wm.png --message DAKSH
    python watermark_cli.py extract --input photo_wm.png
    python watermark_cli.py benchmark --images data/demo/test/real --limit 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from veripixel.attacks import WATERMARK_ATTACKS
from veripixel.config import OUTPUTS_DIR
from veripixel.data import IMG_EXTS
from veripixel.metrics import markdown_table
from veripixel.watermark import DEFAULT_KEY, MIN_SIDE, STRENGTHS, Watermarker, bit_accuracy, psnr, ssim, text_to_bits


def cmd_embed(args):
    wm = Watermarker(args.key)
    img = Image.open(args.input).convert("RGB")
    out = wm.embed(img, args.message, args.strength)
    out_path = Path(args.output)
    if out_path.suffix.lower() not in {".png", ".bmp", ".tif", ".tiff"}:
        print("note: saving as PNG so the watermark is not damaged by lossy compression")
        out_path = out_path.with_suffix(".png")
    out.save(out_path)
    print(f"watermarked image -> {out_path}")
    print(f"PSNR {psnr(img, out):.2f} dB | SSIM {ssim(img, out):.4f}  (higher = more invisible)")


def cmd_extract(args):
    res = Watermarker(args.key).extract(Image.open(args.input).convert("RGB"))
    if res.detected:
        print(f"WATERMARK FOUND  message: {res.message!r}")
    else:
        print("no VeriPixel watermark found (or wrong key)")
    print(f"sync bits matched {res.sync_matches}/16 | confidence {res.confidence:.3f} | strength {res.strength}")


def _collect_images(paths: list[str], limit: int) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        elif p.exists():
            files.append(p)
    return files[:limit]


def cmd_benchmark(args):
    wm = Watermarker(args.key)
    expected = text_to_bits(args.message)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, example = [], None
    files = _collect_images(args.images, args.limit)
    if not files:
        raise SystemExit("no images found for benchmark")
    for f in tqdm(files, desc="benchmark", ncols=90):
        img = Image.open(f).convert("RGB")
        if min(img.size) < MIN_SIDE:
            img = img.resize((max(MIN_SIDE, img.width), max(MIN_SIDE, img.height)), Image.BICUBIC)
        marked = wm.embed(img, args.message, args.strength)
        if example is None:
            example = (img, marked)
        base = {"image": f.name, "psnr": psnr(img, marked), "ssim": ssim(img, marked)}
        for name, attack in WATERMARK_ATTACKS.items():
            res = wm.extract(attack(marked))
            rows.append({**base, "attack": name, "bit_accuracy": bit_accuracy(expected, res.payload_bits),
                         "detected": int(res.detected), "exact_message": int(res.message == args.message)})
        # false-positive check: original (unmarked) image must NOT be detected
        res = wm.extract(img)
        rows.append({**base, "attack": "unmarked_original", "bit_accuracy": bit_accuracy(expected, res.payload_bits),
                     "detected": int(res.detected), "exact_message": int(res.message == args.message)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "watermark_results.csv", index=False)
    summary = df.groupby("attack", sort=False)[["bit_accuracy", "detected", "exact_message"]].mean().reset_index()
    summary.columns = ["attack", "bit_accuracy", "detection_rate", "message_recovered"]
    quality = {"psnr_db": float(df.psnr.mean()), "ssim": float(df.ssim.mean()), "images": len(files),
               "strength": args.strength}
    md = (f"Images: {len(files)} | strength: {args.strength} | mean PSNR {quality['psnr_db']:.2f} dB | "
          f"mean SSIM {quality['ssim']:.4f}\n\n" + markdown_table(summary.to_dict("records"), list(summary.columns)))
    (out / "watermark_summary.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    print("\n('unmarked_original' should have detection_rate 0: it measures false alarms.)")

    # plots
    fig, ax = plt.subplots(figsize=(8, 3.8))
    s = summary[summary.attack != "unmarked_original"]
    ax.bar(s.attack, s.bit_accuracy, color="#55A868")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Bit accuracy")
    ax.set_title(f"Watermark robustness (PSNR {quality['psnr_db']:.1f} dB)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "watermark_robustness.png", dpi=150)
    plt.close()

    orig, marked = example
    diff = np.abs(np.asarray(marked, np.float32) - np.asarray(orig, np.float32)).mean(2)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, im, t in zip(axes, [orig, marked, diff],
                         ["Original", "Watermarked", "Difference x10 (invisible to the eye)"]):
        ax.imshow(np.clip(im * 10, 0, 255).astype(np.uint8) if t.startswith("Diff") else im,
                  cmap="gray" if t.startswith("Diff") else None)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "watermark_example.png", dpi=150)
    plt.close()
    print(f"saved to {out}")
    return summary, quality


def main(argv=None):
    p = argparse.ArgumentParser(description="VeriPixel invisible watermarking")
    p.add_argument("--key", type=int, default=DEFAULT_KEY, help="secret key (must match for extraction)")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--input", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--message", default="VERIPIX", help="up to 8 characters")
    e.add_argument("--strength", default="medium", choices=list(STRENGTHS))
    x = sub.add_parser("extract")
    x.add_argument("--input", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--images", nargs="+", default=["data/demo/test/real"])
    b.add_argument("--limit", type=int, default=20)
    b.add_argument("--message", default="VERIPIX")
    b.add_argument("--strength", default="medium", choices=list(STRENGTHS))
    b.add_argument("--out", default=str(OUTPUTS_DIR / "watermark"))
    args = p.parse_args(argv)
    return {"embed": cmd_embed, "extract": cmd_extract, "benchmark": cmd_benchmark}[args.cmd](args)


if __name__ == "__main__":
    main()
