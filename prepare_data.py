"""Get data ready.

    python prepare_data.py demo                 # small synthetic dataset -> data/demo (no download, for testing)
    python prepare_data.py cifake               # download CIFAKE from Kaggle -> data/cifake
    python prepare_data.py custom --real-dir my_photos --fake-dir my_ai_images   # -> data/custom
    python prepare_data.py check --data-dir data/cifake                          # show what was found
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

from veripixel.config import DATA_DIR
from veripixel.data import IMG_EXTS, build_splits, describe


# --------------------------------------------------------------------------- demo data
def _scene(rng: np.random.Generator, size: int) -> Image.Image:
    """Random 'photo-like' scene: colour gradient + soft shapes + texture."""
    c1, c2 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    t = np.linspace(0, 1, size)
    angle = rng.uniform(0, np.pi)
    grid = np.cos(angle) * t[None, :] + np.sin(angle) * t[:, None]
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)
    base = (c1 * (1 - grid[..., None]) + c2 * grid[..., None]).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.integers(4, 12)):
        x0, y0 = rng.uniform(-0.2, 1.0, 2) * size
        w, h = rng.uniform(0.1, 0.6, 2) * size
        color = tuple(int(v) for v in rng.uniform(0, 255, 3))
        shape = rng.integers(0, 3)
        box = [x0, y0, x0 + w, y0 + h]
        if shape == 0:
            draw.ellipse(box, fill=color)
        elif shape == 1:
            draw.rectangle(box, fill=color)
        else:
            draw.line(box, fill=color, width=int(rng.integers(1, max(2, size // 16))))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    # multi-scale texture
    arr = np.asarray(img, dtype=np.float32)
    for s in (4, 8, 16):
        n = rng.normal(0, 6, (size // s, size // s, 1)).repeat(s, 0).repeat(s, 1)
        arr += gaussian_filter(n[..., 0], s / 2)[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _real_image(rng, size):
    """'Camera' image: full-resolution scene + sensor noise."""
    arr = np.asarray(_scene(rng, size), dtype=np.float32)
    arr += rng.normal(0, rng.uniform(2, 6), arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _fake_image(rng, size):
    """'Generator' image: produced at half resolution and up-sampled 2x (leaves periodic traces)."""
    small = _scene(rng, size // 2)
    up = np.asarray(small.resize((size, size), Image.NEAREST), dtype=np.float32)
    up = np.asarray(Image.fromarray(up.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32)
    yy, xx = np.indices((size, size))
    up += rng.uniform(1.0, 3.0) * (((yy + xx) % 2) * 2 - 1)[..., None]  # faint checkerboard
    up += rng.normal(0, rng.uniform(0.5, 1.5), up.shape)
    return Image.fromarray(up.clip(0, 255).astype(np.uint8))


def make_demo(out: Path, n_train: int, n_test: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    if out.exists():
        shutil.rmtree(out)
    for split, n in (("train", n_train), ("test", n_test)):
        for label, maker in (("real", _real_image), ("fake", _fake_image)):
            d = out / split / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                maker(rng, size).save(d / f"{label}_{i:05d}.png")
    (out / ".complete").write_text("ok")
    print(f"demo dataset written to {out}  ({n_train}+{n_test} images per class, {size}px)")
    print("NOTE: synthetic images are only for testing that the pipeline runs. Use CIFAKE / GenImage for real results.")


# --------------------------------------------------------------------------- CIFAKE
def download_cifake(out: Path) -> None:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Run:  pip install kagglehub\n"
            "Or download manually from https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images\n"
            f"and unzip it so that you have {out}/train/REAL, {out}/train/FAKE, {out}/test/REAL, {out}/test/FAKE"
        )
    print("downloading CIFAKE (~100 MB) from Kaggle ...")
    path = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
    src = next((p.parent for p in path.rglob("train") if (p / "REAL").exists() or (p / "real").exists()), path)
    print(f"copying {src} -> {out}")
    shutil.copytree(src, out, dirs_exist_ok=True)
    print(describe(build_splits(out)))


# --------------------------------------------------------------------------- custom
def make_custom(real_dir: Path, fake_dir: Path, out: Path) -> None:
    for src, label in ((real_dir, "real"), (fake_dir, "fake")):
        if src is None:
            continue
        dst = out / label
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(Path(src).rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                shutil.copy2(f, dst / f"{n:05d}{f.suffix.lower()}")
                n += 1
        print(f"{label}: copied {n} images -> {dst}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare datasets for VeriPixel")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="synthetic dataset for a quick test")
    d.add_argument("--out", default=str(DATA_DIR / "demo"))
    d.add_argument("--n-train", type=int, default=500)
    d.add_argument("--n-test", type=int, default=120)
    d.add_argument("--size", type=int, default=128)
    d.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("cifake", help="download CIFAKE from Kaggle")
    c.add_argument("--out", default=str(DATA_DIR / "cifake"))
    u = sub.add_parser("custom", help="build a test set from your own images")
    u.add_argument("--real-dir", type=Path)
    u.add_argument("--fake-dir", type=Path)
    u.add_argument("--out", default=str(DATA_DIR / "custom"))
    k = sub.add_parser("check", help="show how many images were found")
    k.add_argument("--data-dir", required=True)
    args = p.parse_args(argv)

    if args.cmd == "demo":
        make_demo(Path(args.out), args.n_train, args.n_test, args.size, args.seed)
    elif args.cmd == "cifake":
        download_cifake(Path(args.out))
    elif args.cmd == "custom":
        make_custom(args.real_dir, args.fake_dir, Path(args.out))
    else:
        print(describe(build_splits(args.data_dir)))


if __name__ == "__main__":
    main()
