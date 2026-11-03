"""One-click demo of the whole pipeline (runs on a normal laptop CPU in a few minutes).

    python quickstart.py

1. builds a small synthetic dataset (data/demo)
2. trains a small fusion detector
3. evaluates it, including robustness to JPEG / resize / blur / noise
4. runs the frequency-spectrum analysis
5. benchmarks the invisible watermark
Then start the app with:  python app.py
"""
from __future__ import annotations

import time

import evaluate
import experiments
import prepare_data
import train
import watermark_cli
from veripixel.config import DATA_DIR


def step(title: str):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    t0 = time.time()
    demo = DATA_DIR / "demo"
    step("1/5  Building demo dataset")
    if (demo / ".complete").exists():
        print(f"{demo} already exists, skipping")
    else:
        prepare_data.main(["demo"])

    step("2/5  Training detector (small CNN, fusion of RGB + frequency branches)")
    train.main(["--data-dir", str(demo), "--backbone", "small", "--mode", "fusion",
                "--img-size", "64", "--epochs", "4", "--batch-size", "32", "--run-name", "quickstart_demo"])

    step("3/5  Evaluation + robustness")
    evaluate.main(["--data-dir", str(demo), "--robustness"])

    step("4/5  Frequency spectrum analysis")
    experiments.main(["spectrum", "--data-dir", str(demo), "--n", "200"])

    step("5/5  Watermark benchmark")
    watermark_cli.main(["benchmark", "--images", str(demo / "test" / "real"), "--limit", "10"])

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Results are in runs/ and outputs/.")
    print("Next:  python app.py   (opens the web demo in your browser)")


if __name__ == "__main__":
    main()
