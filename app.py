"""VeriPixel web app: detect AI-generated images and add / verify invisible watermarks.

    python app.py                      # opens http://127.0.0.1:7860
    python app.py --checkpoint runs/<run>/best.pt --share
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from veripixel.frequency import spectrum_image
from veripixel.inference import FakeImageDetector, find_checkpoint
from veripixel.watermark import DEFAULT_KEY, MIN_SIDE, PAYLOAD_BYTES, STRENGTHS, Watermarker, psnr, ssim

DETECTOR: FakeImageDetector | None = None
WM = Watermarker(DEFAULT_KEY)


# --------------------------------------------------------------------------- callbacks
def detect(img: Image.Image | None):
    if img is None:
        return None, None, None, "Upload an image to analyse."
    img = img.convert("RGB")
    spec = spectrum_image(img)
    wm_note = ""
    if min(img.size) >= MIN_SIDE:
        res = WM.extract(img)
        if res.detected:
            wm_note = f"\n\n🔏 **This image carries a VeriPixel watermark:** `{res.message}`"
    if DETECTOR is None:
        return None, None, spec, ("⚠️ **No trained model found.** Train one first:\n\n"
                                  "`python train.py --data-dir data/demo --backbone small --img-size 64 --epochs 3`"
                                  + wm_note)
    prob, heat = DETECTOR.explain(img)
    label = {"AI-generated": prob, "Real": 1 - prob}
    if prob >= 0.5:
        verdict = f"### 🤖 Likely AI-generated  ({prob:.1%})"
    else:
        verdict = f"### 📷 Likely real  ({1 - prob:.1%})"
    certainty = abs(prob - 0.5) * 2
    if certainty < 0.3:
        verdict += "\n\nLow confidence: the model is unsure about this image."
    verdict += (f"\n\nHeatmap shows the regions that drove this decision. The spectrum shows the image's "
                f"frequency content; grids or bright off-centre dots are typical up-sampling traces of AI generators."
                f"\n\n<sub>Model: {DETECTOR.describe()}</sub>" + wm_note)
    return label, heat, spec, verdict


def add_watermark(img: Image.Image | None, message: str, strength: str):
    if img is None:
        return None, None, "Upload an image first."
    img = img.convert("RGB")
    if min(img.size) < MIN_SIDE:
        return None, None, f"Image must be at least {MIN_SIDE}x{MIN_SIDE} pixels."
    message = (message or "").strip() or "VERIPIX"
    marked = WM.embed(img, message, strength)
    out = Path(tempfile.mkdtemp()) / "watermarked.png"
    marked.save(out)
    stored = message.encode("utf-8")[:PAYLOAD_BYTES].decode("utf-8", errors="ignore")
    info = (f"✅ Watermark embedded: `{stored}`\n\n"
            f"- PSNR: **{psnr(img, marked):.2f} dB** (above 40 dB is invisible to the eye)\n"
            f"- SSIM: **{ssim(img, marked):.4f}** (1.0 = identical)\n\n"
            "Download the **PNG** file. Re-saving as heavy JPEG weakens the mark.")
    return marked, str(out), info


def verify_watermark(img: Image.Image | None):
    if img is None:
        return "Upload an image to check."
    img = img.convert("RGB")
    if min(img.size) < MIN_SIDE:
        return f"Image must be at least {MIN_SIDE}x{MIN_SIDE} pixels."
    res = WM.extract(img)
    if res.detected:
        return (f"### 🔏 Watermark found\n\nMessage: **`{res.message}`**\n\n"
                f"Sync bits matched {res.sync_matches}/16 · confidence {res.confidence:.2f} · strength `{res.strength}`")
    return (f"### ❌ No VeriPixel watermark detected\n\n"
            f"Sync bits matched {res.sync_matches}/16 (needs 14) · confidence {res.confidence:.2f}")


# --------------------------------------------------------------------------- UI
def build_ui() -> gr.Blocks:
    model_line = f"Model loaded: `{DETECTOR.describe()}`" if DETECTOR else "⚠️ No detector model yet (run `train.py`)."
    with gr.Blocks(title="VeriPixel") as demo:
        gr.Markdown("# VeriPixel\n**Detect AI-generated images · Protect your own images with an invisible watermark**\n\n"
                    + model_line)
        with gr.Tab("1 · Detect"):
            with gr.Row():
                with gr.Column():
                    d_in = gr.Image(type="pil", label="Image to check")
                    d_btn = gr.Button("Analyse", variant="primary")
                with gr.Column():
                    d_label = gr.Label(num_top_classes=2, label="Prediction")
                    d_text = gr.Markdown()
            with gr.Row():
                d_heat = gr.Image(label="Grad-CAM: where the model looked")
                d_spec = gr.Image(label="Frequency spectrum (FFT)")
            d_btn.click(detect, d_in, [d_label, d_heat, d_spec, d_text])
        with gr.Tab("2 · Add watermark"):
            with gr.Row():
                with gr.Column():
                    w_in = gr.Image(type="pil", label="Your image")
                    w_msg = gr.Textbox(label=f"Message (max {PAYLOAD_BYTES} characters)", value="VERIPIX", max_lines=1)
                    w_str = gr.Radio(list(STRENGTHS), value="medium", label="Strength (higher = more robust, slightly less invisible)")
                    w_btn = gr.Button("Embed watermark", variant="primary")
                with gr.Column():
                    w_out = gr.Image(label="Watermarked image (looks identical)", format="png")
                    w_file = gr.File(label="Download PNG")
                    w_info = gr.Markdown()
            w_btn.click(add_watermark, [w_in, w_msg, w_str], [w_out, w_file, w_info])
        with gr.Tab("3 · Verify watermark"):
            with gr.Row():
                v_in = gr.Image(type="pil", label="Image to verify")
                v_out = gr.Markdown()
            gr.Button("Verify", variant="primary").click(verify_watermark, v_in, v_out)
        with gr.Tab("About"):
            gr.Markdown(
                "**Detector:** dual-branch network. The RGB branch (CNN) sees content and texture; the "
                "frequency branch sees the FFT spectrum, where up-sampling artifacts of generators "
                "(Stable Diffusion, Midjourney, GANs) show up. Both feature vectors are fused for the decision. "
                "Grad-CAM explains the prediction.\n\n"
                "**Watermark:** DWT → DCT → QIM. A 16-bit sync code plus an 8-character message is hidden in the "
                "low-frequency band of the brightness channel, repeated across the image in a key-dependent order. "
                "It survives JPEG compression, resizing, blur, noise, brightness change and border cropping.\n\n"
                "**Limits:** no detector is perfect. Results depend on training data, and images from unseen "
                "generators or heavily edited images are harder. Treat the output as evidence, not proof."
            )
    return demo


def main():
    global DETECTOR
    p = argparse.ArgumentParser(description="VeriPixel web app")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    ckpt = find_checkpoint(args.checkpoint)
    if ckpt:
        DETECTOR = FakeImageDetector(ckpt, args.device)
        print(f"loaded detector: {ckpt}")
    else:
        print("no detector checkpoint found; the Detect tab will ask you to train a model first")
    build_ui().launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
