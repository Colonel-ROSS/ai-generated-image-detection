"""
app/gradio_app_v2.py

Gradio interface for the AI Image Detector (streams_123_trained_prnu).
Three-stream fusion: F1 (PRNU) + F2 (FFT) + F3 (NSS) — 448-dim concatenation.

Model: streams_123_trained_prnu_best.pth
  F1(128) + F2(256) + F3(64) = 448-dim  →  FusionHead(256) → Sigmoid
  Test AUC=0.9685  ACC=91.15%  AP=0.9084  (in-distribution, stored features)
  Threshold: 0.4587 (Youden's J, full-set ROC curve)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

# ──────────────────────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHECKPOINT   = str(PROJECT_ROOT / "models" / "checkpoints" / "streams_123_trained_prnu_best.pth")
_DEFAULT_FEATURES_DIR = str(PROJECT_ROOT / "features_new_3stream")   # stats + fallback encoder lookup
_ENCODER_FALLBACK_DIR = str(PROJECT_ROOT / "features_combined")      # S2/S3 encoder weights

DEVICE   = "cpu"
IMG_SIZE = 224

# Optimal decision threshold (Youden's J on full 80k feature set, 3-stream model)
# At 0.4587: TPR=0.9368, TNR=0.9224, balanced_acc=0.9296, AUC=0.9789
DECISION_THRESHOLD = 0.4587

# ──────────────────────────────────────────────────────────────────────────────
# FusionHead definition (must match train_fusion_3stream.py)
# ──────────────────────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    def __init__(self, in_dim: int = 448, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Cached model state
# ──────────────────────────────────────────────────────────────────────────────

_state: dict = {}


def _find_encoder_w(features_dir: str, subpath: str):
    """Search features_dir then features_combined fallback for encoder weights."""
    for base in [features_dir, _ENCODER_FALLBACK_DIR]:
        p = Path(base) / subpath
        if p.exists():
            return p
    return None


def _load_components(features_dir: str, checkpoint: str) -> dict:
    """Load (and cache) all model components for 3-stream inference."""
    key = f"{features_dir}|{checkpoint}"
    if _state.get("key") == key:
        return _state

    from src.stream1_prnu.denoiser import DnCNNWrapper
    from src.stream1_prnu.encoder import PRNUEncoder
    from src.stream2_fft.encoder import Stream2
    from src.stream3_nss.encoder import NSSEncoder
    from src.stream3_nss.statistics import NSS_DIM

    features_dir_p = Path(features_dir)

    # Stream 1: DnCNN denoiser + PRNUEncoder
    denoiser = DnCNNWrapper(device="cpu")   # auto-loads models/pretrained/dncnn.pth
    prnu_enc = PRNUEncoder(out_dim=128).eval()
    prnu_w = PROJECT_ROOT / "models" / "pretrained" / "prnu_encoder_trained.pth"
    if prnu_w.exists():
        state = torch.load(str(prnu_w), map_location="cpu")
        prnu_enc.load_state_dict(state["encoder_state"])

    # Stream 2: FFT + ResNet-18
    torch.manual_seed(0)
    s2 = Stream2(out_dim=256, spectral_size=IMG_SIZE).eval()
    s2_w = _find_encoder_w(features_dir, "stream2/stream2_encoder.pth")
    if s2_w:
        s2.load_state_dict(torch.load(str(s2_w), map_location="cpu"))

    # Stream 3: NSS + MLP
    torch.manual_seed(0)
    nss = NSSEncoder(in_dim=NSS_DIM, out_dim=64).eval()
    nss_w = _find_encoder_w(features_dir, "stream3/nss_encoder.pth")
    if nss_w:
        nss.load_state_dict(torch.load(str(nss_w), map_location="cpu"))

    # Normalisation stats (must match features used during training)
    feat_mean = feat_std = None
    stats_path = features_dir_p / "feature_stats.npz"
    if stats_path.exists():
        s = np.load(str(stats_path))
        feat_mean = s["mean"].astype(np.float32)
        feat_std  = np.maximum(s["std"].astype(np.float32), 1e-2)

    # FusionHead (448-dim: S1=128, S2=256, S3=64)
    model = FusionHead(in_dim=448)
    ckpt = Path(checkpoint)
    if ckpt.exists():
        state = torch.load(str(ckpt), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    model.eval()

    _state.update({
        "key": key, "denoiser": denoiser, "prnu_enc": prnu_enc,
        "s2": s2, "nss": nss, "model": model,
        "feat_mean": feat_mean, "feat_std": feat_std,
    })
    return _state


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

_NORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _make_verdict_html(prob: float, f1_score: float, f2_score: float, f3_score: float) -> str:
    """Render a probability bar + per-stream confidence breakdown as HTML."""
    is_fake = prob >= DECISION_THRESHOLD
    label = "FAKE" if is_fake else "REAL"
    label_color = "#c1121f" if is_fake else "#2d6a4f"

    if is_fake:
        fake_pct = round(50.0 + (prob - DECISION_THRESHOLD) / (1.0 - DECISION_THRESHOLD) * 50.0)
    else:
        fake_pct = round(prob / DECISION_THRESHOLD * 50.0)
    fake_pct = max(1, min(99, fake_pct))
    real_pct = 100 - fake_pct

    s1_pct = round(f1_score * 100)
    s2_pct = round(f2_score * 100)
    s3_pct = round(f3_score * 100)

    return f"""<div style="font-family:'Segoe UI',Arial,sans-serif;padding:16px 8px;">
  <div style="text-align:center;margin-bottom:14px;">
    <span style="font-size:2.4rem;font-weight:900;color:{label_color};letter-spacing:4px;">{label}</span>
  </div>
  <div style="max-width:600px;height:60px;border-radius:10px;overflow:hidden;display:flex;margin:0 auto;box-shadow:0 2px 10px rgba(0,0,0,0.2);">
    <div style="width:{real_pct}%;background-color:#2d6a4f;"></div>
    <div style="width:{fake_pct}%;background-color:#c1121f;"></div>
  </div>
  <div style="max-width:600px;margin:10px auto 0;display:flex;justify-content:space-between;">
    <span style="color:#2d6a4f;font-weight:700;font-size:1rem;">REAL &nbsp;{real_pct}%</span>
    <span style="color:#888;font-size:0.85rem;">P(fake) = {prob:.4f} &nbsp;·&nbsp; threshold = {DECISION_THRESHOLD}</span>
    <span style="color:#c1121f;font-weight:700;font-size:1rem;">{fake_pct}% &nbsp;FAKE</span>
  </div>
  <div style="max-width:600px;margin:18px auto 0;padding:12px;background:#f8f9fa;border-radius:8px;">
    <div style="font-size:0.85rem;font-weight:600;color:#444;margin-bottom:8px;">Per-stream activation (higher = stronger fake signal)</div>
    <div style="display:flex;align-items:center;margin:4px 0;">
      <span style="width:110px;font-size:0.82rem;color:#555;">S1 PRNU</span>
      <div style="flex:1;height:14px;background:#e0e0e0;border-radius:4px;overflow:hidden;">
        <div style="width:{s1_pct}%;height:100%;background:#6a4c93;"></div>
      </div>
      <span style="width:38px;text-align:right;font-size:0.82rem;color:#555;">{s1_pct}%</span>
    </div>
    <div style="display:flex;align-items:center;margin:4px 0;">
      <span style="width:110px;font-size:0.82rem;color:#555;">S2 FFT</span>
      <div style="flex:1;height:14px;background:#e0e0e0;border-radius:4px;overflow:hidden;">
        <div style="width:{s2_pct}%;height:100%;background:#1982c4;"></div>
      </div>
      <span style="width:38px;text-align:right;font-size:0.82rem;color:#555;">{s2_pct}%</span>
    </div>
    <div style="display:flex;align-items:center;margin:4px 0;">
      <span style="width:110px;font-size:0.82rem;color:#555;">S3 NSS</span>
      <div style="flex:1;height:14px;background:#e0e0e0;border-radius:4px;overflow:hidden;">
        <div style="width:{s3_pct}%;height:100%;background:#ff595e;"></div>
      </div>
      <span style="width:38px;text-align:right;font-size:0.82rem;color:#555;">{s3_pct}%</span>
    </div>
  </div>
</div>"""


@torch.no_grad()
def predict(pil_image: Image.Image,
            checkpoint: str = _DEFAULT_CHECKPOINT,
            features_dir: str = _DEFAULT_FEATURES_DIR):
    """Run 3-stream inference and return detailed results."""
    if pil_image is None:
        return "<p style='color:#888;font-style:italic;'>Upload an image and click Analyse.</p>", {}, None, None

    comps = _load_components(features_dir, checkpoint)
    denoiser = comps["denoiser"]
    prnu_enc = comps["prnu_enc"]
    s2_enc   = comps["s2"]
    nss_enc  = comps["nss"]
    m        = comps["model"]
    feat_mean = comps["feat_mean"]
    feat_std  = comps["feat_std"]

    from src.stream3_nss.statistics import extract_nss_features

    img_rgb = pil_image.convert("RGB")

    # Safety cap: phone photos can be 3000–4000px, which causes >4 GB RAM usage
    # in Stream 3 (NSS operates on the full tensor). Cap before any stream runs.
    import psutil
    _MAX_DIM = 1600
    _w, _h = img_rgb.size
    if max(_w, _h) > _MAX_DIM:
        _scale = _MAX_DIM / max(_w, _h)
        _new_w, _new_h = int(_w * _scale), int(_h * _scale)
        img_rgb = img_rgb.resize((_new_w, _new_h), Image.LANCZOS)
        print(f"[predict] Resized oversized input from {_w}x{_h} to {_new_w}x{_new_h}")
    _avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    if _avail_gb < 1.0:
        print(f"[predict] WARNING: only {_avail_gb:.2f} GB memory available, processing may be slow")

    # Stream 1: PRNU residual → 128-dim (un-normalized 0-1, resize to 128)
    t_raw = TF.to_tensor(img_rgb)
    g     = TF.rgb_to_grayscale(t_raw)
    g_s   = TF.resize(g, [128, 128], interpolation=TF.InterpolationMode.BILINEAR)
    res   = denoiser.noise_residual(g_s)          # [1, 128, 128]
    prnu_img = _get_prnu_image(res)
    f1    = prnu_enc(res.unsqueeze(0)).squeeze(0).numpy()   # [128]

    # Stream 2: FFT + spectral mask → 256-dim (ImageNet normalized, 224×224)
    t_norm = _NORM(img_rgb).unsqueeze(0)
    f2 = s2_enc(t_norm).squeeze(0).numpy()        # [256]

    # Stream 3: NSS statistics → 64-dim (un-normalized [0,1] tensor)
    t_01    = TF.to_tensor(img_rgb)
    raw_nss = extract_nss_features(t_01)          # [18,]
    f3 = nss_enc(torch.from_numpy(raw_nss).unsqueeze(0).float()).squeeze(0).numpy()  # [64]

    # Concatenate S1+S2+S3 and normalise
    feat = np.concatenate([f1, f2, f3]).astype(np.float32)
    if feat_mean is not None:
        feat = (feat - feat_mean) / feat_std

    # FusionHead prediction
    prob = float(m(torch.from_numpy(feat).unsqueeze(0)).squeeze())

    # Per-stream proxy confidence scores (sigmoid of mean activation magnitude, capped 0-1)
    def _stream_score(arr):
        return float(np.clip(np.abs(arr).mean() / 2.0, 0.0, 1.0))

    f1_score = _stream_score(f1)
    f2_score = _stream_score(f2)
    f3_score = _stream_score(f3)

    verdict = _make_verdict_html(prob, f1_score, f2_score, f3_score)

    # NSS feature breakdown
    _nss_names = [
        "psd_slope", "kurtosis", "skewness",
        "gradient_mean", "gradient_std", "gradient_anisotropy", "gradient_max",
        "wavelet_l1_lh", "wavelet_l1_hl", "wavelet_l1_hh",
        "wavelet_l2_lh", "wavelet_l2_hl", "wavelet_l2_hh",
        "wavelet_l3_lh", "wavelet_l3_hl", "wavelet_l3_hh",
        "autocorr_x", "autocorr_y",
    ]
    nss_breakdown = {name: round(float(v), 4)
                     for name, v in zip(_nss_names, raw_nss)}

    fft_img = _get_fft_image(img_rgb)

    return verdict, nss_breakdown, fft_img, prnu_img


def _get_fft_image(pil_image):
    """Compute and return the 2D FFT magnitude spectrum as a PIL image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        gray = np.array(pil_image.convert("L").resize((224, 224)), dtype=np.float32)
        fft  = np.fft.fftshift(np.fft.fft2(gray))
        mag  = np.log(1 + np.abs(fft))

        # Replace the central DC spike with the image median so it stops
        # dominating the color scale (the spike is always ~15 regardless of content)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        mag[cy-1:cy+2, cx-1:cx+2] = np.median(mag)

        vmin = np.percentile(mag, 2)
        vmax = np.percentile(mag, 99.5)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(mag, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title("FFT Magnitude Spectrum", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf).copy()
    except Exception as e:
        print(f"FFT image error: {e}")
        return None


def _get_prnu_image(residual_tensor):
    """Render the PRNU noise residual as a heatmap image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        res_np = residual_tensor.squeeze().detach().cpu().numpy()

        vmin = np.percentile(res_np, 2)
        vmax = np.percentile(res_np, 98)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(res_np, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title("PRNU Noise Residual", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf).copy()
    except Exception as e:
        print(f"PRNU image error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AI Image Detector — MEng Thesis") as demo:
        gr.Markdown("""
# AI Image Detector

Upload an image to analyse whether it is **real** (camera-captured) or **AI-generated**.
""")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                with gr.Accordion("Advanced settings", open=False):
                    ckpt_box = gr.Textbox(
                        value=_DEFAULT_CHECKPOINT, label="FusionHead checkpoint (.pth)")
                    feat_box = gr.Textbox(
                        value=_DEFAULT_FEATURES_DIR, label="Features dir (stats & encoder weights)")
                submit_btn = gr.Button("Analyse Image", variant="primary")

            with gr.Column(scale=2):
                verdict_html = gr.HTML(
                    "<p style='color:#888;font-style:italic;'>Upload an image and click Analyse.</p>")

        fft_out = gr.Image(label="Stream 2: FFT Magnitude Spectrum (per-image)")
        prnu_out = gr.Image(label="Stream 1: PRNU Noise Residual (per-image)")

        nss_json = gr.JSON(label="Stream 3: Natural Scene Statistics (raw 18-dim features)")

        submit_btn.click(
            fn=predict,
            inputs=[image_input, ckpt_box, feat_box],
            outputs=[verdict_html, nss_json, fft_out, prnu_out],
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
