"""
app.py — Streamlit demo: upload a chest X-ray image and get a NORMAL / PNEUMONIA prediction
from the fine-tuned EfficientNet-B0 model. Includes a History tab showing past uploads
from this session.

Usage:
    pip install streamlit torch torchvision pillow numpy
    streamlit run app.py

The app always loads weights from artifacts/effnet_model.pt (see DEFAULT_WEIGHTS_PATH below) —
place effnet_model.pt in an "artifacts/" folder next to this file.

Note: history is kept in Streamlit's session_state, so it lasts for the current browser
session/tab only (it resets on a full page reload or when the server restarts).
"""

import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_WEIGHTS_PATH = "effnet_model.pt"

# Path to a background image — put the actual file here (jpg/png). Leave the file missing
# to fall back to the plain blue gradient background.
BACKGROUND_PATH = "mm.png"


def load_background_css(bg_path: str) -> str:
    """Returns a <style> block that sets the app background to the given image (base64-embedded),
    with a dark blue overlay so text stays readable. Returns "" if the file isn't found, so the
    plain gradient background is used instead."""
    path = Path(bg_path)
    if not path.exists():
        return ""

    ext = path.suffix.lower().lstrip(".")
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"""
    <style>
        html, body, [data-testid="stAppViewContainer"] {{
            background:
                linear-gradient(180deg, rgba(6, 15, 46, 0.88), rgba(6, 15, 46, 0.92)),
                url("data:{mime};base64,{data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
    </style>
    """

CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}

    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 0%, #1e3a8a 0%, #0b1e4d 45%, #060f2e 100%);
    }
    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }
    [data-testid="stHeader"] { background: transparent; }

    .block-container {
        padding-top: 3rem;
        max-width: 880px;
    }

    * { color: #e5edff; }

    .app-hero {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin-bottom: 2rem;
    }
    .app-hero .logo-wrap {
        margin-bottom: 0.6rem;
        filter: drop-shadow(0 4px 14px rgba(59, 130, 246, 0.45));
    }
    .app-hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        background: linear-gradient(90deg, #93c5fd, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-hero p {
        color: #93a5d1;
        font-size: 0.95rem;
        margin: 0;
    }

    div[data-testid="stTabs"] button {
        color: #93a5d1;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff;
    }

    div[data-testid="stFileUploader"] {
        border: 1.5px dashed #3b5aa6;
        border-radius: 16px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(6px);
    }
    div[data-testid="stFileUploader"] section {
        background: transparent;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #b9c6ec !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    .result-card {
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 0.5rem;
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .result-card.pneumonia {
        box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.35) inset;
    }
    .result-card.normal {
        box-shadow: 0 0 0 1px rgba(74, 222, 128, 0.35) inset;
    }
    .result-label {
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: 0.02em;
    }
    .result-label.pneumonia { color: #f87171; }
    .result-label.normal { color: #4ade80; }
    .result-confidence {
        color: #93a5d1;
        font-size: 0.9rem;
    }

    .prob-row {
        margin-bottom: 0.9rem;
    }
    .prob-row-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .prob-row-label.normal { color: #4ade80; }
    .prob-row-label.pneumonia { color: #f87171; }
    .prob-bar-track {
        width: 100%;
        height: 10px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
    }
    .prob-bar-fill.normal {
        background: linear-gradient(90deg, #15803d, #4ade80);
    }
    .prob-bar-fill.pneumonia {
        background: linear-gradient(90deg, #b91c1c, #f87171);
    }

    .disclaimer {
        margin-top: 1.5rem;
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.35);
        color: #fcd34d;
        font-size: 0.85rem;
    }

    div[data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
    }

    .history-card {
        border-radius: 14px;
        padding: 0.6rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .history-label {
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }
    .history-label.pneumonia { color: #f87171; }
    .history-label.normal { color: #4ade80; }
    .history-meta {
        color: #93a5d1;
        font-size: 0.75rem;
    }
</style>
"""


def build_efficientnet(num_classes=2):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_path, device_str="cpu"):
    device = torch.device(device_str)
    model = build_efficientnet()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(model, image: Image.Image, device):
    batch = preprocess(image).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def render_hero():
    st.markdown(
        """
        <div class="app-hero">
            <h1>Chest X-Ray Pneumonia Detector</h1>
            <p>EfficientNet-B0 · Graduation project demo</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detector_tab(model, device):
    uploaded_image = st.file_uploader(
        "Upload a chest X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

        with col2:
            with st.spinner("Analyzing..."):
                probs = predict(model, image, device)
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            css_class = "pneumonia" if pred_label == "PNEUMONIA" else "normal"

            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <div class="result-label {css_class}">{pred_label}</div>
                    <div class="result-confidence">{probs[pred_idx]:.1%} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f"""
                <div class="prob-row">
                    <div class="prob-row-label normal"><span>NORMAL</span><span>{probs[0]:.1%}</span></div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill normal" style="width:{probs[0]*100:.1f}%"></div>
                    </div>
                </div>
                <div class="prob-row">
                    <div class="prob-row-label pneumonia"><span>PNEUMONIA</span><span>{probs[1]:.1%}</span></div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill pneumonia" style="width:{probs[1]*100:.1f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Save this result into session history (dedup by file identity + timestamp bucket).
        file_id = f"{uploaded_image.name}-{uploaded_image.size}"
        if st.session_state.get("last_file_id") != file_id:
            thumb = image.convert("RGB").copy()
            thumb.thumbnail((300, 300))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG")
            st.session_state.history.insert(0, {
                "thumb": buf.getvalue(),
                "label": pred_label,
                "confidence": float(probs[pred_idx]),
                "time": datetime.now().strftime("%H:%M:%S"),
            })
            st.session_state.last_file_id = file_id

        st.markdown(
            """
            <div class="disclaimer">
                ⚠️ This is a graduation-project demo, not a certified diagnostic tool.
                Always consult a qualified radiologist/physician for actual diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload a chest X-ray image above to get a prediction.")


def render_history_tab():
    history = st.session_state.get("history", [])

    if not history:
        st.info("No images analyzed yet in this session — results will appear here.")
        return

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

    cols_per_row = 3
    for row_start in range(0, len(history), cols_per_row):
        row_items = history[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, row_items):
            css_class = "pneumonia" if item["label"] == "PNEUMONIA" else "normal"
            with col:
                st.markdown('<div class="history-card">', unsafe_allow_html=True)
                st.image(item["thumb"], use_container_width=True)
                st.markdown(
                    f"""
                    <div class="history-label {css_class}">{item['label']}</div>
                    <div class="history-meta">{item['confidence']:.1%} · {item['time']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def main():
    st.set_page_config(page_title="Chest X-Ray Pneumonia Detector", page_icon="🫁", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    bg_css = load_background_css(BACKGROUND_PATH)
    if bg_css:
        st.markdown(bg_css, unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    render_hero()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model(DEFAULT_WEIGHTS_PATH, device_str=str(device))
    except FileNotFoundError:
        st.error(
            f"Couldn't find weights at `{DEFAULT_WEIGHTS_PATH}`. "
            "Place `effnet_model.pt` inside an `artifacts/` folder next to this app."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    tab_detect, tab_history = st.tabs(["🔍 Detector", "🕓 History"])
    with tab_detect:
        render_detector_tab(model, device)
    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    main()