"""
app.py — Streamlit Web Application for Plant Disease Detection
──────────────────────────────────────────────────────────────
Run with:
    streamlit run app.py

Features:
  • Upload a leaf image (JPG/PNG)
  • See the predicted disease class + confidence bar
  • View top-5 prediction probabilities
  • Read disease description, symptoms, causes, prevention & treatment
"""

import os
import sys
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

# ── App must be run from the project root, or we fix the path ─────────────────
sys.path.append(str(Path(__file__).parent))

from config import MODEL_SAVE_PATH, NUM_CLASSES, CLASS_DISPLAY_NAMES
from data_prep.data_loader import get_dataloaders, get_device
from models.resnet9 import ResNet9
from utils.predict import predict_image, load_model, INFER_TRANSFORM

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-left: 5px solid #4caf50;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .disease-box {
        background: linear-gradient(135deg, #fff3e0, #fbe9e7);
        border-left: 5px solid #f44336;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .healthy-box {
        background: linear-gradient(135deg, #e8f5e9, #f0f4c3);
        border-left: 5px solid #66bb6a;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-section h4 {
        color: #1b5e20;
        margin-bottom: 0.5rem;
    }
    .severity-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model & disease info (cached)
# ─────────────────────────────────────────────────────────────────────────────

# @st.cache_resource
# def load_cached_model():
#     device = get_device()
#     if not os.path.exists(MODEL_SAVE_PATH):
#         return None, None, None
#     model = load_model(device)
#     _, _, class_names = get_dataloaders()
#     return model, device, class_names

@st.cache_resource
def load_cached_model():
    if not os.path.exists(MODEL_SAVE_PATH):
        return None, None, None
    model = load_model(device)
    
    # Derive class names from disease_info.json instead of scanning data/train
    with open("disease_info.json") as f:
        disease_info = json.load(f)
    class_names = sorted(disease_info.keys())  # sorted to match ImageFolder order
    
    return model, device, class_names

@st.cache_data
def load_disease_info():
    info_path = os.path.join(os.path.dirname(__file__), "disease_info.json")
    with open(info_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant-under-sun.png", width=80)
    st.title("🌿 Plant Disease\nDetector")
    st.markdown("---")
    st.markdown("### 📋 Supported Plants")
    st.markdown("""
    - 🍎 **Apple** (Scab, Black Rot, Healthy)
    - 🌽 **Corn** (Common Rust, Healthy)
    - 🍇 **Grape** (Black Rot, Healthy)
    - 🥔 **Potato** (Early Blight, Late Blight, Healthy)
    - 🍅 **Tomato** (Bacterial Spot, Early Blight, Late Blight, Healthy)
    """)
    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.markdown("""
    - **Architecture**: EfficientNet-B0
    - **Classes**: 14
    - **Input Size**: 224×224
    """)
    st.markdown("---")
    st.caption("Built with PyTorch & Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a leaf image to detect diseases using deep learning (ResNet9)</div>', unsafe_allow_html=True)

# Load resources
model, device, class_names = load_cached_model()
disease_info = load_disease_info()

# Model not trained yet warning
if model is None:
    st.warning(
        "⚠️ Model not found! Please train the model first:\n"
        "```bash\npython train_model.py\n```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Upload section
# ─────────────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG/PNG image of a plant leaf",
        type=["jpg", "jpeg", "png"],
        help="Supported crops: Apple, Corn, Grape, Potato, Tomato"
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        predict_btn = st.button("🔍 Detect Disease", type="primary", use_container_width=True)

with col2:
    if uploaded_file and predict_btn:
        st.markdown("### 🔬 Detection Results")

        # Save temp file for predict_image()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        with st.spinner("Analysing leaf …"):
            best_label, confidence, top5 = predict_image(
                tmp_path, model=model, device=device,
                class_names=class_names, top_k=5
            )
        os.unlink(tmp_path)

        # Determine healthy vs diseased
        is_healthy = "Healthy" in best_label
        box_class  = "healthy-box" if is_healthy else "disease-box"
        icon       = "✅" if is_healthy else "⚠️"

        st.markdown(
            f'<div class="{box_class}">'
            f'<h3>{icon} {best_label}</h3>'
            f'<p><strong>Confidence:</strong> {confidence*100:.1f}%</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Confidence bar
        st.progress(confidence)

        # Top-5 chart
        st.markdown("#### 📊 Top-5 Predictions")
        for label, prob in top5:
            col_label, col_bar = st.columns([2, 3])
            col_label.caption(label)
            col_bar.progress(prob, text=f"{prob*100:.1f}%")

    elif not uploaded_file:
        st.markdown("### 👈 Upload an image to get started")
        st.info(
            "Tips for best results:\n"
            "- Use clear, well-lit images\n"
            "- Focus on a single leaf\n"
            "- Avoid blurry or dark photos"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Disease information section
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file and predict_btn:
    st.markdown("---")
    st.markdown("## 📚 Disease Information")

    info = disease_info.get(best_label)
    if info:
        # Severity badge color
        severity_colors = {
            "None":            ("#c8e6c9", "#1b5e20"),
            "Moderate":        ("#fff9c4", "#f57f17"),
            "Moderate to High":("#ffe0b2", "#e65100"),
            "High":            ("#ffcdd2", "#b71c1c"),
            "Very High":       ("#d50000", "#ffffff"),
        }
        bg, fg = severity_colors.get(info["severity"], ("#e0e0e0", "#000"))

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"**Severity:** "
                        f'<span class="severity-badge" style="background:{bg};color:{fg}">'
                        f'{info["severity"]}</span>', unsafe_allow_html=True)
            st.markdown(f"**Description:** {info['description']}")

            st.markdown("#### 🔍 Symptoms")
            for s in info["symptoms"]:
                st.markdown(f"- {s}")

            st.markdown("#### 🦠 Causes")
            for c in info["causes"]:
                st.markdown(f"- {c}")

        with col_b:
            st.markdown("#### 🛡️ Prevention")
            for p in info["prevention"]:
                st.markdown(f"- {p}")

            st.markdown("#### 💊 Treatment")
            for t in info["treatment"]:
                st.markdown(f"- {t}")
    else:
        st.info("No detailed information available for this class.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "Built with PyTorch · ResNet9 · Streamlit · PlantVillage Dataset"
    "</div>",
    unsafe_allow_html=True
)
