import os

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from analysis import build_analysis

st.set_page_config(page_title="AI-Based Multi-Task Analysis of Bone Scan", layout="wide")

with open("styles.css", "r", encoding="utf-8") as handle:
    styles = handle.read()
st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="pill">AI Medical Imaging</div>
        <h1>AI-Based Multi-Task Analysis of Bone Scan</h1>
        <p>Upload a bone scan image or provide a local path to generate multi-step preprocessing, feature extraction, and visual analytics in one view.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

is_cloud = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true" or os.environ.get("STREAMLIT_CLOUD") == "true"
input_options = ["Upload image"] if is_cloud else ["Upload image", "Local path"]
input_mode = st.radio("Input source", input_options, horizontal=True)

uploaded_file = None
image_path = ""
fname = ""
img_bgr = None

if input_mode == "Upload image":
    uploaded_file = st.file_uploader("Upload Bone Scan Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        fname = uploaded_file.name
elif input_mode == "Local path":
    image_path = st.text_input("Image path (local)", placeholder=r"C:\path\to\image.jpg")
    if image_path:
        fname = os.path.basename(image_path)
        if os.path.exists(image_path):
            img_bgr = cv2.imread(image_path)
        else:
            st.error(f"Image not found: {image_path}")
if img_bgr is not None:
    fig, metrics = build_analysis(img_bgr, fname)

    left, right = st.columns([2.1, 1])
    with left:
        st.pyplot(fig, width="stretch")
    with right:
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.subheader("Key Metrics")
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric"><div class="label">Contrast</div><div class="value">{metrics['contrast']:.4f}</div></div>
                <div class="metric"><div class="label">Energy</div><div class="value">{metrics['energy']:.4f}</div></div>
                <div class="metric"><div class="label">Homogeneity</div><div class="value">{metrics['homogeneity']:.4f}</div></div>
                <div class="metric"><div class="label">Mean</div><div class="value">{metrics['mean']:.2f}</div></div>
                <div class="metric"><div class="label">Std</div><div class="value">{metrics['std']:.2f}</div></div>
                <div class="metric"><div class="label">Median</div><div class="value">{metrics['median']:.2f}</div></div>
                <div class="metric"><div class="label">Otsu Thr</div><div class="value">{metrics['otsu']:.0f}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.checkbox("Save analysis image"):
        if image_path:
            base_dir = os.path.dirname(image_path)
        else:
            base_dir = os.getcwd()
        stem, ext = os.path.splitext(fname)
        save_path = st.text_input("Save path", value=os.path.join(base_dir, f"analysis_{stem}.png"))
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            st.success(f"Saved: {save_path}")

    plt.close(fig)
