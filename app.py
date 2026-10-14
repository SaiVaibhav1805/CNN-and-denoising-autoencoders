"""Interactive Streamlit Web Application for Smart Waste Classification.

PRD Requirements:
- FR-07: Display clean, noisy, and reconstructed images side by side.
- FR-09: Image upload, predicted class, and confidence visualization.
- FR-10: Display reconstruction metrics (PSNR, MSE) and classification comparison.
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.prepare_data import add_gaussian_noise, add_salt_and_pepper_noise

# Configure Page
st.set_page_config(
    page_title="Smart Waste Classifier & Denoising AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #10b981, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .badge-cardboard { background-color: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 6px; }
    .badge-glass { background-color: #e0f2fe; color: #075985; padding: 4px 8px; border-radius: 6px; }
    .badge-metal { background-color: #f1f5f9; color: #334155; padding: 4px 8px; border-radius: 6px; }
    .badge-paper { background-color: #f3e8ff; color: #6b21a8; padding: 4px 8px; border-radius: 6px; }
    .badge-plastic { background-color: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 6px; }
    .badge-trash { background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained models into memory...")
def load_models():
    """Loads trained CNN and DAE models with caching."""
    import tensorflow as tf
    from tensorflow import keras
    
    cnn = None
    dae = None
    
    if config.CNN_MODEL_PATH.exists():
        try:
            cnn = keras.models.load_model(config.CNN_MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load CNN model: {e}")
            
    if config.DAE_MODEL_PATH.exists():
        try:
            dae = keras.models.load_model(config.DAE_MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load DAE model: {e}")
            
    return cnn, dae


def compute_metrics(clean_arr: np.ndarray, recon_arr: np.ndarray):
    """Calculates MSE and PSNR."""
    mse = float(np.mean((clean_arr - recon_arr) ** 2))
    psnr = float(20 * np.log10(1.0 / np.sqrt(mse))) if mse > 0 else 100.0
    return mse, psnr


def build_prob_chart(probs_dict: dict, title: str, bar_color: str):
    """Creates a horizontal probability bar chart using Plotly."""
    classes = list(probs_dict.keys())
    values = [probs_dict[c] * 100 for c in classes]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=classes,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, "#e2e8f0"], [1, bar_color]],
            line=dict(color="#94a3b8", width=1)
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="auto"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, weight="bold")),
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        xaxis=dict(range=[0, 100], title="Probability (%)", showgrid=True),
        yaxis=dict(autorange="reversed"),
        template="plotly_white"
    )
    return fig


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/recycle-sign.png", width=70)
    st.title("Settings & Controls")
    
    st.markdown("### 🎛️ Noise Simulation")
    noise_type = st.selectbox("Noise Type", ["Gaussian Noise", "Salt & Pepper Noise"])
    
    if noise_type == "Gaussian Noise":
        noise_level = st.slider("Gaussian Noise Std (σ)", 0.0, 0.6, 0.25, 0.05,
                                help="Simulates visual sensor noise, fog, or dust on camera lens.")
    else:
        noise_level = st.slider("Corrupted Pixel Ratio", 0.0, 0.3, 0.08, 0.02,
                                help="Simulates transmission dropouts or dead camera sensor pixels.")
                                
    st.markdown("---")
    st.markdown("### 📦 Waste Classes")
    for cls_name in config.CLASSES:
        st.markdown(f"- `<span class='badge-{cls_name}'>{cls_name.capitalize()}</span>`", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### ℹ️ Architecture Info")
    st.caption("**CNN Classifier**: 3 Convolutional Blocks + GAP + Dense(128) + Softmax(6)")
    st.caption("**DAE Autoencoder**: 3 Conv-Pool Encoder layers -> 16x16x128 Latent -> 3 Transposed Conv Decoder layers")
    st.caption("**Resolution**: 128 × 128 × 3 RGB")


# --- MAIN HEADER ---
st.markdown("<div class='main-title'>Smart Waste Classification AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Deep Learning System powered by CNN Classification & Convolutional Denoising Autoencoder (DAE)</div>", unsafe_allow_html=True)

cnn_model, dae_model = load_models()

# Warning banner if models not trained yet
if cnn_model is None or dae_model is None:
    st.warning("⚠️ **Trained models not detected in `models/` directory.**")
    st.info("""
    To train the models on your machine:
    1. Prepare dataset: `python -m src.prepare_data`
    2. Train CNN: `python -m src.train_cnn --epochs 15`
    3. Train Autoencoder: `python -m src.train_denoising_ae --epochs 20`
    4. Run benchmark evaluation: `python -m src.evaluate`
    """)

# Tabs for Navigation
tab_demo, tab_evaluation, tab_about = st.tabs(["🖼️ Interactive Classifier & DAE", "📊 Benchmark Evaluation (FR-08)", "📖 PRD & Architecture Guide"])

with tab_demo:
    col_input, col_meta = st.columns([2, 1])
    
    with col_input:
        input_source = st.radio("Choose Input Source:", ["Upload My Own Image", "Pick a Dataset Sample"], horizontal=True)
        
        selected_image = None
        sample_label = "Unknown"
        
        if input_source == "Upload My Own Image":
            uploaded_file = st.file_uploader("Upload waste image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
        else:
            # Look for samples in data/raw or sample starter data
            sample_images = []
            valid_exts = {".jpg", ".png", ".jpeg"}
            for cls_name in config.CLASSES:
                cls_dir = config.RAW_DATA_DIR / cls_name
                if cls_dir.exists():
                    for f in cls_dir.iterdir():
                        if f.suffix.lower() in valid_exts:
                            sample_images.append((f, cls_name))
                            if len(sample_images) >= 30:
                                break
                                
            if sample_images:
                sample_names = [f"[{cls.upper()}] {p.name}" for p, cls in sample_images]
                idx_choice = st.selectbox("Select Sample Image:", range(len(sample_names)), format_func=lambda i: sample_names[i])
                chosen_path, sample_label = sample_images[idx_choice]
                selected_image = Image.open(chosen_path).convert("RGB")
            else:
                st.info("No raw images found. Upload an image above or run `python -m src.dataset_downloader --mock` to populate sample images.")
                
    with col_meta:
        st.markdown("#### Input Information")
        if selected_image is not None:
            st.write(f"- **Dimensions**: {selected_image.size[0]} × {selected_image.size[1]} px")
            st.write(f"- **Format**: {selected_image.format or 'RGB'}")
            st.write(f"- **True Category**: `{sample_label}`")
            st.write(f"- **Active Noise Level**: `{noise_level}`")
        else:
            st.info("Please select or upload an image to begin classification.")
            
    if selected_image is not None:
        st.markdown("---")
        # Preprocess input image to 128x128
        clean_pil = selected_image.resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH), Image.Resampling.BILINEAR)
        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
        
        # Inject artificial noise
        if noise_type == "Gaussian Noise":
            noisy_arr = add_gaussian_noise(np.expand_dims(clean_arr, 0), noise_factor=noise_level)[0]
        else:
            noisy_arr = add_salt_and_pepper_noise(np.expand_dims(clean_arr, 0), amount=noise_level)[0]
            
        # Reconstruct with DAE
        if dae_model is not None:
            recon_arr = dae_model.predict(np.expand_dims(noisy_arr, 0), verbose=0)[0]
            recon_arr = np.clip(recon_arr, 0.0, 1.0)
        else:
            # Fallback mock reconstruction if model weights not yet generated
            recon_arr = np.clip(clean_arr * 0.9 + 0.05, 0.0, 1.0)
            
        # Classify with CNN
        if cnn_model is not None:
            probs_clean = cnn_model.predict(np.expand_dims(clean_arr, 0), verbose=0)[0]
            probs_noisy = cnn_model.predict(np.expand_dims(noisy_arr, 0), verbose=0)[0]
            probs_recon = cnn_model.predict(np.expand_dims(recon_arr, 0), verbose=0)[0]
        else:
            # Fallback uniform/demo probabilities
            probs_clean = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES
            probs_noisy = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES
            probs_recon = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES
            
        pred_clean = config.IDX_TO_CLASS[int(np.argmax(probs_clean))]
        pred_noisy = config.IDX_TO_CLASS[int(np.argmax(probs_noisy))]
        pred_recon = config.IDX_TO_CLASS[int(np.argmax(probs_recon))]
        
        conf_clean = float(np.max(probs_clean)) * 100
        conf_noisy = float(np.max(probs_noisy)) * 100
        conf_recon = float(np.max(probs_recon)) * 100
        
        # Quality Metrics
        mse_noisy, psnr_noisy = compute_metrics(clean_arr, noisy_arr)
        mse_recon, psnr_recon = compute_metrics(clean_arr, recon_arr)
        
        # FR-07: Display clean, noisy, and reconstructed images side by side
        st.markdown("### 🔍 Side-by-Side Visual Inspection & Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1. Clean Original")
            st.image(clean_arr, caption="Original Input (Uncorrupted)", use_container_width=True)
            st.markdown(f"**Predicted:** <span class='badge-{pred_clean}'>{pred_clean.upper()}</span> ({conf_clean:.1f}%)", unsafe_allow_html=True)
            dict_clean = {config.IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs_clean)}
            st.plotly_chart(build_prob_chart(dict_clean, "Clean Class Probabilities", "#2563eb"), use_container_width=True)
            
        with col2:
            st.markdown(f"#### 2. Corrupted Noisy")
            st.image(noisy_arr, caption=f"Noisy Input ({noise_type})", use_container_width=True)
            st.markdown(f"**Predicted:** <span class='badge-{pred_noisy}'>{pred_noisy.upper()}</span> ({conf_noisy:.1f}%)", unsafe_allow_html=True)
            dict_noisy = {config.IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs_noisy)}
            st.plotly_chart(build_prob_chart(dict_noisy, "Noisy Class Probabilities", "#ef4444"), use_container_width=True)
            
        with col3:
            st.markdown("#### 3. DAE Reconstructed")
            st.image(recon_arr, caption="Denoised through Autoencoder", use_container_width=True)
            st.markdown(f"**Predicted:** <span class='badge-{pred_recon}'>{pred_recon.upper()}</span> ({conf_recon:.1f}%)", unsafe_allow_html=True)
            dict_recon = {config.IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs_recon)}
            st.plotly_chart(build_prob_chart(dict_recon, "Denoised Class Probabilities", "#10b981"), use_container_width=True)
            
        # FR-10: Show reconstruction and classification metrics in interface
        st.markdown("---")
        st.markdown("### 📈 Quantitative Image Quality Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Corrupted PSNR", f"{psnr_noisy:.2f} dB")
        m_col2.metric("Denoised PSNR", f"{psnr_recon:.2f} dB", delta=f"{psnr_recon - psnr_noisy:+.2f} dB")
        m_col3.metric("Reconstruction MSE", f"{mse_recon:.5f}")
        m_col4.metric("Denoising Status", "Restored" if psnr_recon > psnr_noisy else "Unchanged")

with tab_evaluation:
    st.markdown("### 📊 Held-Out Test Evaluation Experiment (FR-08, Section 8.3)")
    st.write("Controlled evaluation isolating whether denoising improves classification performance across clean, noisy, and denoised test images.")
    
    eval_file = config.EVALUATION_REPORT_FILE
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        c_m = report_data["clean"]
        n_m = report_data["noisy"]
        d_m = report_data["denoised"]
        dae_m = report_data["dae_reconstruction"]
        
        # Metric Table
        st.markdown("#### Test Set Performance Summary")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.info(f"**1. Clean Images**\n- Accuracy: **{c_m['accuracy']*100:.2f}%**\n- Macro F1: **{c_m['macro_f1']:.3f}**\n- Precision: **{c_m['macro_precision']:.3f}**")
        with col_t2:
            st.error(f"**2. Noisy Images**\n- Accuracy: **{n_m['accuracy']*100:.2f}%**\n- Macro F1: **{n_m['macro_f1']:.3f}**\n- Precision: **{n_m['macro_precision']:.3f}**")
        with col_t3:
            st.success(f"**3. DAE Denoised**\n- Accuracy: **{d_m['accuracy']*100:.2f}%**\n- Macro F1: **{d_m['macro_f1']:.3f}**\n- Precision: **{d_m['macro_precision']:.3f}**")
            
        st.markdown(f"**Autoencoder Reconstruction**: MSE: `{dae_m['mse']:.5f}` | PSNR: `{dae_m['reconstructed_psnr']:.2f} dB` (PSNR Gain: `+{dae_m['psnr_gain']:.2f} dB`)")
        
        plot_path = config.REPORTS_DIR / "evaluation_comparison.png"
        if plot_path.exists():
            st.image(str(plot_path), caption="Detailed Evaluation Plots: Confusion Matrices & Metrics Comparison", use_container_width=True)
    else:
        st.info("Evaluation report not yet generated. Run `python -m src.evaluate` to execute the 3-condition test experiment.")

with tab_about:
    st.markdown("### 📘 Product Requirements Document Overview")
    st.markdown("""
    **Project Title:** Smart Waste Classification using CNN and Denoising Autoencoder  
    **Version:** 1.0 | Academic Deep Learning Project  
    
    #### Core Objectives:
    1. **6-Class Classification**: Cardboard, Glass, Metal, Paper, Plastic, Trash.
    2. **Denoising Autoencoder**: Encoder-decoder CNN to filter out camera noise, dust, and artifacts.
    3. **Controlled 3-Condition Evaluation**: Quantify accuracy restoration across clean, noisy, and denoised test inputs.
    
    #### Recommended Datasets:
    - **TrashNet (Primary Recommendation)**: 2,527 images in 6 balanced/imbalanced classes.
    - **TrashBusters Combined (Hugging Face)**: 29,849 images across 6 harmonized classes.
    """)
