#!/usr/bin/env python3
"""
CipherFrame — Image Encryption Studio (Streamlit Version)
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import time
from io import BytesIO
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Core imports ──────────────────────────
from core.encryption import (
    encrypt_image,
    decrypt_image,
    encrypt_matrix,
    chaotic_diffusion,
    _pad_to_block,
)
from core.matrix import (
    generate_keys_from_password,
    generate_permutation_matrix,
    shuffle_image,
)
from core.metrics import mae, npcr, uaci, adjacent_pixel_correlation, entropy
from core.histogram_report import plot_rgb_line_histogram_grid
from core.security_report import (
    table_pixel_average_distance,
    tables_correlation_all_stages,
    table_correlation_comparison,
    table_statistical_parameters,
    analyze_key_space,
    table_correlation_per_stage,
    table_entropy,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_image(uploaded_file):
    """Load image from Streamlit uploaded file."""
    img = Image.open(uploaded_file).convert('RGB')
    return np.array(img)

def numpy_to_pil(arr):
    """Convert numpy array to PIL Image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def get_image_download_link(arr, filename, format="PNG"):
    """Generate download link for image."""
    img = numpy_to_pil(arr)
    buf = BytesIO()
    if format.upper() == "PNG":
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf

def plot_histograms_side_by_side(img1, img2, title1="Original", title2="Encrypted"):
    """Plot two histograms side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")
    
    for ax, img, title in zip(axes, [img1, img2], [title1, title2]):
        for idx, (ch, color) in enumerate(zip(["R", "G", "B"], ["#dc2626", "#16a34a", "#2563eb"])):
            hist, _ = np.histogram(img[:, :, idx], bins=256, range=(0, 256))
            ax.plot(hist, color=color, linewidth=1, label=ch, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.2)
    
    fig.tight_layout()
    return fig

def plot_all_stages_histogram(original, after_matrix, after_shuffle, after_chaos):
    """Plot histograms for all encryption stages."""
    h, w, _ = original.shape
    stages = {
        "Original": original,
        "GTSnm": after_matrix[:h, :w],
        "Shuffle": after_shuffle[:h, :w],
        "Encrypted": after_chaos[:h, :w]
    }
    return plot_rgb_line_histogram_grid(stages, figsize=(14, 4))

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize session state variables."""
    if "encrypted_data" not in st.session_state:
        st.session_state.encrypted_data = None
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    if "keys" not in st.session_state:
        st.session_state.keys = None

init_session_state()

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="CipherFrame — Image Encryption Studio",
        page_icon="⛓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #0B0F19;
            color: #E8EAF0;
        }
        .stSidebar {
            background-color: #121826;
        }
        h1, h2, h3 {
            color: #00D9A3 !important;
        }
        .stButton>button {
            background-color: #00D9A3;
            color: #06140F;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #00A87E;
        }
        div[data-testid="stSidebarNav"] {
            background-color: #121826;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### ⛓ CipherFrame")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Beranda",
             "🔐 Enkripsi",
             "🗝 Dekripsi",
             "📊 Histogram",
             "🛡 Analisis Keamanan",
             "📐 Eval GTSnm",
             "〰 Eval Chaotic",
             "ℹ Tentang"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption("GTSnm · Shuffle · Henon Map")
    
    # Page routing
    if "Beranda" in page:
        show_home()
    elif "Enkripsi" in page:
        show_encrypt_page()
    elif "Dekripsi" in page:
        show_decrypt_page()
    elif "Histogram" in page:
        show_histogram_page()
    elif "Keamanan" in page:
        show_security_page()
    elif "GTSnm" in page:
        show_eval_gtsnm_page()
    elif "Chaotic" in page:
        show_eval_chaos_page()
    elif "Tentang" in page:
        show_about_page()


# ============================================================
# HOME PAGE
# ============================================================

def show_home():
    st.title("⛓ CipherFrame — Image Encryption Studio")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Tentang Aplikasi
        
        CipherFrame adalah aplikasi untuk **enkripsi citra digital** menggunakan 
        kombinasi tiga teknik kriptografi:
        
        1. **GTSnm Matrix** — Matriks Pascal yang dimodifikasi untuk transformasi blok
        2. **Shuffle** — Permutasi piksel untuk konfusi posisi
        3. **Henon Map** — Difusi chaos untuk sensitivitas tinggi
        
        ---
        
        ### Fitur Utama
        
        - 🔐 Enkripsi citra dengan algoritma GTSnm + Shuffle + Henon Map
        - 🗝 Dekripsi citra yang tersimpan (dengan metadata)
        - 📊 Visualisasi histogram RGB per tahap
        - 🛡 Analisis keamanan (MAE, NPCR, UACI, Entropy, Korelasi)
        - ⚖ Perbandingan metrik dengan/tanpa komponen tertentu
        """)
    
    with col2:
        st.info("""
        **Cara Penggunaan:**
        
        1. Pilih halaman **Enkripsi**
        2. Upload gambar
        3. Masukkan password (min 8 karakter)
        4. Klik **Enkripsi**
        5. Download hasil beserta file metadata
        """)
        
        st.success("""
        **Untuk Dekripsi:**
        
        1. Pilih halaman **Dekripsi**
        2. Upload citra terenkripsi
        3. Upload file metadata (.json)
        4. Masukkan password yang sama
        5. Klik **Dekripsi**
        """)


# ============================================================
# ENCRYPT PAGE
# ============================================================

def show_encrypt_page():
    st.title("🔐 Enkripsi Citra")
    st.markdown("---")
    
    # Upload image
    uploaded_file = st.file_uploader(
        "Pilih gambar untuk dienkripsi",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        key="encrypt_upload"
    )
    
    if uploaded_file is None:
        st.info("Silakan upload gambar terlebih dahulu.")
        return
    
    # Load and display image
    img = load_image(uploaded_file)
    h, w, c = img.shape
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Citra Asli")
        st.image(img, caption=f"Ukuran: {w}×{h}×{c}")
    
    with col2:
        st.subheader("Parameter Kunci")
        st.markdown("*Akan ditampilkan setelah enkripsi*")
    
    # Password input
    st.markdown("---")
    password = st.text_input(
        "Masukkan Password (minimal 8 karakter)",
        type="password",
        key="encrypt_password"
    )
    
    if len(password) < 8 and password != "":
        st.warning("Password minimal 8 karakter!")
    
    # Encrypt button
    if st.button("🔐 Enkripsi", type="primary", disabled=(len(password) < 8)):
        with st.spinner("Memproses enkripsi..."):
            try:
                # Generate keys
                seed, d, l = generate_keys_from_password(password, img.shape)
                
                # Encrypt
                start = time.perf_counter()
                after_matrix, after_shuffle, after_chaos, enc_full, profiling = encrypt_image(img, d, l, seed)
                encrypt_time = time.perf_counter() - start
                
                # Store in session state
                st.session_state.original_image = img
                st.session_state.encrypted_data = {
                    "after_matrix": after_matrix,
                    "after_shuffle": after_shuffle,
                    "after_chaos": after_chaos,
                    "d": d,
                    "l": l,
                    "seed": seed,
                    "original_shape": img.shape,
                    "encrypt_time": encrypt_time,
                    "profiling": profiling
                }
                
                st.success(f"✅ Enkripsi selesai dalam {encrypt_time:.4f} detik!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
    
    # Display results
    if st.session_state.encrypted_data is not None:
        data = st.session_state.encrypted_data
        d, l, seed = data["d"], data["l"], data["seed"]
        
        # Show key parameters
        with col2:
            st.subheader("Parameter Kunci")
            st.code(f"""
Block Size (d) : {d}
Power (l)      : {l}
Seed           : {seed}
            """)
        
        # Show encryption stages
        st.markdown("---")
        st.subheader("Tahap Enkripsi")
        
        col1, col2, col3, col4 = st.columns(4
