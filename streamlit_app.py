#!/usr/bin/env python3
"""
CipherFrame — Image Encryption Studio (Streamlit Version)
Aplikasi web untuk enkripsi citra menggunakan:
  GTSnm Matrix  ·  Shuffle  ·  Henon Map (Chaotic Diffusion)
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time
import json
import io
import os
import sys

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
from core.histogram_report import plot_rgb_line_histogram_grid, save_figure_high_res
from core.security_report import (
    table_pixel_average_distance,
    tables_correlation_all_stages,
    table_correlation_comparison,
    table_statistical_parameters,
    analyze_key_space,
    export_all_to_excel,
    table_correlation_per_stage,
    table_entropy,
)
from utils.image_utils import load_image

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CipherFrame — Image Encryption Studio",
    page_icon="⛓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B0F19 0%, #121826 100%);
        }
        [data-testid="stSidebar"] * {
            color: #E8EAF0 !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #00D9A3 !important;
        }
        
        /* Cards */
        .card {
            background-color: #121826;
            border: 1px solid #232B3D;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        }
        
        .card-header {
            background-color: #171F30;
            border-radius: 8px;
            padding: 12px 16px;
            margin: -20px -20px 15px -20px;
            border-bottom: 1px solid #232B3D;
        }
        
        .card-title {
            font-size: 14px;
            font-weight: bold;
            color: #00D9A3;
            margin: 0;
        }
        
        /* Key chips */
        .key-chip {
            display: inline-block;
            background-color: #171F30;
            border: 1px solid #232B3D;
            border-radius: 8px;
            padding: 8px 16px;
            margin: 4px;
        }
        
        .key-label {
            font-size: 10px;
            color: #7A8194;
            margin-bottom: 4px;
        }
        
        .key-value {
            font-size: 16px;
            font-weight: bold;
            color: #00D9A3;
            font-family: monospace;
        }
        
        /* Image container */
        .img-box {
            background-color: #1A1A2E;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
        
        .img-label {
            font-size: 11px;
            font-weight: bold;
            color: #E8EAF0;
            margin-bottom: 8px;
        }
        
        /* Profiling box */
        .profiling-box {
            background-color: #0F1422;
            border: 1px solid #232B3D;
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 12px;
            color: #00D9A3;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        
        /* Status */
        .status-success {
            color: #00D9A3;
        }
        
        .status-warning {
            color: #FFD700;
        }
        
        .status-error {
            color: #FF5470;
        }
        
        /* Metric card */
        .metric-card {
            background-color: #121826;
            border: 1px solid #232B3D;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #00D9A3;
        }
        
        .metric-label {
            font-size: 12px;
            color: #7A8194;
            margin-top: 5px;
        }
        
        /* Sidebar nav */
        .nav-item {
            padding: 10px 15px;
            border-radius: 8px;
            margin: 2px 0;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .nav-item:hover {
            background-color: #171F30;
        }
        
        .nav-item.active {
            background-color: #171F30;
            border-left: 3px solid #00D9A3;
        }
        
        /* Table styling */
        .dataframe {
            font-size: 12px;
        }
        
        .dataframe th {
            background-color: #171F30 !important;
            color: #00D9A3 !important;
        }
        
        /* Hide default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: bold;
        }
        
        /* File uploader */
        .stFileUploader > div {
            background-color: #121826;
            border: 2px dashed #232B3D;
            border-radius: 10px;
        }
        
        .stFileUploader > div:hover {
            border-color: #00D9A3;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'image': None,
        'after_matrix': None,
        'after_shuffle': None,
        'after_chaos': None,
        'enc_full': None,
        'decrypted': None,
        'd': None,
        'l': None,
        'seed': None,
        'encrypt_time': 0,
        'decrypt_time': 0,
        'page': 'home',
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def numpy_to_pil(arr):
    """Convert numpy array to PIL Image."""
    if arr is None:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def get_image_download_link(arr, filename="image.png"):
    """Generate download link for image."""
    if arr is None:
        return None
    img = numpy_to_pil(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def display_key_chips(d, l, seed):
    """Display key parameter chips."""
    if d is None:
        st.info("🔑 Parameter kunci belum tersedia")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="key-chip">
            <div class="key-label">BLOK (d)</div>
            <div class="key-value">{d}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="key-chip">
            <div class="key-label">PANGKAT (l)</div>
            <div class="key-value">{l}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="key-chip">
            <div class="key-label">SEED</div>
            <div class="key-value">{seed}</div>
        </div>
        """, unsafe_allow_html=True)

def display_image_box(arr, title, key=None):
    """Display image in a styled box with download button."""
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <div class="card-title">{title}</div>
        </div>
    """, unsafe_allow_html=True)
    
    if arr is not None:
        img = numpy_to_pil(arr)
        st.image(img, use_container_width=True)
        
        if key:
            buf = get_image_download_link(arr)
            if buf:
                st.download_button(
                    label="↓ Download",
                    data=buf,
                    file_name=f"{key}.png",
                    mime="image/png",
                    use_container_width=True
                )
    else:
        st.markdown("""
        <div class="img-box">
            <p style="color: #4B5468; padding: 40px 0;">Belum ada gambar</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_dataframe(title, df, note=None):
    """Render a DataFrame in a styled card."""
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <div class="card-title">{title}</div>
        </div>
    """, unsafe_allow_html=True)
    
    if note:
        st.caption(note)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" / ".join(map(str, c)).strip(" /") for c in df.columns]
    
    df_display = df.reset_index() if df.index.name else df
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_profiling(text):
    """Render profiling text in a styled box."""
    st.markdown(f"""
    <div class="profiling-box">{text}</div>
    """, unsafe_allow_html=True)

def save_metadata_file(original_shape, d, l, seed):
    """Create metadata JSON content."""
    metadata = {
        "original_shape": list(original_shape),
        "block_size": int(d),
        "power": int(l),
        "seed": int(seed)
    }
    return json.dumps(metadata, indent=2)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
def render_sidebar():
    """Render sidebar with navigation."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <span style="font-size: 32px;">⛓</span>
            <h2 style="margin: 10px 0 5px 0; color: #00D9A3;">CipherFrame</h2>
            <p style="font-size: 11px; color: #7A8194; margin: 0;">GTSnm · Shuffle · Henon Map</p>
        </div>
        <hr style="border-color: #232B3D; margin: 10px 0 20px 0;">
        """, unsafe_allow_html=True)
        
        pages = [
            ("home", "🏠", "Beranda"),
            ("encrypt", "🔐", "Enkripsi Citra"),
            ("decrypt", "🗝", "Dekripsi Citra"),
            ("eval_gtsmn", "📐", "Eval GTSnm"),
            ("eval_chaos", "〰", "Eval Chaotic"),
            ("histogram", "📊", "Histogram RGB"),
            ("security", "🛡", "Analisis Keamanan"),
            ("about", "ℹ", "Tentang"),
        ]
        
        for key, icon, label in pages:
            active = st.session_state.page == key
            btn_type = "primary" if active else "secondary"
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True, 
                        type=btn_type if active else "secondary"):
                st.session_state.page = key
                st.rerun()
        
        st.markdown("<br><hr style='border-color: #232B3D;'>", unsafe_allow_html=True)
        
        if st.button("🚪 Keluar", use_container_width=True, type="secondary"):
            st.stop()

# ============================================================
# PAGE: HOME
# ============================================================
def page_home():
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <span style="font-size: 64px;">⛓</span>
        <h1 style="margin: 20px 0 10px 0;">Image Encryption Studio</h1>
        <p style="font-size: 16px; color: #7A8194;">GTSnm Algorithm  +  Shuffle  +  Henon Map (Chaotic)</p>
    </div>
    """, unsafe_allow_html=True)
    
    cards = [
        ("encrypt", "🔐", "Enkripsi Citra", "Proses citra lewat GTSnm Matrix, Shuffle, dan Henon Map."),
        ("decrypt", "🗝", "Dekripsi File", "Buka citra terenkripsi tersimpan dan pulihkan dengan password."),
        ("eval_gtsmn", "📐", "Evaluasi GTSnm", "Analisis statistik GTSnm Matrix (tanpa Shuffle & Chaotic)."),
        ("eval_chaos", "〰", "Evaluasi Chaotic", "Analisis statistik Difusi Henon Map (tanpa GTSnm & Shuffle)."),
        ("histogram", "📊", "Histogram RGB", "Bandingkan distribusi warna di tiap tahap enkripsi."),
        ("security", "🛡", "Analisis Keamanan", "Hitung MAE, NPCR, UACI, korelasi piksel, dan key space."),
        ("about", "ℹ", "Tentang Aplikasi", "Penjelasan algoritma dan referensi yang digunakan."),
    ]
    
    cols = st.columns(2)
    for i, (key, icon, title, desc) in enumerate(cards):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="card" style="cursor: pointer;" onclick="document.querySelector('[data-testid=\"stButton\"]')">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 24px; margin-right: 12px;">{icon}</span>
                    <span style="font-size: 16px; font-weight: bold; color: #E8EAF0;">{title}</span>
                </div>
                <p style="font-size: 13px; color: #7A8194; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"→ Buka {title}", key=f"home_{key}", use_container_width=True, type="secondary"):
                st.session_state.page = key
                st.rerun()

# ============================================================
# PAGE: ENCRYPT
# ============================================================
def page_encrypt():
    st.markdown("### 🔐 Enkripsi Citra")
    st.caption("GTSnm Algorithm  ·  Shuffle  ·  Henon Map (Chaotic)")
    
    # Upload section
    with st.expander("📂 Pilih Gambar", expanded=st.session_state.image is None):
        uploaded_file = st.file_uploader(
            "Upload gambar untuk dienkripsi",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="encrypt_upload"
        )
        
        if uploaded_file:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = load_image(uploaded_file.name)
                st.session_state.image = img
                
                h, w = img.shape[:2]
                st.success(f"✅ Gambar dimuat: {w}×{h}")
                st.image(numpy_to_pil(img), width=300)
            except Exception as e:
                st.error(f"❌ Gagal memuat gambar: {e}")
    
    if st.session_state.image is None:
        st.info("👆 Upload gambar terlebih dahulu")
        return
    
    # Password input
    st.markdown("---")
    password = st.text_input("🔐 Password Enkripsi", type="password", placeholder="Minimal 8 karakter")
    
    if password and len(password) < 8:
        st.warning("⚠️ Password minimal 8 karakter")
    
    # Encrypt button
    if st.button("🔐 Enkripsi", type="primary", use_container_width=True, disabled=not password or len(password) < 8):
        try:
            img = st.session_state.image
            seed, d, l = generate_keys_from_password(password, img.shape)
            
            st.session_state.d = d
            st.session_state.l = l
            st.session_state.seed = seed
            
            with st.spinner("Mengenkripsi..."):
                start = time.perf_counter()
                (after_matrix, after_shuffle, after_chaos,
                 enc_full, profiling) = encrypt_image(img, d, l, seed)
                st.session_state.encrypt_time = time.perf_counter() - start
            
            st.session_state.after_matrix = after_matrix
            st.session_state.after_shuffle = after_shuffle
            st.session_state.after_chaos = after_chaos
            st.session_state.enc_full = enc_full
            
            st.success(f"✅ Enkripsi selesai dalam {st.session_state.encrypt_time:.4f} detik")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal mengenkripsi: {e}")
    
    # Display results
    if st.session_state.after_chaos is not None:
        st.markdown("---")
        st.markdown("### 🔑 Parameter Kunci")
        display_key_chips(st.session_state.d, st.session_state.l, st.session_state.seed)
        
        st.markdown("---")
        st.markdown("### 📊 Alur Proses Enkripsi")
        
        h, w, _ = st.session_state.image.shape
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_image_box(st.session_state.image, "1 · Citra Asli", "citra_asli")
        with col2:
            display_image_box(st.session_state.after_matrix[:h, :w], "2 · GTSnm Matrix", "gtsnm_matrix")
        with col3:
            display_image_box(st.session_state.after_shuffle[:h, :w], "3 · Shuffle", "shuffle")
        with col4:
            display_image_box(st.session_state.after_chaos[:h, :w], "4 · Henon Map (Final)", "terenkripsi")
        
        # Download with metadata
        st.markdown("---")
        st.markdown("### 💾 Simpan Hasil Enkripsi")
        
        col1, col2 = st.columns(2)
        with col1:
            buf = get_image_download_link(st.session_state.after_chaos)
            if buf:
                st.download_button(
                    label="↓ Download Citra Terenkripsi (.png)",
                    data=buf,
                    file_name="terenkripsi.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with col2:
            meta_content = save_metadata_file(
                st.session_state.image.shape,
                st.session_state.d,
                st.session_state.l,
                st.session_state.seed
            )
            st.download_button(
                label="↓ Download Metadata (.json)",
                data=meta_content,
                file_name="terenkripsi.meta.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.warning("⚠️ Pastikan kedua file (.png dan .meta.json) disimpan di folder yang sama untuk dekripsi!")

# ============================================================
# PAGE: DECRYPT
# ============================================================
def page_decrypt():
    st.markdown("### 🗝 Dekripsi Citra Tersimpan")
    st.caption("Muat file citra terenkripsi (.png) dan metadata (.meta.json) lalu pulihkan dengan password")
    
    # Upload section
    with st.expander("📂 Pilih File Terenkripsi", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            enc_file = st.file_uploader(
                "Citra Terenkripsi (.png)",
                type=["png", "bmp", "tiff"],
                key="decrypt_enc"
            )
        
        with col2:
            meta_file = st.file_uploader(
                "Metadata (.json) - Opsional",
                type=["json"],
                key="decrypt_meta"
            )
    
    metadata = None
    enc_img = None
    
    if enc_file:
        try:
            # Load encrypted image
            temp_path = f"temp_{enc_file.name}"
            with open(temp_path, "wb") as f:
                f.write(enc_file.read())
            enc_img = load_image(temp_path)
            os.remove(temp_path)
            
            st.success(f"✅ Citra terenkripsi dimuat: {enc_img.shape[1]}×{enc_img.shape[0]}")
            st.image(numpy_to_pil(enc_img), width=300)
        except Exception as e:
            st.error(f"❌ Gagal memuat citra: {e}")
    
    if meta_file:
        try:
            metadata = json.load(meta_file)
            if 'original_shape' in metadata:
                metadata['original_shape'] = tuple(metadata['original_shape'])
            st.success(f"✅ Metadata ditemukan: {metadata['original_shape']}")
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat metadata: {e}")
    
    if enc_img is None:
        st.info("👆 Upload citra terenkripsi terlebih dahulu")
        return
    
    # Password input
    st.markdown("---")
    password = st.text_input("🔑 Password Dekripsi", type="password", placeholder="Password yang sama saat enkripsi")
    
    # Decrypt button
    if st.button("🔓 Dekripsi", type="primary", use_container_width=True, disabled=not password or len(password) < 8):
        try:
            enc_h, enc_w, enc_c = enc_img.shape
            
            if metadata is not None:
                original_shape = metadata['original_shape']
                d = metadata['block_size']
                l = metadata['power']
                seed = metadata['seed']
                use_metadata = True
            else:
                seed, d, l = generate_keys_from_password(password, (enc_h, enc_w, enc_c))
                original_shape = (enc_h, enc_w)
                use_metadata = False
                
                if enc_h % d != 0 or enc_w % d != 0:
                    st.error(f"❌ Ukuran citra ({enc_w}×{enc_h}) bukan kelipatan blok (d={d}). File metadata diperlukan!")
                    return
            
            with st.spinner("Mendekripsi..."):
                start = time.perf_counter()
                dec, profiling = decrypt_image(enc_img, d, l, seed=seed, original_shape=original_shape)
                st.session_state.decrypt_time = time.perf_counter() - start
            
            st.session_state.decrypted = dec
            
            # Display results
            st.success(f"✅ Dekripsi selesai dalam {st.session_state.decrypt_time:.4f} detik")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Citra Terenkripsi")
                st.image(numpy_to_pil(enc_img))
            
            with col2:
                st.markdown("#### Hasil Dekripsi")
                st.image(numpy_to_pil(dec))
                
                buf = get_image_download_link(dec)
                if buf:
                    st.download_button(
                        label="↓ Download Hasil Dekripsi",
                        data=buf,
                        file_name="hasil_dekripsi.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            # Profiling
            st.markdown("---")
            st.markdown("### ⏱ Profiling")
            
            ict = profiling["inverse_chaos_time"]
            ust = profiling["unshuffle_time"]
            mdt = profiling["matrix_decrypt_time"]
            total = ict + ust + mdt
            
            prof_text = f"""========== DECRYPTION PROFILING =========

Encrypted Image Size : {enc_w} x {enc_h} x {enc_c}
Decrypted Image Size : {dec.shape[1]} x {dec.shape[0]}
Block Size (d)      : {d}
Power (l)           : {l}
Seed                : {seed}

-----------------------------------------
Inverse Chaos       : {ict:.4f} s
Unshuffle           : {ust:.4f} s
Matrix Decryption   : {mdt:.4f} s

-----------------------------------------
Total Decryption    : {total:.4f} s

Metadata Source     : {"File .meta.json ✅" if use_metadata else "Estimasi ⚠️"}"""
            
            render_profiling(prof_text)
            
        except Exception as e:
            st.error(f"❌ Gagal mendekripsi: {e}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================
# PAGE: EVAL GTSNM
# ============================================================
def page_eval_gtsmn():
    st.markdown("### 📐 Evaluasi GTSnm Matrix")
    st.caption("Analisis statistik enkripsi menggunakan GTSnm Matrix (tanpa Shuffle & Henon Map)")
    
    uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg", "bmp"])
    password = st.text_input("Password", type="password")
    
    if uploaded_file and password:
        img = load_image(uploaded_file.name if hasattr(uploaded_file, 'name') else "temp.png")
        seed, d, l = generate_keys_from_password(password, img.shape)
        padded = _pad_to_block(img, d)
        
        with st.spinner("Memproses..."):
            res = encrypt_matrix(padded, d, l, seed)
            h, w, _ = img.shape
            res_cropped = res[:h, :w]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Citra Asli")
            st.image(numpy_to_pil(img))
        with col2:
            st.markdown("#### Hasil GTSnm")
            st.image(numpy_to_pil(res_cropped))
        
        st.markdown("---")
        st.markdown("### 📊 Histogram RGB")
        stage_images = {"Original": img, "GTSnm": res_cropped}
        fig = plot_rgb_line_histogram_grid(stage_images, figsize=(12, 4))
        st.pyplot(fig)
        plt.close()
        
        st.markdown("### 📈 Korelasi Piksel")
        df_orig = table_correlation_per_stage(img).T
        render_dataframe("Korelasi - Original", df_orig)
        
        df_proc = table_correlation_per_stage(res_cropped).T
        render_dataframe("Korelasi - GTSnm", df_proc)

# ============================================================
# PAGE: EVAL CHAOS
# ============================================================
def page_eval_chaos():
    st.markdown("### 〰 Evaluasi Chaotic Diffusion")
    st.caption("Analisis statistik enkripsi menggunakan Henon Map (tanpa GTSnm & Shuffle)")
    
    uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg", "bmp"])
    password = st.text_input("Password", type="password")
    
    if uploaded_file and password:
        img = load_image(uploaded_file.name if hasattr(uploaded_file, 'name') else "temp.png")
        seed, d, l = generate_keys_from_password(password, img.shape)
        padded = _pad_to_block(img, d)
        
        with st.spinner("Memproses..."):
            res = chaotic_diffusion(padded, seed)
            h, w, _ = img.shape
            res_cropped = res[:h, :w]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Citra Asli")
            st.image(numpy_to_pil(img))
        with col2:
            st.markdown("#### Hasil Henon Map")
            st.image(numpy_to_pil(res_cropped))
        
        st.markdown("---")
        st.markdown("### 📊 Histogram RGB")
        stage_images = {"Original": img, "Henon Map": res_cropped}
        fig = plot_rgb_line_histogram_grid(stage_images, figsize=(12, 4))
        st.pyplot(fig)
        plt.close()
        
        st.markdown("### 📈 Korelasi Piksel")
        df_orig = table_correlation_per_stage(img).T
        render_dataframe("Korelasi - Original", df_orig)
        
        df_proc = table_correlation_per_stage(res_cropped).T
        render_dataframe("Korelasi - Henon Map", df_proc)

# ============================================================
# PAGE: HISTOGRAM
# ============================================================
def page_histogram():
    st.markdown("### 📊 Histogram RGB")
    st.caption("Perbandingan distribusi warna di tiap tahap enkripsi")
    
    if st.session_state.image is None:
        st.info("Lakukan enkripsi terlebih dahulu di halaman Enkripsi")
        return
    
    stages = {}
    if st.session_state.image is not None:
        stages["Original"] = st.session_state.image
    if st.session_state.after_matrix is not None:
        h, w, _ = st.session_state.image.shape
        stages["GTSnm Matrix"] = st.session_state.after_matrix[:h, :w]
    if st.session_state.after_shuffle is not None:
        h, w, _ = st.session_state.image.shape
        stages["Shuffle"] = st.session_state.after_shuffle[:h, :w]
    if st.session_state.after_chaos is not None:
        h, w, _ = st.session_state.image.shape
        stages["Encrypted"] = st.session_state.after_chaos[:h, :w]
    
    if stages:
        fig = plot_rgb_line_histogram_grid(stages, figsize=(5 * len(stages), 4))
        st.pyplot(fig)
        plt.close()

# ============================================================
# PAGE: SECURITY
# ============================================================
def page_security():
    st.markdown("### 🛡 Analisis Keamanan")
    st.caption("MAE, NPCR, UACI, Korelasi Piksel, Entropy, dan Key Space")
    
    if st.session_state.image is None or st.session_state.after_chaos is None:
        st.info("Lakukan enkripsi terlebih dahulu di halaman Enkripsi")
        return
    
    if st.button("🛡 Jalankan Analisis", type="primary", use_container_width=True):
        h, w, _ = st.session_state.image.shape
        enc_cropped = st.session_state.after_chaos[:h, :w]
        
        # MAE, NPCR, UACI
        df_stats = table_statistical_parameters(st.session_state.image, enc_cropped)
        render_dataframe("MAE, NPCR, UACI", df_stats.reset_index())
        
        # Entropy
        df_entropy = table_entropy(st.session_state.image, enc_cropped)
        render_dataframe("Entropy", df_entropy.reset_index())
        
        # Correlation per stage
        stages = {
            "Original": st.session_state.image,
            "GTSnm Matrix": st.session_state.after_matrix[:h, :w],
            "Shuffle": st.session_state.after_shuffle[:h, :w],
            "Encrypted": enc_cropped
        }
        corr_tables = tables_correlation_all_stages(stages)
        for name, df in corr_tables.items():
            render_dataframe(f"Korelasi - {name}", df)
        
        # Pixel Average Distance
        df_dist = table_pixel_average_distance(st.session_state.image, enc_cropped)
        render_dataframe("Rata-rata Jarak Piksel Tetangga", df_dist.reset_index())
        
        # Key Space
        if st.session_state.d is not None:
            df_kspace, total, bits, summary = analyze_key_space(st.session_state.d)
            render_dataframe("Analisis Key Space", df_kspace)
            st.markdown("---")
            st.code(summary)
        
        # Export
        st.markdown("---")
        tables = {
            "MAE_NPCR_UACI": df_stats.reset_index(),
            "Entropy": df_entropy.reset_index(),
            "Pixel_Distance": df_dist.reset_index(),
        }
        if st.session_state.d is not None:
            tables["Key_Space"] = df_kspace
        
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        excel_buf.seek(0)
        
        st.download_button(
            label="📊 Export ke Excel",
            data=excel_buf,
            file_name="security_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ============================================================
# PAGE: ABOUT
# ============================================================
def page_about():
    st.markdown("### ℹ Tentang Aplikasi")
    
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-title">CipherFrame — Image Encryption Studio</div>
        </div>
        <p>CipherFrame adalah aplikasi untuk enkripsi citra digital menggunakan kombinasi tiga teknik:</p>
        
        <h4>1. GTSnm MATRIX</h4>
        <p>Matriks Pascal yang dimodifikasi dengan parameter pangkat l. Setiap blok piksel dikalikan dengan matriks enkripsi untuk memberikan diffusi pada level blok.</p>
        
        <h4>2. SHUFFLE (Permutasi Piksel)</h4>
        <p>Mengacak posisi piksel menggunakan seed dari password. Menggunakan permutasi acak yang deterministik untuk memberikan konfusi pada level piksel.</p>
        
        <h4>3. HENON MAP (Chaotic Diffusion)</h4>
        <p>Peta chaos Henon untuk menghasilkan sequence kunci. Difusi berantai (chained XOR) untuk memberikan sensitivitas tinggi terhadap perubahan input.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-title">Format File Metadata</div>
        </div>
        <p>Saat menyimpan citra terenkripsi, aplikasi juga membuat file <code>.meta.json</code> yang berisi:</p>
        <ul>
            <li><strong>original_shape</strong> : Ukuran citra asli (sebelum padding)</li>
            <li><strong>block_size</strong> : Parameter d</li>
            <li><strong>power</strong> : Parameter l</li>
            <li><strong>seed</strong> : Seed untuk RNG dan chaos</li>
        </ul>
        <p>⚠️ File ini <strong>WAJIB</strong> ada untuk proses dekripsi yang benar.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================
def main():
    init_session_state()
    render_sidebar()
    
    # Route to selected page
    page_router = {
        'home': page_home,
        'encrypt': page_encrypt,
        'decrypt': page_decrypt,
        'eval_gtsnm': page_eval_gtsmn,
        'eval_chaos': page_eval_chaos,
        'histogram': page_histogram,
        'security': page_security,
        'about': page_about,
    }
    
    page_func = page_router.get(st.session_state.page, page_home)
    page_func()

if __name__ == "__main__":
    main()
