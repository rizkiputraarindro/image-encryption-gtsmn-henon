"""
histogram_report.py
====================
Modul TAMBAHAN untuk menghasilkan 'Histogram Garis RGB' (line plot)
sebagai pengganti histogram bar lama -- sesuai standar penyusunan skripsi:
  - Kurva distribusi R, G, B digambar sebagai garis dalam satu bidang grafik
  - Background figure & axes berwarna putih bersih (facecolor='white')
  - Bisa diekspor ke PNG resolusi tinggi (dpi tinggi)

Tidak menyentuh algoritma enkripsi -- modul ini murni visualisasi.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # aman dipakai bersama Tkinter (FigureCanvasTkAgg override saat embed)
import matplotlib.pyplot as plt


RGB_COLORS = {"R": "#dc2626", "G": "#16a34a", "B": "#2563eb"}


def compute_rgb_histogram_counts(img, bins=256):
    """Hitung histogram count tiap channel R, G, B (vektor, tanpa loop piksel)."""
    counts = {}
    for idx, ch in enumerate(["R", "G", "B"]):
        counts[ch], _ = np.histogram(img[:, :, idx], bins=bins, range=(0, 256))
    return counts


def plot_rgb_line_histogram(img, title="Histogram RGB", ax=None, linewidth=1.1):
    """
    Gambar histogram garis RGB pada satu Axes (background putih).
    Jika ax=None, buat figure baru (dipakai untuk export tunggal).
    """
    counts = compute_rgb_histogram_counts(img)
    x = np.arange(256)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 3.2), facecolor="white")
    else:
        fig = ax.figure

    ax.set_facecolor("white")
    for ch, color in RGB_COLORS.items():
        ax.plot(x, counts[ch], color=color, linewidth=linewidth, label=ch)

    ax.set_title(title, fontsize=10, color="#1a1a2e")
    ax.set_xlabel("Intensitas Piksel", fontsize=8)
    ax.set_ylabel("Frekuensi", fontsize=8)
    ax.set_xlim(0, 255)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15)

    if own_fig:
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        return fig, ax
    return None, ax


def plot_rgb_line_histogram_grid(stage_images: dict, figsize=(16, 3.5)):
    """
    Gambar grid histogram garis untuk beberapa tahap sekaligus
    (1 baris x N kolom), background putih bersih untuk semua axes & figure.

    stage_images: dict {judul: array_gambar}, urutan dict dipertahankan.
    Return: matplotlib Figure (siap di-embed ke Tkinter atau disimpan PNG).
    """
    n = len(stage_images)
    fig, axes = plt.subplots(1, n, figsize=figsize, facecolor="white")
    fig.patch.set_facecolor("white")

    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, stage_images.items()):
        plot_rgb_line_histogram(img, title=title, ax=ax)

    fig.tight_layout(pad=1.5)
    return fig


def save_figure_high_res(fig, path, dpi=300):
    """
    Simpan figure ke PNG resolusi tinggi dengan latar belakang putih
    (facecolor dipaksa putih saat save, terlepas dari tema window).
    """
    fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
    return path
