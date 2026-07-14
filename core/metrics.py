import numpy as np
import pandas as pd

# =========================
# MAE (Mean Absolute Error)
# =========================
def mae(a, b):
    """
    Mengukur rata-rata selisih absolut antara dua citra.
    Cocok untuk evaluasi dekripsi (harus mendekati 0).
    """
    if a.shape != b.shape:
        raise ValueError("Ukuran citra harus sama")

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    return np.mean(np.abs(a - b))


# =========================
# NPCR (Number of Pixels Change Rate)
# =========================
def npcr(a, b):
    """
    Mengukur persentase piksel yang berubah.
    Digunakan untuk evaluasi sensitivity terhadap perubahan input.
    """
    if a.shape != b.shape:
        raise ValueError("Ukuran citra harus sama")

    return np.sum(a != b) / a.size * 100


# =========================
# UACI (Unified Average Changing Intensity)
# =========================
def uaci(a, b):
    """
    Mengukur rata-rata intensitas perubahan piksel.
    Ideal untuk citra terenkripsi ~33%.
    """
    if a.shape != b.shape:
        raise ValueError("Ukuran citra harus sama")

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    return np.mean(np.abs(a - b) / 255.0) * 100


# =========================
# ADJACENT PIXEL CORRELATION
# =========================
def adjacent_pixel_correlation(channel, direction='horizontal', sample_size=5000):
    """
    Mengukur korelasi antar piksel bertetangga.
    Semakin mendekati 0 → semakin baik untuk citra terenkripsi.
    """

    h, w = channel.shape
    rng = np.random.default_rng(42)

    idx_i = np.empty(sample_size, dtype=np.int64)
    idx_j = np.empty(sample_size, dtype=np.int64)

    # sampling tetap deterministik
    for k in range(sample_size):
        idx_i[k] = rng.integers(0, h - 1)
        idx_j[k] = rng.integers(0, w - 1)

    if direction == 'horizontal':
        mask = idx_j < w - 1
        x = channel[idx_i[mask], idx_j[mask]]
        y = channel[idx_i[mask], idx_j[mask] + 1]

    elif direction == 'vertical':
        mask = idx_i < h - 1
        x = channel[idx_i[mask], idx_j[mask]]
        y = channel[idx_i[mask] + 1, idx_j[mask]]

    elif direction == 'diagonal':
        mask = (idx_i < h - 1) & (idx_j < w - 1)
        x = channel[idx_i[mask], idx_j[mask]]
        y = channel[idx_i[mask] + 1, idx_j[mask] + 1]

    else:
        raise ValueError("Direction harus: horizontal, vertical, atau diagonal")

    if len(x) == 0:
        return 0

    return np.corrcoef(x, y)[0, 1]


# =========================
# PER CHANNEL CORRELATION
# =========================
def pixel_corr_channel(img, channel, direction='horizontal', sample_size=5000):
    return adjacent_pixel_correlation(
        img[:, :, channel],
        direction,
        sample_size
    )


# =========================
# EVALUASI KORELASI
# =========================
def evaluasi_korelasi(img_ori, img_enc):
    """
    Menghasilkan tabel korelasi:
    - horizontal
    - vertical
    - diagonal
    """

    directions = ['horizontal', 'vertical', 'diagonal']
    results = []

    for d in directions:
        ori_vals = []
        enc_vals = []

        for ch in range(3):
            ori_vals.append(pixel_corr_channel(img_ori, ch, d))
            enc_vals.append(pixel_corr_channel(img_enc, ch, d))

        results.append([
            d.capitalize(),
            np.mean(ori_vals),
            np.mean(enc_vals)
        ])

    return pd.DataFrame(
        results,
        columns=["Direction", "Original", "Encrypted"]
    )

# =========================
# ENTROPY
# =========================
def entropy(img):
    """
    Menghitung Shannon Entropy citra.
    Nilai ideal untuk citra terenkripsi: ~7.99
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    prob = hist / np.sum(hist)

    prob = prob[prob > 0]  # hindari log(0)

    return -np.sum(prob * np.log2(prob))