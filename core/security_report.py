import numpy as np
import pandas as pd
from math import factorial, log2

from core.metrics import mae, npcr, uaci, adjacent_pixel_correlation, entropy


# =========================
# HELPER: rata-rata korelasi 1 channel utk 1 arah
# =========================
def _corr_per_channel(img, direction, sample_size=5000):
    """Hitung korelasi piksel tetangga untuk channel R, G, B sekaligus."""
    return [
        adjacent_pixel_correlation(img[:, :, ch], direction, sample_size)
        for ch in range(3)
    ]


def _avg_pixel_distance(img, sample_size=5000):
    """
    Rata-rata jarak absolut antar piksel tetangga (Horizontal/Vertical/
    Diagonal), dirata-ratakan dari 3 channel. Digunakan untuk Tabel (a).
    """
    h, w, _ = img.shape
    rng = np.random.default_rng(42)

    idx_i = np.empty(sample_size, dtype=np.int64)
    idx_j = np.empty(sample_size, dtype=np.int64)
    for k in range(sample_size):
        idx_i[k] = rng.integers(0, h - 1)
        idx_j[k] = rng.integers(0, w - 1)

    out = {}
    for direction in ['horizontal', 'vertical', 'diagonal']:
        if direction == 'horizontal':
            mask = idx_j < w - 1
            a = img[idx_i[mask], idx_j[mask]].astype(np.float32)
            b = img[idx_i[mask], idx_j[mask] + 1].astype(np.float32)
        elif direction == 'vertical':
            mask = idx_i < h - 1
            a = img[idx_i[mask], idx_j[mask]].astype(np.float32)
            b = img[idx_i[mask] + 1, idx_j[mask]].astype(np.float32)
        else:  # diagonal
            mask = (idx_i < h - 1) & (idx_j < w - 1)
            a = img[idx_i[mask], idx_j[mask]].astype(np.float32)
            b = img[idx_i[mask] + 1, idx_j[mask] + 1].astype(np.float32)

        out[direction] = float(np.mean(np.abs(a - b))) if len(a) else 0.0

    return out

# =========================
# TABEL ENTROPY
# =========================
def table_entropy(img_original, img_encrypted):
    """
    Perbandingan entropy citra asli dan terenkripsi
    """
    rows = []

    for name, img in [
        ("Gambar Asli", img_original),
        ("Gambar Terenkripsi", img_encrypted)
    ]:
        r = entropy(img[:, :, 0])
        g = entropy(img[:, :, 1])
        b = entropy(img[:, :, 2])
        total = entropy(img)

        rows.append([name, r, g, b, total])

    df = pd.DataFrame(
        rows,
        columns=["Citra", "R", "G", "B", "Keseluruhan"]
    )

    return df.set_index("Citra").round(4)

# =========================
# TABEL (a) PIXEL AVERAGE DISTANCE
# =========================
def table_pixel_average_distance(img_original, img_encrypted, sample_size=5000):
    """
    Baris: Gambar Asli, Gambar Terenkripsi
    Kolom: Between Two Adjacent Pixels -> Horizontal, Vertical, Diagonal
    """
    d_orig = _avg_pixel_distance(img_original, sample_size)
    d_enc = _avg_pixel_distance(img_encrypted, sample_size)

    df = pd.DataFrame(
        [
            [d_orig['horizontal'], d_orig['vertical'], d_orig['diagonal']],
            [d_enc['horizontal'], d_enc['vertical'], d_enc['diagonal']],
        ],
        index=["Gambar Asli", "Gambar Terenkripsi"],
        columns=pd.MultiIndex.from_product(
            [["Between Two Adjacent Pixels"], ["Horizontal", "Vertical", "Diagonal"]]
        ),
    ).round(4)
    return df


# =========================
# TABEL (b) KORELASI PER TAHAP
# =========================
def table_correlation_per_stage(img, sample_size=5000):
    """
    Baris: R, G, B
    Kolom: Horizontal (H), Vertical (V), Diagonal (D)
    Dipanggil sekali per tahap (Asli / Matrix / Shuffle / Chaos).
    """
    h_vals = _corr_per_channel(img, 'horizontal', sample_size)
    v_vals = _corr_per_channel(img, 'vertical', sample_size)
    d_vals = _corr_per_channel(img, 'diagonal', sample_size)

    df = pd.DataFrame(
        {
            "Horizontal (H)": h_vals,
            "Vertical (V)": v_vals,
            "Diagonal (D)": d_vals,
        },
        index=["R", "G", "B"],
    ).round(4)
    return df


def tables_correlation_all_stages(stage_images: dict, sample_size=5000):
    """
    stage_images: dict {nama_tahap: array_gambar}, contoh:
        {
          "Gambar Asli": img,
          "Gambar GTSnm Matrix": after_matrix,
          "Gambar Shuffle": after_shuffle,
          "Gambar Chaotic Henon Map": after_chaos,
        }
    Return: dict {nama_tahap: DataFrame}
    """
    return {
        name: table_correlation_per_stage(img, sample_size)
        for name, img in stage_images.items()
    }


# =========================
# TABEL (c) PERBANDINGAN KORELASI UMUM
# =========================
def table_correlation_comparison(img_original, img_shuffle, img_encrypted, sample_size=5000):
    """
    Baris: Gambar Asli, Gambar Shuffle, Gambar Encrypt
    Kolom: Horizontal, Vertical, Diagonal (rata-rata 3 channel)
    """
    rows = []
    for img in (img_original, img_shuffle, img_encrypted):
        h = np.mean(_corr_per_channel(img, 'horizontal', sample_size))
        v = np.mean(_corr_per_channel(img, 'vertical', sample_size))
        d = np.mean(_corr_per_channel(img, 'diagonal', sample_size))
        rows.append([h, v, d])

    df = pd.DataFrame(
        rows,
        index=["Gambar Asli", "Gambar Shuffle", "Gambar Encrypt"],
        columns=["Horizontal", "Vertical", "Diagonal"],
    ).round(4)
    return df


# =========================
# TABEL (d) MAE, NPCR, UACI
# =========================
def table_statistical_parameters(img_original, img_encrypted):
    """
    Baris: MAE, NPCR, UACI -- dihitung per channel (R,G,B) + keseluruhan.
    Memakai fungsi mae(), npcr(), uaci() yang sudah ada di metrics.py
    (rumus tidak diubah).
    """
    h = min(img_original.shape[0], img_encrypted.shape[0])
    w = min(img_original.shape[1], img_encrypted.shape[1])
    orig = img_original[:h, :w]
    enc = img_encrypted[:h, :w]

    rows = []
    for name, fn, suffix in [
        ("MAE (Mean Absolute Error)", mae, ""),
        ("NPCR (Number of Pixels Change Rate)", npcr, "%"),
        ("UACI (Unified Average Changing Intensity)", uaci, "%"),
    ]:
        r = fn(orig[:, :, 0], enc[:, :, 0])
        g = fn(orig[:, :, 1], enc[:, :, 1])
        b = fn(orig[:, :, 2], enc[:, :, 2])
        total = fn(orig, enc)
        rows.append([name, r, g, b, total])

    df = pd.DataFrame(rows, columns=["Parameter", "R", "G", "B", "Keseluruhan"])
    df[["R", "G", "B", "Keseluruhan"]] = df[["R", "G", "B", "Keseluruhan"]].round(4)
    return df.set_index("Parameter")


# =========================
# (e) ANALISIS KEY SPACE
# =========================
def analyze_key_space(block_size, password_length=10, seed_bits=32):
    """
    Mencetak & mengembalikan ringkasan ruang kunci (key space) sistem,
    berdasarkan parameter kunci yang SUDAH ADA di core/matrix.py:
      - Password ASCII (10 karakter, masing-masing 0-255)
      - Seed PRNG (32-bit, dipakai untuk shuffle + Henon map + permutasi S)
      - Matriks permutasi S berukuran block_size x block_size (d!)
      - Parameter pangkat l (matriks Pascal GTSnm, mod 256 -> 256 kemungkinan)

    Tidak mengubah algoritma generate_keys_from_password / generate_P_l;
    hanya menghitung kombinatorik dari parameter yang sudah dipakai.
    """
    password_space = 256 ** password_length          # ASCII per karakter
    seed_space = 2 ** seed_bits                       # ruang seed PRNG
    permutation_space = factorial(block_size)          # matriks S (permutasi)
    l_space = 256                                       # l mod 256

    total_space = password_space * permutation_space * l_space
    # seed_space tidak dikalikan terpisah karena seed diturunkan dari
    # password (bukan kunci independen tambahan) -- dicatat sebagai info.

    bits_password = log2(password_space)
    bits_total = log2(total_space)

    report = {
        "Komponen Kunci": [
            "Password (10 karakter ASCII)",
            f"Matriks Permutasi S ({block_size}x{block_size})",
            "Parameter Pangkat l (mod 256)",
            "Seed PRNG (turunan password)",
        ],
        "Ruang Kemungkinan": [
            f"256^10 = {password_space:.4e}",
            f"{block_size}! = {permutation_space:.4e}",
            f"256",
            f"2^32 = {seed_space:.4e} (turunan, bukan independen)",
        ],
        "Setara Bit": [
            f"{bits_password:.2f} bit",
            f"{log2(permutation_space):.2f} bit",
            f"{log2(l_space):.2f} bit",
            "-",
        ],
    }
    df = pd.DataFrame(report)

    summary_text = (
        f"Total Key Space (Password x Permutasi S x Parameter l)\n"
        f"= 256^10 x {block_size}! x 256\n"
        f"= {total_space:.4e}\n"
        f"~= 2^{bits_total:.2f}\n\n"
        f"Standar keamanan kriptografi modern menganggap ruang kunci\n"
        f">= 2^128 sudah tahan terhadap brute-force attack dengan\n"
        f"kemampuan komputasi saat ini."
    )

    return df, total_space, bits_total, summary_text


# =========================
# EXPORT: Markdown / LaTeX / Excel
# =========================
def df_to_markdown(df, **kwargs):
    """Wrapper tipis di atas DataFrame.to_markdown() (memerlukan paket 'tabulate')."""
    return df.to_markdown(**kwargs)


def df_to_latex(df, **kwargs):
    """Wrapper tipis di atas DataFrame.to_latex()."""
    return df.to_latex(**kwargs)


def export_all_to_excel(tables: dict, path):
    """
    tables: dict {nama_sheet: DataFrame}
    Menulis semua tabel ke satu file .xlsx, satu sheet per tabel.
    Nama sheet Excel dibatasi 31 karakter -> dipotong otomatis jika perlu.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet_name = name[:31] if len(name) > 31 else name
            df.to_excel(writer, sheet_name=sheet_name)
    return path

