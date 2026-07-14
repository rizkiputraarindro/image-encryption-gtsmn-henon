import numpy as np
from math import comb
from functools import lru_cache

# =========================
# PASCAL MATRIX (GTSM)
# =========================
@lru_cache(maxsize=None)
def generate_P_power(d, l):
    P = np.zeros((d, d), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            if i >= j:
                P[i, j] = comb(i - j + l - 1, l - 1) % 256
    return P


@lru_cache(maxsize=None)
def generate_P_inverse(d, l):

    P_inv = np.zeros((d, d), dtype=np.int32)

    for i in range(d):
        for j in range(i+1):
            P_inv[i,j]=((-1)**(i-j))*comb(l,i-j)

    return P_inv

# =========================
# PERMUTATION
# =========================
def generate_permutation_indices(size, seed):
    rng = np.random.default_rng(seed)
    return rng.permutation(size)

def shuffle_image(img, seed):
    h, w, c = img.shape
    flat = img.reshape(-1, c)
    perm = generate_permutation_indices(len(flat), seed)
    return flat[perm].reshape(h, w, c)

def unshuffle_image(img, seed):
    h, w, c = img.shape
    flat = img.reshape(-1, c)
    perm = generate_permutation_indices(len(flat), seed)
    inv = np.zeros_like(flat)
    inv[perm] = flat
    return inv.reshape(h, w, c)

def generate_permutation_matrix(d, seed):
    """
    Alias/jembatan agar app.py (yang memanggil generate_permutation_matrix)
    tetap berjalan TANPA mengubah algoritma asli di file ini.

    Algoritma shuffle/unshuffle yang sesungguhnya (lihat shuffle_image /
    unshuffle_image di atas) menurunkan urutan piksel langsung dari
    'seed' lewat generate_permutation_indices() -- tidak ada matriks Sn
    independen yang dipakai untuk shuffle citra.

    Nilai 'd' di sini merujuk pada ukuran blok GTSnm Matrix (state.d di
    app.py), bukan ukuran array permutasi shuffle (yang besarnya
    h*w*c). Fungsi ini hanya dipakai app.py untuk mengisi state.Sn
    (ditampilkan/disimpan sebagai info kunci), jadi dikembalikan array
    indeks permutasi berukuran d x d agar konsisten dengan d (bukan
    dipakai ulang untuk proses enkripsi/dekripsi sesungguhnya).
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(d)


def generate_keys_from_password(password, img_shape):
    """
    Generate parameter kunci dari password:
    - seed : untuk RNG / chaos
    - d    : ukuran blok matrix
    - l    : pangkat matrix

    Parameter:
    password : string (min 8 karakter)
    img_shape: (h, w, c)
    """

    if len(password) < 8:
        raise ValueError("Password minimal 10 karakter")

    ascii_vals = np.array([ord(c) for c in password], dtype=np.int64)

    # =========================
    # SEED (UNTUK CHAOS)
    # =========================
    seed = int(np.sum(ascii_vals * np.arange(1, len(ascii_vals)+1)) % (2**32 - 1))

    # =========================
    # BLOCK SIZE (d)
    # =========================
    h, w, _ = img_shape
    max_block = min(h, w) // 8

    if max_block < 2:
        max_block = 2

    d = int((np.sum(ascii_vals) % max_block) + 2)

    # =========================
    # POWER (l)
    # =========================
    l = int((np.prod(ascii_vals) % 5) + 3)  # range 3–7

    return seed, d, l