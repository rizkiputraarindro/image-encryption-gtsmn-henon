import numpy as np
from PIL import Image

# =========================
# LOAD IMAGE
# =========================
def load_image(path):
    return np.array(Image.open(path).convert('RGB'))

# =========================
# PADDING
# =========================
def pad_image(img, block_size):
    h, w, c = img.shape
    new_h = ((h + block_size - 1) // block_size) * block_size
    new_w = ((w + block_size - 1) // block_size) * block_size

    padded = np.zeros((new_h, new_w, c), dtype=img.dtype)
    padded[:h, :w] = img
    return padded

# =========================
# KEY GENERATION
# =========================
def generate_keys(password, img_shape):
    if len(password) != 10:
        raise ValueError("Password harus 10 karakter")

    h, w, _ = img_shape
    N = h * w

    ascii_vals = np.array([ord(c) for c in password], dtype=np.int64)

    weights = np.arange(1, len(ascii_vals) + 1)
    seed = int(np.sum((ascii_vals * weights) % N) % (2**32 - 1))

    ascii_str = ''.join([str(x) for x in ascii_vals])
    big_number = int(ascii_str)

    raw_block = big_number % N
    scaled = int(np.sqrt(np.sqrt(raw_block + 1)))

    max_block = min(h, w) // 8
    if max_block < 2:
        max_block = 2

    block_size = max(2, min(scaled, max_block))

    l = int((np.prod(ascii_vals[:3]) % 8) + 3)

    return seed, block_size, l