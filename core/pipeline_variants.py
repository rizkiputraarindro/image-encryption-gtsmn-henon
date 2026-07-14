# core/pipeline_variants.py
import numpy as np

from core.encryption import (
    encrypt_matrix,
    decrypt_matrix,
    encrypt_image,
    decrypt_image,
    henon_map_sequence,
    _pad_to_block,
)

from core.matrix import (
    shuffle_image,
    unshuffle_image,
)

# ==========================================================
# 1. TANPA GTSMN (hanya XOR chaos)
# ==========================================================
def encrypt_no_gtsmn(img, seed):
    img = img.astype(np.uint8)
    seq = henon_map_sequence(img.size, seed)
    key = (seq * 255).astype(np.uint8).reshape(img.shape)
    cipher = np.bitwise_xor(img, key)
    return cipher

def decrypt_no_gtsmn(cipher, seed):
    seq = henon_map_sequence(cipher.size, seed)
    key = (seq * 255).astype(np.uint8).reshape(cipher.shape)
    plain = np.bitwise_xor(cipher, key)
    return plain

# ==========================================================
# 2. GTSMN TANPA CHAOS
# ==========================================================
def encrypt_gtsmn_no_chaos(img, block_size, l, seed):
    padded = _pad_to_block(img, block_size)
    cipher = encrypt_matrix(padded, block_size, l, seed)
    return cipher

def decrypt_gtsmn_no_chaos(cipher, block_size, l, seed):
    plain = decrypt_matrix(cipher, block_size, l, seed)
    return plain

# ==========================================================
# 3. FULL (GTSnm + Shuffle + Chaos)
# ==========================================================
def encrypt_full(img, block_size, l, seed):
    return encrypt_image(img, block_size, l, seed)

def decrypt_full(cipher, block_size, l, seed, original_shape=None):
    """
    BUG FIX: original_shape sekarang diteruskan ke decrypt_image
    agar proses unpadding berjalan sesuai ukuran asli citra.
    """
    return decrypt_image(cipher, block_size, l, seed, original_shape)

__all__ = [
    "_pad_to_block",
    "encrypt_no_gtsmn", "decrypt_no_gtsmn",
    "encrypt_gtsmn_no_chaos", "decrypt_gtsmn_no_chaos",
    "encrypt_full", "decrypt_full",
]