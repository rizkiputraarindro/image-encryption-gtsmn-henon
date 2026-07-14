# core/encryption.py
import numpy as np
import time
from numba import njit
from core.matrix import (
    generate_P_power,
    generate_P_inverse,
    shuffle_image,
    unshuffle_image,
)

# =========================
# HENON MAP (NumPy, tanpa Numba)
# =========================
def henon_map_sequence(size, seed):
    rng = np.random.default_rng(seed)
    x = rng.random()
    y = rng.random()

    seq = np.empty(size, dtype=np.uint8)

    for i in range(size):
        x_new = 1 - 1.4 * x * x + y
        y_new = 0.3 * x
        x, y = x_new, y_new

        x = min(max(x, -2.0), 2.0)
        y = min(max(y, -0.5), 0.5)

        seq[i] = int((abs(x) % 1.0) * 255)

    return seq

# core/encryption.py

# =========================
# NUMBA KERNELS (Chained XOR / CBC-like)
# =========================
@njit(cache=True)
def diffusion_kernel(flat, key):
    """
    Enkripsi Difusi (Chained):
    C[0] = P[0] ^ K[0]
    C[i] = P[i] ^ C[i-1] ^ K[i]
    """
    out = np.empty_like(flat)
    out[0] = flat[0] ^ key[0]
    for i in range(1, flat.size):
        out[i] = flat[i] ^ out[i-1] ^ key[i]
    return out

@njit(cache=True)
def inverse_diffusion_kernel(flat, key):
    """
    Dekripsi Difusi (Chained):
    P[0] = C[0] ^ K[0]
    P[i] = C[i] ^ C[i-1] ^ K[i]
    """
    out = np.empty_like(flat)
    out[0] = flat[0] ^ key[0]
    for i in range(1, flat.size):
        out[i] = flat[i] ^ flat[i-1] ^ key[i]
    return out

def chaotic_diffusion(img, seed):
    """Menjalankan difusi chained pada gambar."""
    flat = img.ravel().astype(np.uint8)
    key = henon_map_sequence(flat.size, seed)
    return diffusion_kernel(flat, key).reshape(img.shape)

def inverse_chaotic_diffusion(img, seed):
    """Menjalankan inverse difusi chained pada gambar."""
    flat = img.ravel().astype(np.uint8)
    key = henon_map_sequence(flat.size, seed)
    return inverse_diffusion_kernel(flat, key).reshape(img.shape)

def encrypt_matrix(img, block_size, l, seed):
    h,w,c=img.shape
    encrypted=np.zeros_like(img,dtype=np.uint8)
    I=np.eye(block_size,dtype=np.int32)
    for ch in range(c):
        rng=np.random.default_rng(seed+ch)
        S=I[rng.permutation(block_size)]
        P=generate_P_power(block_size,l+ch)
        for i in range(0,h,block_size):
            for j in range(0,w,block_size):
                block=img[i:i+block_size,j:j+block_size,ch]
                if block.shape!=(block_size,block_size): continue
                temp=(block.astype(np.int32).T@S)%256
                E=(temp@P)%256
                encrypted[i:i+block_size,j:j+block_size,ch]=E.astype(np.uint8)
    return encrypted

def decrypt_matrix(enc_img, block_size, l, seed):
    h,w,c=enc_img.shape
    dec=np.zeros_like(enc_img,dtype=np.uint8)
    for ch in range(c):
        rng=np.random.default_rng(seed+ch)
        S=np.eye(block_size)[rng.permutation(block_size)]
        P_inv=generate_P_inverse(block_size,l+ch)
        S_inv=S.T
        for i in range(0,h,block_size):
            for j in range(0,w,block_size):
                block=enc_img[i:i+block_size,j:j+block_size,ch]
                if block.shape!=(block_size,block_size): continue
                temp=(block.astype(np.int32)@P_inv)%256
                W=((temp@S_inv)%256).T
                dec[i:i+block_size,j:j+block_size,ch]=W.astype(np.uint8)
    return dec

def _pad_to_block(img, block_size):
    h,w,c=img.shape
    nh=((h+block_size-1)//block_size)*block_size
    nw=((w+block_size-1)//block_size)*block_size
    if (nh,nw)==(h,w): return img
    out=np.zeros((nh,nw,c),dtype=img.dtype)
    out[:h,:w]=img
    return out

def encrypt_image(img, block_size, l, seed):
    padded=_pad_to_block(img,block_size)
    t=time.perf_counter(); after_matrix=encrypt_matrix(padded,block_size,l,seed); mt=time.perf_counter()-t
    t=time.perf_counter(); after_shuffle=shuffle_image(after_matrix,seed); st=time.perf_counter()-t
    t=time.perf_counter(); after_chaos=chaotic_diffusion(after_shuffle,seed); ct=time.perf_counter()-t
    profiling={"image_size":img.shape,"block_size":block_size,"power":l,"matrix_time":mt,"shuffle_time":st,"chaos_time":ct}
    return after_matrix,after_shuffle,after_chaos,after_chaos,profiling


def decrypt_image(enc, block_size, l, seed, original_shape):
    t=time.perf_counter(); dec=inverse_chaotic_diffusion(enc,seed); ict=time.perf_counter()-t
    t=time.perf_counter(); dec=unshuffle_image(dec,seed); ust=time.perf_counter()-t
    t=time.perf_counter(); dec=decrypt_matrix(dec,block_size,l,seed); mdt=time.perf_counter()-t
    profiling={"inverse_chaos_time":ict,"unshuffle_time":ust,"matrix_decrypt_time":mdt}
    h,w=original_shape[:2]  # <-- DIPERBAIKI: Ambil hanya h dan w
    return dec[:h,:w],profiling