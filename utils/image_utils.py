# utils/image_utils.py
#
# File jembatan (TIDAK mengubah logika apa pun).
# app.py mengimpor "from utils.image_utils import load_image", sedangkan
# implementasi asli ada di core/preprocess.py. Modul ini hanya
# meneruskan (re-export) fungsi yang sudah ada.

from core.preprocess import load_image, pad_image

__all__ = ["load_image", "pad_image"]
