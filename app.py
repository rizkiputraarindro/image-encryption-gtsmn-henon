#!/usr/bin/env python3
"""
CipherFrame — Image Encryption Studio
Aplikasi desktop untuk enkripsi citra menggunakan:
  GTSnm Matrix  ·  Shuffle  ·  Henon Map (Chaotic Diffusion)
"""

import sys
import time
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    table_entropy,  # <-- TAMBAHKAN INI
)
from utils.image_utils import load_image

# ============================================================
# DESIGN TOKENS
# ============================================================
FONT_BODY    = "Segoe UI"
FONT_DISPLAY = "Segoe UI Semibold"
FONT_MONO    = "Cascadia Mono"

BG          = "#0B0F19"
BG_SOFT     = "#0F1422"
SURFACE     = "#121826"
SURFACE_ALT = "#171F30"
BORDER      = "#232B3D"
TEXT        = "#E8EAF0"
SUBTEXT     = "#7A8194"
MUTED       = "#4B5468"

ACCENT      = "#00D9A3"
ACCENT_DIM  = "#00A87E"
ACCENT_SOFT = "#0E2A24"
DANGER      = "#FF5470"
DANGER_DIM  = "#E03F5C"
DANGER_SOFT = "#2A1620"

C_PRIMARY   = (ACCENT, ACCENT_DIM, "#06140F")
C_SECONDARY = (SURFACE, SURFACE_ALT, TEXT)
C_DANGER    = (DANGER, DANGER_DIM, "#1A0408")
C_GHOST     = (BG, SURFACE, SUBTEXT)


# ============================================================
# SCROLLABLE FRAME HELPER
# ============================================================
def make_scrollable(parent, bg=BG):
    """Buat frame yang dapat di-scroll vertikal."""
    canvas = tk.Canvas(parent, bg=bg, highlightthickness=0)
    vsb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = tk.Frame(canvas, bg=bg)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>",
               lambda e: canvas.itemconfig(window_id, width=e.width))

    def _on_wheel(e):
        canvas.yview_scroll(-1 * (e.delta // 120), "units")

    def _enter(e):
        canvas.bind_all("<MouseWheel>", _on_wheel)

    def _leave(e):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", _enter)
    canvas.bind("<Leave>", _leave)

    return inner


# ============================================================
# ROUNDED BUTTON
# ============================================================
class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, color=C_PRIMARY,
                 width=160, height=38, radius=10, font_size=10,
                 bold=False, bg_override=None, **kw):
        bg = bg_override if bg_override is not None else parent.cget("bg")
        super().__init__(parent, width=width, height=height,
                         bg=bg, highlightthickness=0, **kw)
        self.command   = command
        self.bg_n      = color[0]
        self.bg_h      = color[1]
        self.fg        = color[2]
        self.radius    = radius
        self.w         = width
        self.h         = height
        self.font_size = font_size
        self.font_w    = "bold" if bold else "normal"
        self.text_str  = text
        self._draw(self.bg_n)
        self.bind("<Enter>",    lambda e: self._safe_draw(self.bg_h))
        self.bind("<Leave>",    lambda e: self._safe_draw(self.bg_n))
        self.bind("<Button-1>", lambda e: self._click())

    def _safe_draw(self, bg):
        try:
            if self.winfo_exists():
                self._draw(bg)
        except tk.TclError:
            pass

    def destroy(self):
        try:
            self.unbind("<Enter>")
            self.unbind("<Leave>")
            self.unbind("<Button-1>")
        except tk.TclError:
            pass
        super().destroy()

    def _draw(self, bg):
        self.delete("all")
        r, w, h = self.radius, self.w, self.h
        outline = BORDER if bg == SURFACE else bg
        self.create_polygon(
            r, 0, w - r, 0, w, 0, w, r,
            w, h - r, w, h, w - r, h, r, h,
            0, h, 0, h - r, 0, r, 0, 0, r, 0,
            smooth=True, fill=bg, outline=outline,
            width=1 if bg == SURFACE else 0
        )
        self.create_text(w // 2, h // 2, text=self.text_str, fill=self.fg,
                         font=(FONT_BODY, self.font_size, self.font_w))

    def set_enabled(self, enabled: bool):
        if enabled:
            self.bind("<Button-1>", lambda e: self._click())
            self._safe_draw(self.bg_n)
        else:
            self.unbind("<Button-1>")
            self._safe_draw(MUTED)

    def _click(self):
        self._draw(self.bg_h)
        self.after(100, lambda: self._safe_draw(self.bg_n))
        if self.command:
            self.command()


def btn(parent, text, cmd, color=C_PRIMARY, w=160, h=36, fs=10, bold=False):
    return RoundedButton(parent, text, cmd, color=color, width=w, height=h,
                         font_size=fs, bold=bold)


# ============================================================
# PASSWORD DIALOG
# ============================================================
class PasswordDialog(tk.Toplevel):
    KEY_OPEN   = "🔓"
    KEY_CLOSED = "🔒"

    def __init__(self, parent, title="Masukkan Password"):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.configure(bg=SURFACE)
        self.grab_set()
        self.result = None

        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width()  // 2 - 210
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 110
        self.geometry(f"420x230+{px}+{py}")

        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")
        tk.Label(self, text="🔐", font=("Segoe UI Emoji", 20), bg=SURFACE).pack(pady=(18, 0))
        tk.Label(self, text=title, font=(FONT_BODY, 13, "bold"), bg=SURFACE, fg=TEXT).pack(pady=(4, 2))
        tk.Label(self, text="Minimal 8 karakter", font=(FONT_BODY, 9), bg=SURFACE, fg=SUBTEXT).pack(pady=(0, 14))

        ef = tk.Frame(self, bg=SURFACE)
        ef.pack(padx=36, fill="x")

        self._pwd_var  = tk.StringVar()
        self._show_pwd = False

        entry_wrap = tk.Frame(ef, bg=BG_SOFT, highlightthickness=1,
                              highlightbackground=BORDER, highlightcolor=ACCENT)
        entry_wrap.pack(side="left", fill="x", expand=True)

        self._entry = tk.Entry(entry_wrap, textvariable=self._pwd_var,
                               font=(FONT_MONO, 12), relief="flat", show="●",
                               bg=BG_SOFT, fg=TEXT, insertbackground=ACCENT, width=22)
        self._entry.pack(ipady=7, padx=10, fill="x", expand=True)
        self._entry.focus_set()

        self._key_btn = tk.Label(ef, text=self.KEY_CLOSED, font=("Segoe UI Emoji", 14),
                                 bg=SURFACE, cursor="hand2")
        self._key_btn.pack(side="left", padx=(8, 0))
        self._key_btn.bind("<Button-1>", lambda e: self._toggle())

        bf = tk.Frame(self, bg=SURFACE)
        bf.pack(pady=18)
        btn(bf, "Batal", self.destroy, C_SECONDARY, 110, 36).pack(side="left", padx=6)
        btn(bf, "Konfirmasi", self._ok, C_PRIMARY, 130, 36, bold=True).pack(side="left", padx=6)

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.wait_window()

    def _toggle(self):
        self._show_pwd = not self._show_pwd
        self._entry.config(show="" if self._show_pwd else "●")
        self._key_btn.config(text=self.KEY_OPEN if self._show_pwd else self.KEY_CLOSED)

    def _ok(self):
        pwd = self._pwd_var.get()
        if len(pwd) < 8:
            messagebox.showerror("Peringatan", "Password minimal 8 karakter!", parent=self)
            return
        self.result = pwd
        self.destroy()


def ask_password(parent, title="Masukkan Password"):
    dlg = PasswordDialog(parent, title)
    return dlg.result


# ============================================================
# GLOBAL STATE
# ============================================================
class State:
    def __init__(self):
        self.image          = None
        self.after_matrix   = None
        self.after_shuffle  = None
        self.after_chaos    = None
        self.enc_full       = None
        self.decrypted      = None
        self.Sn             = None
        self.perm           = None
        self.d = self.l = self.seed = None
        self.encrypt_time   = 0
        self.decrypt_time   = 0

state = State()


# ============================================================
# METADATA HELPERS (UNTUK DEKRIPSI)
# ============================================================
def save_encrypted_with_metadata(arr, original_shape, d, l, seed, parent, default_name="terenkripsi.png"):
    """
    Simpan citra terenkripsi BESERTA file metadata (.meta.json).
    Metadata berisi: original_shape, block_size, power, seed
    """
    if arr is None:
        messagebox.showwarning("Peringatan", "Belum ada gambar terenkripsi.", parent=parent)
        return
    
    path = filedialog.asksaveasfilename(
        parent=parent,
        initialfile=default_name,
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("BMP", "*.bmp")])  # Hanya PNG/BMP untuk lossless
    
    if not path:
        return
    
    try:
        # Simpan gambar
        Image.fromarray(arr.astype(np.uint8)).save(path)
        
        # Simpan metadata
        meta_path = path.rsplit('.', 1)[0] + '.meta.json'
        metadata = {
            "original_shape": [original_shape[0], original_shape[1], original_shape[2]],
            "block_size": int(d),
            "power": int(l),
            "seed": int(seed)
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        messagebox.showinfo("Sukses", 
            f"Citra terenkripsi tersimpan:\n{path}\n\n"
            f"Metadata tersimpan:\n{meta_path}\n\n"
            f"⚠ Pastikan kedua file tetap berada di folder yang sama untuk dekripsi!",
            parent=parent)
    except Exception as e:
        messagebox.showerror("Error", f"Gagal menyimpan: {e}", parent=parent)


def load_encrypted_with_metadata(parent):
    """
    Muat citra terenkripsi dan metadata (jika ada).
    Returns: (image_array, metadata_dict_or_None, file_path)
    """
    path = filedialog.askopenfilename(
        parent=parent,
        filetypes=[("Encrypted Image", "*.png *.bmp *.tiff")])
    
    if not path:
        return None, None, None
    
    try:
        img = load_image(path)
    except Exception:
        messagebox.showerror("Error", "File gambar tidak valid atau format tidak didukung.", parent=parent)
        return None, None, None
    
    # Coba muat metadata
    meta_path = path.rsplit('.', 1)[0] + '.meta.json'
    metadata = None
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            # Konversi list ke tuple untuk original_shape
            if 'original_shape' in metadata:
                metadata['original_shape'] = tuple(metadata['original_shape'])
        except Exception:
            metadata = None
    
    return img, metadata, path


# ============================================================
# HELPERS
# ============================================================
def make_img_box(parent, title, col=0, box_size=170, save_cmd=None):
    frame = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                     highlightbackground=BORDER)
    frame.grid(row=0, column=col, padx=10, pady=8)

    tk.Label(frame, text=title, font=(FONT_MONO, 9, "bold"),
             bg=SURFACE, fg=TEXT).pack(pady=(8, 4))

    img_container = tk.Frame(frame, bg="#1A1A2E", width=box_size, height=box_size)
    img_container.pack_propagate(False)
    img_container.pack(padx=10, pady=6)

    lbl = tk.Label(img_container, bg="#1A1A2E", fg=MUTED,
                   text="Belum ada\ngambar", font=(FONT_BODY, 8),
                   justify="center")
    lbl.pack(expand=True)

    save_btn = btn(frame, "↓ Simpan", save_cmd or (lambda: None),
                   C_SECONDARY, 120, 30, fs=9)
    save_btn.pack(pady=(4, 10))

    return lbl, save_btn


def show_img(arr, box, max_size=160):
    if arr is None:
        return
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    box.configure(image=photo, text="")
    box.image = photo


def save_array_as_image(arr, parent, default_name="image.png"):
    if arr is None:
        messagebox.showwarning("Peringatan", "Belum ada gambar pada tahap ini.")
        return
    path = filedialog.asksaveasfilename(
        parent=parent,
        initialfile=default_name,
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
    if path:
        try:
            Image.fromarray(arr.astype(np.uint8)).save(path)
            messagebox.showinfo("Sukses", f"Tersimpan:\n{path}", parent=parent)
        except Exception:
            messagebox.showerror("Error", "Gagal menyimpan gambar.", parent=parent)


def make_key_chip(parent, label, value):
    card = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                    highlightbackground=BORDER)
    tk.Label(card, text=label, font=(FONT_BODY, 8), bg=SURFACE,
             fg=SUBTEXT, anchor="w").pack(fill="x", padx=12, pady=(8, 0))
    tk.Label(card, text=value, font=(FONT_MONO, 14, "bold"),
             bg=SURFACE, fg=ACCENT, anchor="w").pack(fill="x", padx=12, pady=(0, 9))
    return card


def page_header(parent, icon, title, subtitle):
    head = tk.Frame(parent, bg=BG)
    head.pack(fill="x", padx=28, pady=(24, 6))
    row = tk.Frame(head, bg=BG)
    row.pack(fill="x", anchor="w")
    tk.Label(row, text=icon, font=("Segoe UI Emoji", 18), bg=BG, fg=ACCENT).pack(side="left")
    tk.Label(row, text=title, font=(FONT_BODY, 18, "bold"), bg=BG, fg=TEXT).pack(
        side="left", padx=(10, 0))
    tk.Label(head, text=subtitle, font=(FONT_BODY, 10), bg=BG, fg=SUBTEXT,
             anchor="w").pack(fill="x", pady=(4, 0))
    tk.Frame(head, bg=BORDER, height=1).pack(fill="x", pady=(14, 0))
    return head


def setup_treeview_style(widget):
    style = ttk.Style(widget)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Cipher.Treeview",
                    background=BG_SOFT, fieldbackground=BG_SOFT,
                    foreground=TEXT, rowheight=26, borderwidth=0,
                    font=(FONT_MONO, 9))
    style.configure("Cipher.Treeview.Heading",
                    background=SURFACE_ALT, foreground=ACCENT,
                    font=(FONT_BODY, 9, "bold"), relief="flat", borderwidth=0)
    style.map("Cipher.Treeview.Heading", background=[("active", SURFACE_ALT)])
    style.map("Cipher.Treeview",
              background=[("selected", ACCENT_SOFT)],
              foreground=[("selected", ACCENT)])
    style.layout("Cipher.Treeview", [
        ("Cipher.Treeview.treearea", {"sticky": "nswe"})
    ])


def render_table_section(container, title, df, note=None):
    section = tk.Frame(container, bg=SURFACE, highlightthickness=1,
                       highlightbackground=BORDER)
    section.pack(fill="x", pady=8)

    head = tk.Frame(section, bg=SURFACE_ALT)
    head.pack(fill="x")
    tk.Label(head, text=title, font=(FONT_BODY, 11, "bold"),
             bg=SURFACE_ALT, fg=TEXT, anchor="w").pack(
                 fill="x", padx=14, pady=(9, 2 if note else 9))
    if note:
        tk.Label(head, text=note, font=(FONT_BODY, 8), bg=SURFACE_ALT, fg=SUBTEXT,
                 anchor="w", justify="left").pack(fill="x", padx=14, pady=(0, 9))

    if isinstance(df.columns, pd.MultiIndex):
        col_labels = [" / ".join(map(str, c)).strip(" /") for c in df.columns]
    else:
        col_labels = [str(c) for c in df.columns]

    cols = ["(index)"] + col_labels
    tv = ttk.Treeview(section, columns=cols, show="headings",
                      height=min(len(df.index), 15), style="Cipher.Treeview")
    for c in cols:
        tv.heading(c, text=c)
        tv.column(c, width=max(90, int(700 / len(cols))), anchor="center")

    for idx, row in zip(df.index, df.itertuples(index=False)):
        values = [str(idx)] + [
            f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else str(v)
            for v in row
        ]
        tv.insert("", "end", values=values)

    tv.pack(fill="x", padx=14, pady=(8, 14))
    return tv


def render_text_section(container, title, text):
    section = tk.Frame(container, bg=SURFACE, highlightthickness=1,
                       highlightbackground=BORDER)
    section.pack(fill="x", pady=8)
    head = tk.Frame(section, bg=SURFACE_ALT)
    head.pack(fill="x")
    tk.Label(head, text=title, font=(FONT_BODY, 11, "bold"),
             bg=SURFACE_ALT, fg=TEXT, anchor="w").pack(fill="x", padx=14, pady=9)
    tk.Label(section, text=text, font=(FONT_MONO, 9), bg=SURFACE, fg=TEXT,
             anchor="w", justify="left").pack(fill="x", padx=14, pady=(8, 14))


def render_evaluation_results(container, img_orig, img_proc, name):
    """Fungsi reusable untuk menampilkan Histogram dan Tabel Korelasi."""
    for w in container.winfo_children():
        w.destroy()

    tk.Label(container, text="HISTOGRAM RGB", font=(FONT_MONO, 9, "bold"),
             bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(10, 8))
    
    chart_card = tk.Frame(container, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
    chart_card.pack(fill="x", pady=(0, 16))
    
    stage_images = {
        "Original": img_orig,
        name: img_proc
    }
    fig = plot_rgb_line_histogram_grid(stage_images, figsize=(10, 3.5))
    canvas_widget = FigureCanvasTkAgg(fig, master=chart_card)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill="x", padx=10, pady=10)

    tk.Label(container, text="ADJACENT PIXEL CORRELATION", font=(FONT_MONO, 9, "bold"),
             bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(10, 8))

    df_orig = table_correlation_per_stage(img_orig).T
    df_orig.index.name = "Direction"
    render_table_section(container, "Korelasi Piksel - Original", df_orig)

    df_proc = table_correlation_per_stage(img_proc).T
    df_proc.index.name = "Direction"
    render_table_section(container, f"Korelasi Piksel - {name}", df_proc)


# ============================================================
# APP SHELL
# ============================================================
class App(tk.Tk):
    NAV_ITEMS = [
        ("home",       "🏠", "Beranda"),
        ("encrypt",    "🔐", "Enkripsi"),
        ("decrypt",    "🗝", "Dekripsi"),
        ("eval_gtsmn", "📐", "Eval GTSnm"),
        ("eval_chaos", "〰", "Eval Chaotic"),
        ("cmp_gtsmn",  "⚖",  "Vs GTSnm"),
        ("cmp_chaos",  "🌀", "Vs Chaotic"),
        ("histogram",  "📊", "Histogram"),
        ("security",   "🛡",  "Keamanan"),
        ("about",      "ℹ",  "Tentang"),
    ]

    def __init__(self):
        super().__init__()
        self.title("CipherFrame — Image Encryption Studio (GTSnm Algorithm)")
        self.geometry("1320x900")
        self.minsize(1000, 680)
        self.configure(bg=BG)
        self.resizable(True, True)

        self._build_layout()

        self.frames = {}
        for key, F in (("home",       Home),
                       ("encrypt",    EncryptPage),
                       ("decrypt",    DecryptPage),
                       ("eval_gtsmn", EvalGtsmnPage),
                       ("eval_chaos", EvalChaosPage),
                       ("cmp_gtsmn",  GtsmnComparisonPage),
                       ("cmp_chaos",  ChaosComparisonPage),
                       ("histogram",  HistogramPage),
                       ("security",   SecurityPage),
                       ("about",      AboutPage)):
            f = F(self.content_area, self)
            self.frames[key] = f
            f.grid(row=0, column=0, sticky="nsew")

        self.show("home")

    def _build_layout(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        navbar = tk.Frame(self, bg=SURFACE, height=58)
        navbar.grid(row=0, column=0, sticky="ew")
        navbar.grid_propagate(False)
        navbar.grid_columnconfigure(1, weight=1)

        brand = tk.Frame(navbar, bg=SURFACE)
        brand.grid(row=0, column=0, sticky="w", padx=(20, 0))
        brand_row = tk.Frame(brand, bg=SURFACE)
        brand_row.pack(anchor="w", pady=(0, 1))
        tk.Label(brand_row, text="⛓", font=("Segoe UI Emoji", 14), bg=SURFACE,
                 fg=ACCENT).pack(side="left")
        tk.Label(brand_row, text="CipherFrame", font=(FONT_BODY, 13, "bold"),
                 bg=SURFACE, fg=TEXT).pack(side="left", padx=(7, 0))
        tk.Label(brand, text="GTSnm · Shuffle · Henon Map", font=(FONT_BODY, 7),
                 bg=SURFACE, fg=SUBTEXT).pack(anchor="w")

        self._nav_buttons = {}
        nav_wrap = tk.Frame(navbar, bg=SURFACE)
        nav_wrap.grid(row=0, column=1)
        for key, icon, label in self.NAV_ITEMS:
            self._nav_buttons[key] = self._make_nav_button(nav_wrap, key, icon, label)

        foot = tk.Frame(navbar, bg=SURFACE)
        foot.grid(row=0, column=2, sticky="e", padx=(0, 20))
        btn(foot, "Keluar", self.quit_app, color=C_DANGER, w=96, h=34, fs=9).pack()

        tk.Frame(self, bg=BORDER, height=1).grid(row=0, column=0, sticky="sew")

        self.content_area = tk.Frame(self, bg=BG)
        self.content_area.grid(row=1, column=0, sticky="nsew")
        self.content_area.grid_rowconfigure(0, weight=1)
        self.content_area.grid_columnconfigure(0, weight=1)

    def _make_nav_button(self, parent, key, icon, label):
        col = tk.Frame(parent, bg=SURFACE, cursor="hand2")
        col.pack(side="left", padx=3)

        inner = tk.Frame(col, bg=SURFACE)
        inner.pack(fill="x", padx=12, pady=9)

        icon_lbl = tk.Label(inner, text=icon, font=("Segoe UI Emoji", 10),
                            bg=SURFACE, fg=SUBTEXT)
        icon_lbl.pack(side="left")
        text_lbl = tk.Label(inner, text=label, font=(FONT_BODY, 10),
                            bg=SURFACE, fg=SUBTEXT, anchor="w")
        text_lbl.pack(side="left", padx=(7, 0))

        indicator = tk.Frame(col, bg=SURFACE, height=2)
        indicator.pack(fill="x", side="bottom")

        widgets = (col, indicator, inner, icon_lbl, text_lbl)
        for w in widgets:
            w.bind("<Button-1>", lambda e, k=key: self.show(k))
        return widgets

    def show(self, key):
        for k, widgets in self._nav_buttons.items():
            col, indicator, inner, icon_lbl, text_lbl = widgets
            active = (k == key)
            bg = SURFACE_ALT if active else SURFACE
            fg = ACCENT if active else SUBTEXT
            col.configure(bg=bg)
            inner.configure(bg=bg)
            icon_lbl.configure(bg=bg, fg=fg)
            text_lbl.configure(bg=bg, fg=fg,
                               font=(FONT_BODY, 10, "bold" if active else "normal"))
            indicator.configure(bg=ACCENT if active else SURFACE)
        self.frames[key].tkraise()
        if key == "histogram":
            self.frames[key].refresh()

    def quit_app(self):
        self.destroy()
        sys.exit()


# ============================================================
# HOME
# ============================================================
class Home(tk.Frame):
    CARDS = [
        ("encrypt",    "🔐", "Enkripsi Citra",
         "Proses citra lewat GTSnm Matrix, Shuffle, dan Henon Map."),
        ("decrypt",    "🗝", "Dekripsi File",
         "Buka citra terenkripsi tersimpan dan pulihkan dengan password."),
        ("eval_gtsmn", "📐", "Evaluasi GTSnm",
         "Analisis statistik GTSnm Matrix (tanpa Shuffle & Chaotic)."),
        ("eval_chaos", "〰", "Evaluasi Chaotic",
         "Analisis statistik Difusi Henon Map (tanpa GTSnm & Shuffle)."),
        ("cmp_gtsmn",  "⚖",  "Vs GTSnm",
         "Bandingkan enkripsi dengan dan tanpa transformasi GTSnm."),
        ("cmp_chaos",  "🌀", "Vs Chaotic",
         "Bandingkan GTSnm dengan dan tanpa difusi Henon Map."),
        ("histogram",  "📊", "Histogram RGB",
         "Bandingkan distribusi warna di tiap tahap enkripsi."),
        ("security",   "🛡",  "Analisis Keamanan",
         "Hitung MAE, NPCR, UACI, korelasi piksel, dan key space."),
        ("about",      "ℹ",  "Tentang Aplikasi",
         "Penjelasan algoritma dan referensi yang digunakan."),
    ]

    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)

        wrap = tk.Frame(self, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(wrap, text="⛓", font=("Segoe UI Emoji", 28), bg=BG,
                 fg=ACCENT).pack()
        tk.Label(wrap, text="Image Encryption Studio",
                 font=(FONT_BODY, 24, "bold"), bg=BG, fg=TEXT).pack(pady=(6, 4))
        tk.Label(wrap, text="GTSnm Algorithm  +  Shuffle  +  Henon Map (Chaotic)",
                 font=(FONT_MONO, 10), bg=BG, fg=SUBTEXT).pack(pady=(0, 34))

        grid = tk.Frame(wrap, bg=BG)
        grid.pack()
        for i, (key, icon, title, desc) in enumerate(self.CARDS):
            r, c = divmod(i, 2)
            self._make_card(grid, key, icon, title, desc, app).grid(
                row=r, column=c, padx=8, pady=8, sticky="nsew")

    def _make_card(self, parent, key, icon, title, desc, app):
        card = tk.Frame(parent, bg=SURFACE, width=320, height=120,
                        highlightthickness=1, highlightbackground=BORDER,
                        cursor="hand2")
        card.pack_propagate(False)

        accent_bar = tk.Frame(card, bg=BORDER, width=4)
        accent_bar.pack(side="left", fill="y")

        body = tk.Frame(card, bg=SURFACE)
        body.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        head = tk.Frame(body, bg=SURFACE)
        head.pack(fill="x", anchor="w")
        tk.Label(head, text=icon, font=("Segoe UI Emoji", 14), bg=SURFACE,
                 fg=ACCENT).pack(side="left")
        tk.Label(head, text=title, font=(FONT_BODY, 12, "bold"), bg=SURFACE,
                 fg=TEXT).pack(side="left", padx=(8, 0))

        tk.Label(body, text=desc, font=(FONT_BODY, 9), bg=SURFACE, fg=SUBTEXT,
                 anchor="w", justify="left", wraplength=270).pack(
                     fill="x", pady=(8, 0), anchor="w")

        def on_enter(e):
            card.configure(highlightbackground=ACCENT)
            accent_bar.configure(bg=ACCENT)

        def on_leave(e):
            card.configure(highlightbackground=BORDER)
            accent_bar.configure(bg=BORDER)

        widgets = [card, body, head] + list(head.winfo_children()) + list(body.winfo_children())
        for w in widgets:
            w.bind("<Button-1>", lambda e, k=key: app.show(k))
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)

        return card


# ============================================================
# ENCRYPT PAGE
# ============================================================
class EncryptPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self._app = app
        self.main = make_scrollable(self, BG)
        self._build(self.main, app)

    def _build(self, main, app):
        page_header(main, "🔐", "Enkripsi Citra",
                    "GTSnm Algorithm  ·  Shuffle  ·  Henon Map (Chaotic)")

        body = tk.Frame(main, bg=BG)
        body.pack(fill="x", padx=28)

        toolbar = tk.Frame(body, bg=BG)
        toolbar.pack(fill="x", pady=(18, 0))

        btn(toolbar, "📂  Pilih Gambar", self.load, color=C_PRIMARY,
            w=190, h=40, bold=True).pack(side="left")

        self.key_strip = tk.Frame(toolbar, bg=BG)
        self.key_strip.pack(side="left", padx=(16, 0))
        self._render_key_chips(None, None, None)

        tk.Label(body, text="ALUR PROSES ENKRIPSI", font=(FONT_MONO, 9, "bold"),
                 bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(26, 8))

        stage_card = tk.Frame(body, bg=SURFACE, highlightthickness=1,
                              highlightbackground=BORDER)
        stage_card.pack(fill="x", pady=(0, 4))
        enc_frame = tk.Frame(stage_card, bg=SURFACE)
        enc_frame.pack(pady=14, padx=8)

        self.box_orig, self.btn_save_orig = make_img_box(
            enc_frame, "1 · Citra Asli", 0, save_cmd=self.save_orig)
        self.box_matrix, self.btn_save_matrix = make_img_box(
            enc_frame, "2 · GTSnm Matrix", 1, save_cmd=self.save_matrix)
        self.box_shuffle, self.btn_save_shuffle = make_img_box(
            enc_frame, "3 · Shuffle", 2, save_cmd=self.save_shuffle)
        self.box_chaos, self.btn_save_chaos = make_img_box(
            enc_frame, "4 · Henon Map (Final)", 3, save_cmd=self.save_encrypted)

        action_card = tk.Frame(body, bg=SURFACE, highlightthickness=1,
                               highlightbackground=BORDER)
        action_card.pack(fill="x", pady=(18, 0))
        gf = tk.Frame(action_card, bg=SURFACE)
        gf.pack(pady=16, padx=16)

        BW, BH = 200, 40
        btn(gf, "🔐 Enkripsi", self.encrypt, C_PRIMARY, BW, BH, bold=True).grid(
            row=0, column=0, padx=6, pady=4)
        btn(gf, "↓ Simpan Hasil", self.save_encrypted, C_SECONDARY, BW, BH).grid(
            row=0, column=1, padx=6, pady=4)
        btn(gf, "📊 Histogram", lambda: app.show("histogram"), C_GHOST, BW, BH).grid(
            row=0, column=2, padx=6, pady=4)
        btn(gf, "🛡 Analisis Keamanan", lambda: app.show("security"), C_GHOST, BW, BH).grid(
            row=0, column=3, padx=6, pady=4)

        self.info_card = tk.Frame(body, bg=BG)
        self.info_card.pack(fill="x", pady=(14, 4))
        self.info = tk.Label(self.info_card, text="Status: menunggu gambar…",
                             font=(FONT_MONO, 9), bg=BG, fg=SUBTEXT)
        self.info.pack(anchor="w")

        prof_label_row = tk.Frame(body, bg=BG)
        prof_label_row.pack(fill="x", pady=(14, 4))
        tk.Label(prof_label_row, text="PERFORMANCE PROFILING",
                 font=(FONT_MONO, 9, "bold"), bg=BG, fg=SUBTEXT).pack(side="left")

        prof_card = tk.Frame(body, bg=SURFACE, highlightthickness=1,
                             highlightbackground=BORDER)
        prof_card.pack(fill="x", pady=(0, 28))

        prof_header = tk.Frame(prof_card, bg=SURFACE_ALT)
        prof_header.pack(fill="x")
        tk.Label(prof_header, text="Waktu Eksekusi per Tahap",
                 font=(FONT_BODY, 10, "bold"), bg=SURFACE_ALT, fg=TEXT,
                 anchor="w").pack(fill="x", padx=14, pady=8)

        self.prof_text = ScrolledText(
            prof_card, height=14, font=(FONT_MONO, 9),
            bg=BG_SOFT, fg=ACCENT, insertbackground=ACCENT,
            relief="flat", state="disabled", wrap="word",
            bd=0, padx=14, pady=10)
        self.prof_text.pack(fill="x", padx=2, pady=(0, 2))

    def _render_key_chips(self, d, l, seed):
        for c in self.key_strip.winfo_children():
            c.destroy()
        make_key_chip(self.key_strip, "BLOK (d)", str(d) if d is not None else "—").pack(
            side="left", padx=4)
        make_key_chip(self.key_strip, "PANGKAT (l)", str(l) if l is not None else "—").pack(
            side="left", padx=4)
        make_key_chip(self.key_strip, "SEED", str(seed) if seed is not None else "—").pack(
            side="left", padx=4)

    def load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return
        try:
            state.image = load_image(path)
        except Exception:
            messagebox.showerror("Error", "File gambar tidak valid atau format tidak didukung.")
            return
        show_img(state.image, self.box_orig)
        state.after_matrix  = None
        state.after_shuffle = None
        state.after_chaos   = None
        state.enc_full      = None
        state.perm = state.Sn = None
        self._render_key_chips(None, None, None)
        self.info.config(text=f"Gambar dimuat: {state.image.shape[1]}×{state.image.shape[0]}")
        self._write_profiling("Menunggu proses enkripsi…")

    def encrypt(self):
        if state.image is None:
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return

        pwd = ask_password(self._app, "Password Enkripsi")
        if pwd is None:
            return

        seed, d, l = generate_keys_from_password(pwd, state.image.shape)
        Sn = generate_permutation_matrix(d, seed)

        state.d    = d
        state.l    = l
        state.seed = seed
        state.Sn   = Sn

        self._render_key_chips(d, l, seed)

        start = time.perf_counter()
        (after_matrix, after_shuffle, after_chaos,
         enc_full, profiling) = encrypt_image(state.image, d, l, seed)
        state.encrypt_time = time.perf_counter() - start

        state.after_matrix  = after_matrix
        state.after_shuffle = after_shuffle
        state.after_chaos   = after_chaos
        state.enc_full      = enc_full

        show_img(after_matrix,  self.box_matrix)
        show_img(after_shuffle, self.box_shuffle)
        show_img(after_chaos,   self.box_chaos)

        self.info.config(
            text=f"Enkripsi selesai dalam {state.encrypt_time:.4f} detik  "
                 f"(d={d}, l={l}, seed={seed})")
        self._show_profiling_encrypt(profiling)

    def save_orig(self):
        save_array_as_image(state.image, self._app, "citra_asli.png")

    def save_matrix(self):
        save_array_as_image(state.after_matrix, self._app, "gtsnm_matrix.png")

    def save_shuffle(self):
        save_array_as_image(state.after_shuffle, self._app, "shuffle.png")

    def save_encrypted(self):
        """Simpan citra terenkripsi BESERTA file metadata untuk dekripsi."""
        if state.after_chaos is None:
            messagebox.showwarning("Peringatan", "Belum ada gambar terenkripsi.")
            return
        save_encrypted_with_metadata(
            state.after_chaos, 
            state.image.shape,  # Original shape
            state.d, 
            state.l, 
            state.seed,
            self._app, 
            "terenkripsi.png"
        )

    def _write_profiling(self, text):
        self.prof_text.configure(state="normal")
        self.prof_text.delete("1.0", "end")
        self.prof_text.insert("end", text)
        self.prof_text.configure(state="disabled")

    def _show_profiling_encrypt(self, p):
        img_size   = p["image_size"]
        block_size = p["block_size"]
        power      = p["power"]
        mt         = p["matrix_time"]
        st         = p["shuffle_time"]
        ct         = p["chaos_time"]
        total      = mt + st + ct

        stages = {
            "Matrix Encryption":  mt,
            "Shuffle":            st,
            "Chaotic Diffusion":  ct,
        }
        bottleneck = max(stages, key=stages.get)

        bs_warn = ""
        if block_size > 64:
            bs_warn = (
                "\n⚠  Warning:\n"
                "   Large block size detected.\n"
                "   Matrix multiplication complexity increases rapidly\n"
                "   for large matrices and may become the dominant bottleneck.\n"
            )

        lines = (
            "========== ENCRYPTION PROFILING =========\n\n"
            f"Image Size        : {img_size[0]} x {img_size[1]} x {img_size[2]}\n"
            f"Block Size (d)    : {block_size}\n"
            f"Power (l)         : {power}\n"
            f"{bs_warn}"
            "-----------------------------------------\n"
            f"Matrix Encryption : {mt:.4f} s\n"
            f"Shuffle           : {st:.4f} s\n"
            f"Chaotic Diffusion : {ct:.4f} s\n\n"
            "-----------------------------------------\n"
            f"Total Encryption  : {total:.4f} s\n\n"
            f">> Bottleneck     : {bottleneck}\n"
        )
        self._write_profiling(lines)


# ============================================================
# DECRYPT PAGE (FIXED: Uses metadata for correct decryption)
# ============================================================
class DecryptPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self._app = app
        self._enc_img = None
        self._metadata = None  # Will store loaded metadata
        self._enc_path = None
        self._last_dec = None
        self.decrypt_time = 0

        self.main = make_scrollable(self, BG)
        self._build(self.main)

    def _build(self, main):
        page_header(main, "🗝", "Dekripsi Citra Tersimpan",
                    "Muat file citra terenkripsi (.png + .meta.json) lalu pulihkan dengan password yang sama")

        body = tk.Frame(main, bg=BG)
        body.pack(fill="x", padx=28)

        toolbar = tk.Frame(body, bg=BG)
        toolbar.pack(fill="x", pady=(18, 0))
        btn(toolbar, "📂  Pilih Citra Terenkripsi", self.load, C_PRIMARY,
            280, 40, bold=True).pack(side="left")

        # --- Metadata status indicator ---
        self.meta_status_frame = tk.Frame(body, bg=BG)
        self.meta_status_frame.pack(fill="x", pady=(8, 0))
        self.meta_status = tk.Label(
            self.meta_status_frame, 
            text="", 
            font=(FONT_MONO, 9), 
            bg=BG, 
            fg=SUBTEXT
        )
        self.meta_status.pack(anchor="w")

        # --- kartu preview ---
        tk.Label(body, text="HASIL DEKRIPSI", font=(FONT_MONO, 9, "bold"),
                 bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(18, 8))

        stage_card = tk.Frame(body, bg=SURFACE, highlightthickness=1,
                              highlightbackground=BORDER)
        stage_card.pack(fill="x", pady=(0, 4))
        enc_frame = tk.Frame(stage_card, bg=SURFACE)
        enc_frame.pack(pady=14, padx=8)

        self.box_enc, _ = make_img_box(
            enc_frame, "1 · Citra Terenkripsi", 0, box_size=200)
        self.box_dec, _ = make_img_box(
            enc_frame, "2 · Hasil Dekripsi", 1, box_size=200,
            save_cmd=self.save_decrypted)

        # --- panel aksi ---
        action_card = tk.Frame(body, bg=SURFACE, highlightthickness=1,
                               highlightbackground=BORDER)
        action_card.pack(fill="x", pady=(18, 0))
        gf = tk.Frame(action_card, bg=SURFACE)
        gf.pack(pady=16, padx=16)
        btn(gf, "🔓 Dekripsi", self.decrypt, C_PRIMARY, 180, 40, bold=True).grid(
            row=0, column=0, padx=6)
        btn(gf, "↓ Simpan Hasil", self.save_decrypted, C_SECONDARY, 180, 40).grid(
            row=0, column=1, padx=6)

        # --- status + profiling ---
        self.info = tk.Label(body, text="Status: menunggu citra terenkripsi…",
                             font=(FONT_MONO, 9), bg=BG, fg=SUBTEXT)
        self.info.pack(anchor="w", pady=(14, 4))

        prof_card = tk.Frame(body, bg=SURFACE, highlightthickness=1,
                             highlightbackground=BORDER)
        prof_card.pack(fill="x", pady=(4, 28))
        prof_header = tk.Frame(prof_card, bg=SURFACE_ALT)
        prof_header.pack(fill="x")
        tk.Label(prof_header, text="Profiling & Validasi Dekripsi",
                 font=(FONT_BODY, 10, "bold"), bg=SURFACE_ALT, fg=TEXT,
                 anchor="w").pack(fill="x", padx=14, pady=8)

        self.prof_text = ScrolledText(
            prof_card, height=16, font=(FONT_MONO, 9),
            bg=BG_SOFT, fg=ACCENT, insertbackground=ACCENT,
            relief="flat", state="disabled", wrap="word",
            bd=0, padx=14, pady=10)
        self.prof_text.pack(fill="x", padx=2, pady=(0, 2))

    def load(self):
        """Load encrypted image AND its metadata file."""
        img, metadata, path = load_encrypted_with_metadata(self._app)
        
        if img is None:
            return
        
        self._enc_img = img
        self._metadata = metadata
        self._enc_path = path
        self._last_dec = None
        
        show_img(self._enc_img, self.box_enc)
        h, w = self._enc_img.shape[:2]
        
        # Update metadata status
        if metadata is not None:
            orig_h, orig_w, orig_c = metadata['original_shape']
            self.meta_status.config(
                text=f"✅ Metadata ditemukan: original_size={orig_w}×{orig_h}, d={metadata['block_size']}, l={metadata['power']}, seed={metadata['seed']}",
                fg=ACCENT
            )
        else:
            self.meta_status.config(
                text=f"⚠ Metadata TIDAK ditemukan! Dekripsi mungkin gagal jika gambar asli bukan kelipatan blok.",
                fg=DANGER
            )
        
        self.info.config(text=f"Status: citra terenkripsi dimuat ({w}×{h}).\n"
                              "Tekan Dekripsi untuk memulai proses.")
        self._write_profiling("Menunggu proses dekripsi...")

    def decrypt(self):
        """Decrypt using metadata (if available) or fallback estimation."""
        if self._enc_img is None:
            messagebox.showwarning("Peringatan", "Pilih citra terenkripsi terlebih dahulu!")
            return
            
        pwd = ask_password(self._app, "Password Dekripsi")
        if pwd is None:
            return

        enc_h, enc_w, enc_c = self._enc_img.shape

        # ========== DETERMINE DECRYPTION PARAMETERS ==========
        if self._metadata is not None:
            # Use metadata from file
            original_shape = self._metadata['original_shape']
            d = self._metadata['block_size']
            l = self._metadata['power']
            seed = self._metadata['seed']
            
            # Verify password by regenerating seed
            # (optional but good for early error detection)
            _, d_check, l_check = generate_keys_from_password(pwd, original_shape)
            if d != d_check or l != l_check:
                # Try with encrypted image shape
                _, d_check2, l_check2 = generate_keys_from_password(pwd, (enc_h, enc_w, enc_c))
                if d != d_check2 or l != l_check2:
                    messagebox.showerror("Error", 
                        "Password tidak cocok dengan metadata!\n"
                        "Pastikan password yang dimasukkan sama dengan saat enkripsi.")
                    return
            
            use_metadata = True
        else:
            # Fallback: try to decrypt using encrypted image shape
            # This ONLY works if original image was already block-aligned
            seed, d, l = generate_keys_from_password(pwd, (enc_h, enc_w, enc_c))
            original_shape = (enc_h, enc_w)  # Assume no padding needed
            use_metadata = False
            
            # Check if encrypted image is block-aligned
            if enc_h % d != 0 or enc_w % d != 0:
                messagebox.showerror("Error",
                    f"Ukuran citra terenkripsi ({enc_w}×{enc_h}) bukan kelipatan blok (d={d}).\n\n"
                    "File metadata (.meta.json) diperlukan untuk dekripsi.\n"
                    "Pastikan file .meta.json berada di folder yang sama dengan citra terenkripsi.")
                return

        # ========== PERFORM DECRYPTION ==========
        try:
            start = time.perf_counter()
            dec, profiling = decrypt_image(
                self._enc_img, d, l, seed=seed, original_shape=original_shape)
            self.decrypt_time = time.perf_counter() - start
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            messagebox.showerror("Error",
                f"Terjadi kesalahan saat mendekripsi:\n\n{str(e)}\n\n"
                f"Detail teknis:\n{error_detail}")
            return

        self._last_dec = dec
        show_img(dec, self.box_dec)

        # ========== CALCULATE PROFILING ==========
        ict = profiling["inverse_chaos_time"]
        ust = profiling["unshuffle_time"]
        mdt = profiling["matrix_decrypt_time"]
        total = ict + ust + mdt

        stages = {
            "Inverse Chaotic Diffusion": ict,
            "Unshuffle": ust,
            "Matrix Decryption": mdt,
        }
        bottleneck = max(stages, key=stages.get)

        orig_h, orig_w = original_shape
        dec_h, dec_w = dec.shape[:2]

        # Verification info
        verification = ""
        if use_metadata:
            verification = (
                "\n=========================================\n"
                "VERIFIKASI METADATA\n"
                "=========================================\n"
                f"Metadata Source    : File .meta.json\n"
                f"Original Size     : {orig_w} x {orig_h}\n"
                f"Decrypted Size    : {dec_w} x {dec_h}\n"
                f"Size Match        : {'✅ Ya' if (dec_h, dec_w) == (orig_h, orig_w) else '❌ Tidak'}\n"
            )
        else:
            verification = (
                "\n=========================================\n"
                "VERIFIKASI (TANPA METADATA)\n"
                "=========================================\n"
                f"Metadata Source    : Estimasi dari ukuran citra\n"
                f"⚠ Warning         : Dekripsi mungkin tidak akurat\n"
                f"                   jika gambar asli memiliki padding!\n"
            )

        prof_text = (
            "========== DECRYPTION PROFILING =========\n\n"
            f"Encrypted Image Size : {enc_w} x {enc_h} x {enc_c}\n"
            f"Decrypted Image Size : {dec_w} x {dec_h} x {dec.shape[2] if len(dec.shape) > 2 else 1}\n"
            f"Block Size (d)      : {d}\n"
            f"Power (l)           : {l}\n"
            f"Seed                : {seed}\n\n"
            "-----------------------------------------\n"
            f"Inverse Chaos       : {ict:.4f} s\n"
            f"Unshuffle           : {ust:.4f} s\n"
            f"Matrix Decryption   : {mdt:.4f} s\n\n"
            "-----------------------------------------\n"
            f"Total Decryption    : {total:.4f} s\n"
            f">> Bottleneck       : {bottleneck}\n"
            f"{verification}"
        )

        self._write_profiling(prof_text)

        self.info.config(
            text=f"Dekripsi selesai dalam {self.decrypt_time:.4f} detik  "
                 f"(d={d}, l={l}, seed={seed})",
            fg=ACCENT)

    def _write_profiling(self, text):
        self.prof_text.configure(state="normal")
        self.prof_text.delete("1.0", "end")
        self.prof_text.insert("end", text)
        self.prof_text.configure(state="disabled")

    def save_decrypted(self):
        dec = getattr(self, "_last_dec", None)
        if dec is None:
            messagebox.showwarning("Peringatan", "Belum ada hasil dekripsi.")
            return
        save_array_as_image(dec, self._app, "hasil_dekripsi.png")


# ============================================================
# EVAL GTSNM PAGE
# ============================================================
class EvalGtsmnPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self._app = app
        self._img = None
        self.main = make_scrollable(self, BG)
        self._build(self.main)

    def _build(self, main):
        page_header(main, "📐", "Evaluasi GTSnm Matrix",
                    "Analisis statistik enkripsi menggunakan GTSnm Matrix (tanpa Shuffle & Henon Map)")
        body = tk.Frame(main, bg=BG)
        body.pack(fill="x", padx=28)

        toolbar = tk.Frame(body, bg=BG)
        toolbar.pack(fill="x", pady=(18, 0))
        btn(toolbar, "📂  Pilih Gambar", self.load, C_PRIMARY, 190, 40, bold=True).pack(side="left")
        btn(toolbar, "🧪  Jalankan Evaluasi", self.run_eval, C_PRIMARY, 200, 40, bold=True).pack(side="left", padx=(10, 0))

        tk.Label(body, text="PREVIEW", font=(FONT_MONO, 9, "bold"),
                 bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(22, 8))
        stage_card = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        stage_card.pack(fill="x", pady=(0, 4))
        prev_frame = tk.Frame(stage_card, bg=SURFACE)
        prev_frame.pack(pady=14, padx=8)

        self.box_orig, _ = make_img_box(prev_frame, "Citra Asli", 0, box_size=180)
        self.box_res, _ = make_img_box(prev_frame, "Hasil GTSnm", 1, box_size=180)

        self.results_container = tk.Frame(body, bg=BG)
        self.results_container.pack(fill="x", pady=(0, 28))
        tk.Label(self.results_container, text="Jalankan evaluasi untuk melihat hasil.",
                 font=(FONT_BODY, 9), bg=BG, fg=SUBTEXT).pack(pady=40)

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path: return
        try:
            self._img = load_image(path)
            show_img(self._img, self.box_orig)
        except Exception:
            messagebox.showerror("Error", "File gambar tidak valid.")

    def run_eval(self):
        if self._img is None:
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return
        pwd = ask_password(self._app, "Password untuk Evaluasi")
        if pwd is None: return

        seed, d, l = generate_keys_from_password(pwd, self._img.shape)
        padded = _pad_to_block(self._img, d)
        
        res = encrypt_matrix(padded, d, l, seed)
        
        h, w, _ = self._img.shape
        res_cropped = res[:h, :w]

        show_img(res_cropped, self.box_res)
        render_evaluation_results(self.results_container, self._img, res_cropped, "GTSnm")


# ============================================================
# EVAL CHAOTIC PAGE
# ============================================================
class EvalChaosPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self._app = app
        self._img = None
        self.main = make_scrollable(self, BG)
        self._build(self.main)

    def _build(self, main):
        page_header(main, "〰", "Evaluasi Chaotic Diffusion",
                    "Analisis statistik enkripsi menggunakan Henon Map (tanpa GTSnm & Shuffle)")
        body = tk.Frame(main, bg=BG)
        body.pack(fill="x", padx=28)

        toolbar = tk.Frame(body, bg=BG)
        toolbar.pack(fill="x", pady=(18, 0))
        btn(toolbar, "📂  Pilih Gambar", self.load, C_PRIMARY, 190, 40, bold=True).pack(side="left")
        btn(toolbar, "🧪  Jalankan Evaluasi", self.run_eval, C_PRIMARY, 200, 40, bold=True).pack(side="left", padx=(10, 0))

        tk.Label(body, text="PREVIEW", font=(FONT_MONO, 9, "bold"),
                 bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(22, 8))
        stage_card = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        stage_card.pack(fill="x", pady=(0, 4))
        prev_frame = tk.Frame(stage_card, bg=SURFACE)
        prev_frame.pack(pady=14, padx=8)

        self.box_orig, _ = make_img_box(prev_frame, "Citra Asli", 0, box_size=180)
        self.box_res, _ = make_img_box(prev_frame, "Hasil Henon Map", 1, box_size=180)

        self.results_container = tk.Frame(body, bg=BG)
        self.results_container.pack(fill="x", pady=(0, 28))
        tk.Label(self.results_container, text="Jalankan evaluasi untuk melihat hasil.",
                 font=(FONT_BODY, 9), bg=BG, fg=SUBTEXT).pack(pady=40)

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path: return
        try:
            self._img = load_image(path)
            show_img(self._img, self.box_orig)
        except Exception:
            messagebox.showerror("Error", "File gambar tidak valid.")

    def run_eval(self):
        if self._img is None:
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return
        pwd = ask_password(self._app, "Password untuk Evaluasi")
        if pwd is None: return

        seed, d, l = generate_keys_from_password(pwd, self._img.shape)
        padded = _pad_to_block(self._img, d)
        
        res = chaotic_diffusion(padded, seed)
        
        h, w, _ = self._img.shape
        res_cropped = res[:h, :w]

        show_img(res_cropped, self.box_res)
        render_evaluation_results(self.results_container, self._img, res_cropped, "Henon Map")


# ============================================================
# COMPARISON PAGES (GTSnm & Chaotic)
# ============================================================
class GtsmnComparisonPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self._app = app
        self._img = None
        self._no_gtsmn_result = None
        self._with_gtsmn_result = None

        self.main = make_scrollable(self, BG)
        self._build(self.main)

    def _build(self, main):
        page_header(main, "⚖", "Perbandingan Dengan dan Tanpa GTSnm",
                    "Analisis dampak transformasi GTSnm Matrix terhadap keamanan dan performa")

        body = tk.Frame(main, bg=BG)
        body.pack(fill="x", padx=28)

        toolbar = tk.Frame(body, bg=BG)
        toolbar.pack(fill="x", pady=(18, 0))
        btn(toolbar, "📂  Pilih Gambar", self.load, C_PRIMARY, 190, 40, bold=True).pack(side="left")
        btn(toolbar, "⚖  Jalankan Perbandingan", self.compare, C_PRIMARY, 230, 40, bold=True).pack(side="left", padx=(10, 0))

        tk.Label(body, text="PREVIEW HASIL", font=(FONT_MONO, 9, "bold"),
                 bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(22, 8))

        stage_card = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        stage_card.pack(fill="x", pady=(0, 4))
        prev_frame = tk.Frame(stage_card, bg=SURFACE)
        prev_frame.pack(pady=14, padx=8)

        self.box_orig, _ = make_img_box(prev_frame, "Citra Asli", 0, box_size=180)
        self.box_no, self.btn_save_no = make_img_box(prev_frame, "Tanpa GTSnm\n(Shuffle + Henon)", 1, box_size=180, save_cmd=lambda: self._save("no"))
        self.box_with, self.btn_save_with = make_img_box(prev_frame, "Dengan GTSnm\n(Full Pipeline)", 2, box_size=180, save_cmd=lambda: self._save("with"))

        tk.Label(body, text="PROFILING", font=(FONT_MONO, 9, "bold"), bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(18, 8))
        prof_card = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        prof_card.pack(fill="x", pady=(0, 4))
        self.prof_text = ScrolledText(prof_card, height=12, font=(FONT_MONO, 9), bg=BG_SOFT, fg=ACCENT, insertbackground=ACCENT, relief="flat", state="disabled", wrap="word", bd=0, padx=14, pady=10)
        self.prof_text.pack(fill="x", padx=2, pady=2)

        tk.Label(body, text="TABEL PERBANDINGAN METRIK", font=(FONT_MONO, 9, "bold"), bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(18, 8))
        self.metrics_container = tk.Frame(body, bg=BG)
        self.metrics_container.pack(fill="x", pady=(0, 28))
        tk.Label(self.metrics_container, text="Jalankan perbandingan untuk melihat hasil metrik.", font=(FONT_BODY, 9), bg=BG, fg=SUBTEXT).pack()

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path: return
        try:
            self._img = load_image(path)
        except Exception:
            messagebox.showerror("Error", "File gambar tidak valid.")
            return
        show_img(self._img, self.box_orig)

    def compare(self):
        if self._img is None:
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return

        pwd = ask_password(self._app, "Password untuk Perbandingan")
        if pwd is None: return

        seed, d, l = generate_keys_from_password(pwd, self._img.shape)
        
        from core.pipeline_variants import encrypt_no_gtsmn, encrypt_full
        
        # Tanpa GTSnm
        padded = _pad_to_block(self._img, d)
        no_gtsmn = encrypt_no_gtsmn(padded, seed)
        h, w, _ = self._img.shape
        self._no_gtsmn_result = no_gtsmn[:h, :w]
        
        # Dengan GTSnm (full pipeline)
        _, _, _, with_gtsmn, _ = encrypt_full(self._img, d, l, seed)
        self._with_gtsmn_result = with_gtsmn[:h, :w]
        
        show_img(self._no_gtsmn_result, self.box_no)
        show_img(self._with_gtsmn_result, self.box_with)
        
        # Show metrics comparison
        for w in self.metrics_container.winfo_children():
            w.destroy()
        
        from core.security_report import table_statistical_parameters, table_correlation_comparison
        
        # MAE, NPCR, UACI comparison
        df_stats = table_statistical_parameters(self._img, self._with_gtsmn_result)
        render_table_section(self.metrics_container, "Parameter Statistik (Dengan GTSnm)", df_stats.reset_index())
        
        # Correlation comparison
        df_corr = table_correlation_comparison(self._img, self._no_gtsmn_result, self._with_gtsmn_result)
        render_table_section(self.metrics_container, "Perbandingan Korelasi", df_corr)

    def _save(self, which):
        if which == "no":
            save_array_as_image(self._no_gtsmn_result, self._app, "tanpa_gtsnm.png")
        else:
            save_array_as_image(self._with_gtsmn_result, self._app, "dengan_gtsnm.png")


# ============================================================
# CHAOS COMPARISON PAGE
# ============================================================
class ChaosComparisonPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self._app = app
        self._img = None
        self._no_chaos_result = None
        self._with_chaos_result = None

        self.main = make_scrollable(self, BG)
        self._build(self.main)

    def _build(self, main):
        page_header(main, "🌀", "Perbandingan Dengan dan Tanpa Chaotic Diffusion",
                    "Analisis dampak difusi Henon Map terhadap keamanan dan performa")

        body = tk.Frame(main, bg=BG)
        body.pack(fill="x", padx=28)

        toolbar = tk.Frame(body, bg=BG)
        toolbar.pack(fill="x", pady=(18, 0))
        btn(toolbar, "📂  Pilih Gambar", self.load, C_PRIMARY, 190, 40, bold=True).pack(side="left")
        btn(toolbar, "🌀  Jalankan Perbandingan", self.compare, C_PRIMARY, 240, 40, bold=True).pack(side="left", padx=(10, 0))

        tk.Label(body, text="PREVIEW HASIL", font=(FONT_MONO, 9, "bold"),
                 bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(22, 8))

        stage_card = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        stage_card.pack(fill="x", pady=(0, 4))
        prev_frame = tk.Frame(stage_card, bg=SURFACE)
        prev_frame.pack(pady=14, padx=8)

        self.box_orig, _ = make_img_box(prev_frame, "Citra Asli", 0, box_size=180)
        self.box_no, _ = make_img_box(prev_frame, "Tanpa Chaos\n(GTSnm + Shuffle)", 1, box_size=180)
        self.box_with, _ = make_img_box(prev_frame, "Dengan Chaos\n(Full Pipeline)", 2, box_size=180)

        tk.Label(body, text="TABEL PERBANDINGAN METRIK", font=(FONT_MONO, 9, "bold"), bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(18, 8))
        self.metrics_container = tk.Frame(body, bg=BG)
        self.metrics_container.pack(fill="x", pady=(0, 28))
        tk.Label(self.metrics_container, text="Jalankan perbandingan untuk melihat hasil metrik.", font=(FONT_BODY, 9), bg=BG, fg=SUBTEXT).pack()

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path: return
        try:
            self._img = load_image(path)
        except Exception:
            messagebox.showerror("Error", "File gambar tidak valid.")
            return
        show_img(self._img, self.box_orig)

    def compare(self):
        if self._img is None:
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return

        pwd = ask_password(self._app, "Password untuk Perbandingan")
        if pwd is None: return

        seed, d, l = generate_keys_from_password(pwd, self._img.shape)
        
        from core.pipeline_variants import encrypt_gtsmn_no_chaos, encrypt_full
        from core.matrix import shuffle_image
        
        h, w, _ = self._img.shape
        padded = _pad_to_block(self._img, d)
        
        # Tanpa Chaos (GTSnm + Shuffle only)
        no_chaos = encrypt_gtsmn_no_chaos(padded, d, l, seed)
        no_chaos = shuffle_image(no_chaos, seed)
        self._no_chaos_result = no_chaos[:h, :w]
        
        # Dengan Chaos (full pipeline)
        _, _, _, with_chaos, _ = encrypt_full(self._img, d, l, seed)
        self._with_chaos_result = with_chaos[:h, :w]
        
        show_img(self._no_chaos_result, self.box_no)
        show_img(self._with_chaos_result, self.box_with)
        
        # Show metrics comparison
        for w in self.metrics_container.winfo_children():
            w.destroy()
        
        from core.security_report import table_statistical_parameters, table_correlation_comparison
        
        df_stats_no = table_statistical_parameters(self._img, self._no_chaos_result)
        render_table_section(self.metrics_container, "Parameter Statistik (Tanpa Chaos)", df_stats_no.reset_index())
        
        df_stats_with = table_statistical_parameters(self._img, self._with_chaos_result)
        render_table_section(self.metrics_container, "Parameter Statistik (Dengan Chaos)", df_stats_with.reset_index())
        
        df_corr = table_correlation_comparison(self._img, self._no_chaos_result, self._with_chaos_result)
        render_table_section(self.metrics_container, "Perbandingan Korelasi", df_corr)


# ============================================================
# HISTOGRAM PAGE
# ============================================================
class HistogramPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self._app = app
        self.main = make_scrollable(self, BG)
        self._build(self.main)

    def _build(self, main):
        page_header(main, "📊", "Histogram RGB",
                    "Perbandingan distribusi warna di tiap tahap enkripsi")
        
        self.container = tk.Frame(main, bg=BG)
        self.container.pack(fill="x", padx=28, pady=(0, 28))

    def refresh(self):
        for w in self.container.winfo_children():
            w.destroy()
            
        stages = {}
        if state.image is not None:
            stages["Original"] = state.image
        if state.after_matrix is not None:
            h, w, _ = state.image.shape
            stages["GTSnm Matrix"] = state.after_matrix[:h, :w]
        if state.after_shuffle is not None:
            h, w, _ = state.image.shape
            stages["Shuffle"] = state.after_shuffle[:h, :w]
        if state.after_chaos is not None:
            h, w, _ = state.image.shape
            stages["Encrypted"] = state.after_chaos[:h, :w]
        
        if not stages:
            tk.Label(self.container, text="Belum ada data. Lakukan enkripsi terlebih dahulu.",
                     font=(FONT_BODY, 11), bg=BG, fg=SUBTEXT).pack(pady=60)
            return
        
        chart_card = tk.Frame(self.container, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        chart_card.pack(fill="x", pady=8)
        
        fig = plot_rgb_line_histogram_grid(stages, figsize=(5 * len(stages), 4))
        canvas_widget = FigureCanvasTkAgg(fig, master=chart_card)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill="x", padx=10, pady=10)


# ============================================================
# SECURITY PAGE
# ============================================================
class SecurityPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self._app = app
        self.main = make_scrollable(self, BG)
        self._build(self.main)

    def _build(self, main):
        page_header(main, "🛡", "Analisis Keamanan",
                    "MAE, NPCR, UACI, Korelasi Piksel, Entropy, dan Key Space")
        
        body = tk.Frame(main, bg=BG)
        body.pack(fill="x", padx=28)
        
        toolbar = tk.Frame(body, bg=BG)
        toolbar.pack(fill="x", pady=(18, 0))
        btn(toolbar, "🛡  Jalankan Analisis", self.analyze, C_PRIMARY, 200, 40, bold=True).pack(side="left")
        btn(toolbar, "📊  Export ke Excel", self.export_excel, C_SECONDARY, 200, 40).pack(side="left", padx=(10, 0))
        
        self.container = tk.Frame(body, bg=BG)
        self.container.pack(fill="x", pady=(18, 28))
        
        tk.Label(self.container, text="Lakukan enkripsi terlebih dahulu, lalu tekan 'Jalankan Analisis'.",
                 font=(FONT_BODY, 11), bg=BG, fg=SUBTEXT).pack(pady=60)

    def analyze(self):
        if state.image is None or state.after_chaos is None:
            messagebox.showwarning("Peringatan", "Lakukan enkripsi terlebih dahulu!")
            return
        
        for w in self.container.winfo_children():
            w.destroy()
        
        h, w, _ = state.image.shape
        enc_cropped = state.after_chaos[:h, :w]
        
        # MAE, NPCR, UACI
        df_stats = table_statistical_parameters(state.image, enc_cropped)
        render_table_section(self.container, "MAE, NPCR, UACI", df_stats.reset_index())
        
        # Entropy
        df_entropy = table_entropy(state.image, enc_cropped)
        render_table_section(self.container, "Entropy", df_entropy.reset_index())
        
        # Correlation per stage
        stages = {
            "Original": state.image,
            "GTSnm Matrix": state.after_matrix[:h, :w],
            "Shuffle": state.after_shuffle[:h, :w],
            "Encrypted": enc_cropped
        }
        corr_tables = tables_correlation_all_stages(stages)
        for name, df in corr_tables.items():
            render_table_section(self.container, f"Korelasi - {name}", df)
        
        # Pixel Average Distance
        df_dist = table_pixel_average_distance(state.image, enc_cropped)
        render_table_section(self.container, "Rata-rata Jarak Piksel Tetangga", df_dist.reset_index())
        
        # Key Space
        if state.d is not None:
            df_kspace, total, bits, summary = analyze_key_space(state.d)
            render_table_section(self.container, "Analisis Key Space", df_kspace)
            render_text_section(self.container, "Ringkasan Key Space", summary)

    def export_excel(self):
        if state.image is None or state.after_chaos is None:
            messagebox.showwarning("Peringatan", "Lakukan enkripsi terlebih dahulu!")
            return
        
        path = filedialog.asksaveasfilename(
            initialfile="security_analysis.xlsx",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")])
        
        if not path:
            return
        
        h, w, _ = state.image.shape
        enc_cropped = state.after_chaos[:h, :w]
        
        tables = {
            "MAE_NPCR_UACI": table_statistical_parameters(state.image, enc_cropped).reset_index(),
            "Entropy": table_entropy(state.image, enc_cropped).reset_index(),
            "Pixel_Distance": table_pixel_average_distance(state.image, enc_cropped).reset_index(),
        }
        
        if state.d is not None:
            df_kspace, _, _, _ = analyze_key_space(state.d)
            tables["Key_Space"] = df_kspace
        
        try:
            export_all_to_excel(tables, path)
            messagebox.showinfo("Sukses", f"Data diekspor ke:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengekspor: {e}")


# ============================================================
# ABOUT PAGE
# ============================================================
class AboutPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.main = make_scrollable(self, BG)
        self._build(self.main)

    def _build(self, main):
        page_header(main, "ℹ", "Tentang Aplikasi",
                    "Informasi mengenai algoritma dan implementasi")
        
        body = tk.Frame(main, bg=BG)
        body.pack(fill="x", padx=28, pady=(0, 28))
        
        render_text_section(body, "CipherFrame — Image Encryption Studio", 
            """CipherFrame adalah aplikasi desktop untuk enkripsi citra digital
menggunakan kombinasi tiga teknik:

1. GTSnm MATRIX (Generalized Pascal-Triangular Symmetric Matrix)
   - Matriks Pascal yang dimodifikasi dengan parameter pangkat l
   - Setiap blok piksel dikalikan dengan matriks enkripsi
   - Memberikan diffusi pada level blok

2. SHUFFLE (Permutasi Piksel)
   - Mengacak posisi piksel menggunakan seed dari password
   - Menggunakan permutasi acak yang deterministik
   - Memberikan konfusi pada level piksel

3. HENON MAP (Chaotic Diffusion)
   - Peta chaos Henon untuk menghasilkan sequence kunci
   - Difusi berantai (chained XOR) untuk spread perubahan
   - Memberikan sensitivitas tinggi terhadap perubahan input

Ketiga teknik ini bekerja secara berurutan untuk memenuhi
prinsip_confusion dan diffusion dalam kriptografi modern.
""")
        
        render_text_section(body, "Parameter Kunci",
            """Dari satu password, sistem menghasilkan:
• SEED    : Untuk RNG (shuffle) dan chaos (Henon map)
• d       : Ukuran blok matriks GTSnm
• l       : Pangkat matriks Pascal

Semua parameter diturunkan secara deterministik dari password,
sehingga tidak perlu menyimpan kunci terpisah.
""")
        
        render_text_section(body, "Format File Metadata",
            """Saat menyimpan citra terenkripsi, aplikasi juga membuat file
.metadata.json yang berisi:
• original_shape : Ukuran citra asli (sebelum padding)
• block_size     : Parameter d
• power          : Parameter l  
• seed           : Seed untuk RNG dan chaos

File ini WAJIB ada untuk proses dekripsi yang benar.
Pastikan file .meta.json selalu berada di folder yang sama
dengan citra terenkripsi (.png).
""")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()