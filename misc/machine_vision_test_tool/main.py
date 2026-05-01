from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2

from focus_detection import (
    generate_focus_map,
    generate_focus_map_laplacian,
    normalize_score_map,
    compute_focus_scores,
    apply_focus_overlay,
    add_colorbar,
    FocusScores,
    FocusMethod,
    FOCUS_METHOD_TENENGRAD,
    FOCUS_METHOD_LAPLACIAN,
)

IGNORED_FOLDERS = {"output", "tiles"}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
THUMB_SIZE = (96, 72)
MAIN_MAX_SIZE = (900, 600)
FOLDER_SAMPLE_COUNT = 5

# Default background threshold values
DEFAULT_COLOR_TOLERANCE  = 35
DEFAULT_STD_MULTIPLIER   = 2.5
DEFAULT_IMAGE_THRESHOLD  = 0.72
DEFAULT_FOLDER_THRESHOLD = 0.6

# Default focus detection parameters
DEFAULT_FOCUS_KERNEL       = 3
DEFAULT_FOCUS_RADIUS       = 8.0
DEFAULT_FOCUS_THRESHOLD    = 0.0
DEFAULT_FOCUS_ALPHA        = 0.6
DEFAULT_FOCUS_PEAK_PCTILE  = 99.0
DEFAULT_FOCUS_SCORE_CUTOFF = 0.15   # peak score above which an image is "in focus"
DEFAULT_FOCUS_CEILING      = 500.0  # fixed normalisation ceiling in raw gradient units
DEFAULT_FOCUS_METHOD       = FOCUS_METHOD_TENENGRAD
DEFAULT_FOCUS_LAP_WINDOW   = 15     # local window size (px) for Laplacian variance


# ---------------------------------------------------------------------------
# Background analysis helpers
# ---------------------------------------------------------------------------

def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def get_folders(root: Path) -> list[Path]:
    result = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORED_FOLDERS]
        p = Path(dirpath)
        if any(f.is_file() and is_image(f) for f in p.iterdir()):
            result.append(p)
    return sorted(result)


def build_bg_profile(bg_images: list[Path],
                     color_tolerance: float,
                     std_multiplier: float) -> tuple[np.ndarray, np.ndarray] | None:
    from PIL import Image as PILImage
    all_pixels: list[np.ndarray] = []
    for sp in bg_images:
        try:
            im = PILImage.open(sp).convert("RGB").resize((64, 64), PILImage.LANCZOS)
            all_pixels.append(np.array(im).reshape(-1, 3))
        except Exception:
            continue
    if not all_pixels:
        return None
    bg_pixels = np.concatenate(all_pixels, axis=0).astype(np.float32)
    bg_mean = bg_pixels.mean(axis=0)
    bg_std  = bg_pixels.std(axis=0)
    tolerance = np.maximum(color_tolerance, bg_std * std_multiplier)
    return bg_mean, tolerance


def image_is_background(img_path: Path,
                        bg_mean: np.ndarray,
                        tolerance: np.ndarray,
                        image_threshold: float) -> bool:
    try:
        im = Image.open(img_path).convert("RGB").resize((128, 128), Image.LANCZOS)
        cand_pixels = np.array(im).reshape(-1, 3).astype(np.float32)
    except Exception:
        return False
    within = np.all(np.abs(cand_pixels - bg_mean) <= tolerance, axis=1)
    return bool(within.mean() >= image_threshold)


def analyze_folder_is_background(folder: Path,
                                  bg_mean: np.ndarray,
                                  tolerance: np.ndarray,
                                  image_threshold: float,
                                  folder_threshold: float) -> tuple[bool, int, int]:
    images = sorted(p for p in folder.iterdir() if p.is_file() and is_image(p))
    if not images:
        return False, 0, 0
    if len(images) <= FOLDER_SAMPLE_COUNT:
        samples = images
    else:
        step = len(images) / FOLDER_SAMPLE_COUNT
        samples = [images[int(i * step)] for i in range(FOLDER_SAMPLE_COUNT)]
    matches = sum(1 for p in samples
                  if image_is_background(p, bg_mean, tolerance, image_threshold))
    return (matches / len(samples)) >= folder_threshold, matches, len(samples)


# ---------------------------------------------------------------------------
# Focus analysis helpers
# ---------------------------------------------------------------------------

def load_bgr(path: Path) -> np.ndarray | None:
    """Load an image as a BGR numpy array for OpenCV."""
    img = cv2.imread(str(path))
    return img  # None if failed


def compute_focus_for_image(
    path: Path,
    kernel: int,
    radius: float,
    grad_threshold: float,
    peak_percentile: float,
    ceiling: float,
    method: FocusMethod = FOCUS_METHOD_TENENGRAD,
    laplacian_window: int = DEFAULT_FOCUS_LAP_WINDOW,
) -> tuple[np.ndarray | None, FocusScores | None]:
    """
    Compute a normalised focus map and scores for a single image.

    The raw map is normalised against `ceiling` — a fixed user-supplied value in
    raw score units.  Using the same ceiling across all images makes scores
    directly comparable without a two-pass scan.

    method selects the focus measure: FOCUS_METHOD_TENENGRAD (Sobel gradient
    magnitude) or FOCUS_METHOD_LAPLACIAN (local Laplacian variance).
    laplacian_window is the local window size in pixels and is only used when
    method is FOCUS_METHOD_LAPLACIAN.
    """
    bgr = load_bgr(path)
    if bgr is None:
        return None, None
    try:
        if method == FOCUS_METHOD_LAPLACIAN:
            raw = generate_focus_map_laplacian(
                bgr,
                window_size=laplacian_window,
                radius=radius,
                threshold=grad_threshold,
                verbose=False,
                normalize=False,
            )
        else:
            raw = generate_focus_map(
                bgr,
                kernel_size=kernel,
                radius=radius,
                threshold=grad_threshold,
                verbose=False,
                normalize=False,
            )
        norm_map = normalize_score_map(raw, ceiling=ceiling)
        scores = compute_focus_scores(norm_map, peak_percentile=peak_percentile)
        return norm_map, scores
    except Exception:
        return None, None


def render_focus_overlay(path: Path,
                          score_map: np.ndarray,
                          alpha: float,
                          max_size: tuple[int, int]) -> ImageTk.PhotoImage | None:
    """
    Blend the focus heatmap onto the original image, add colorbar, resize for display,
    and return a PhotoImage.
    """
    bgr = load_bgr(path)
    if bgr is None:
        return None
    try:
        overlay = apply_focus_overlay(bgr, score_map, alpha=alpha)
        overlay = add_colorbar(overlay, side="right")
        # Resize to fit display
        h, w = overlay.shape[:2]
        max_w, max_h = max_size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Thumbnail widget
# ---------------------------------------------------------------------------

class Thumbnail(tk.Frame):
    def __init__(self, parent: tk.Widget, img_path: Path,
                 on_click: callable, is_bg: bool = False,
                 is_focused: bool = False, **kwargs):
        super().__init__(parent, **kwargs)
        self.img_path = img_path
        self.on_click = on_click
        self._is_bg: bool = is_bg
        self._is_focused: bool = is_focused
        self._selected: bool = False
        self._photo: ImageTk.PhotoImage | None = None
        self._build()

    def _build(self) -> None:
        self.config(bg=self._border_color(), padx=2, pady=2, cursor="hand2")
        try:
            im = Image.open(self.img_path).convert("RGB")
            im.thumbnail(THUMB_SIZE, Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(im)
            lbl = tk.Label(self, image=self._photo, bg="#1a1a2e")
        except Exception:
            lbl = tk.Label(self, text="?", width=10, height=5,
                           bg="#1a1a2e", fg="#aaa")
        lbl.pack()
        lbl.bind("<Button-1>", lambda e: self.on_click(self.img_path))
        self.bind("<Button-1>",  lambda e: self.on_click(self.img_path))

    def _border_color(self) -> str:
        if self._selected:
            return "#f39c12"
        if self._is_focused:
            return "#1e90ff"   # bright blue for in-focus images
        if self._is_bg:
            return "#e74c3c"
        return "#2c3e50"

    def _apply(self) -> None:
        self.config(bg=self._border_color())

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply()

    def set_bg_flag(self, is_bg: bool) -> None:
        self._is_bg = is_bg
        self._apply()

    def set_focus_flag(self, is_focused: bool) -> None:
        self._is_focused = is_focused
        self._apply()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class ImageViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Vision Dataset Viewer")
        self.configure(bg="#0d0d1a")
        self.geometry("1280x920")
        self.minsize(900, 700)

        # --- dataset state ---
        self._root_dir: Path | None = None
        self._folders: list[Path] = []
        self._current_folder: Path | None = None
        self._current_folder_idx: int = -1
        self._images: list[Path] = []
        self._current_idx: int = 0

        # --- background analysis state ---
        self._bg_folder: Path | None = None
        self._bg_images: list[Path] = []
        self._bg_profile: tuple[np.ndarray, np.ndarray] | None = None
        self._folder_bg_flags: dict[Path, bool] = {}
        self._analysis_running: bool = False

        # --- background threshold vars ---
        self._var_color_tol   = tk.DoubleVar(value=DEFAULT_COLOR_TOLERANCE)
        self._var_std_mult    = tk.DoubleVar(value=DEFAULT_STD_MULTIPLIER)
        self._var_img_thresh  = tk.DoubleVar(value=DEFAULT_IMAGE_THRESHOLD)
        self._var_fold_thresh = tk.DoubleVar(value=DEFAULT_FOLDER_THRESHOLD)

        for var in (self._var_color_tol, self._var_std_mult,
                    self._var_img_thresh, self._var_fold_thresh):
            var.trace_add("write", self._on_bg_threshold_change)

        # --- focus detection state ---
        # score maps cached per image path; cleared when folder changes
        self._focus_score_maps: dict[Path, np.ndarray] = {}
        self._focus_scores: dict[Path, FocusScores] = {}
        self._focus_flags: dict[Path, bool] = {}      # True = in focus
        self._focus_running: bool = False
        self._focus_overlay_on: bool = True            # toggled by button

        # --- focus parameter vars ---
        self._var_focus_kernel      = tk.IntVar(value=DEFAULT_FOCUS_KERNEL)
        self._var_focus_radius      = tk.DoubleVar(value=DEFAULT_FOCUS_RADIUS)
        self._var_focus_grad_thresh = tk.DoubleVar(value=DEFAULT_FOCUS_THRESHOLD)
        self._var_focus_alpha       = tk.DoubleVar(value=DEFAULT_FOCUS_ALPHA)
        self._var_focus_peak_pctile = tk.DoubleVar(value=DEFAULT_FOCUS_PEAK_PCTILE)
        self._var_focus_cutoff      = tk.DoubleVar(value=DEFAULT_FOCUS_SCORE_CUTOFF)
        self._var_focus_ceiling     = tk.DoubleVar(value=DEFAULT_FOCUS_CEILING)
        self._var_focus_method      = tk.StringVar(value=DEFAULT_FOCUS_METHOD)
        self._var_focus_lap_window  = tk.IntVar(value=DEFAULT_FOCUS_LAP_WINDOW)

        self._thumb_widgets: list[Thumbnail] = []
        self._main_photo: ImageTk.PhotoImage | None = None
        self._preview_dirty: bool = False
        self._resize_pending: bool = False

        self._build_ui()
        self.bind("<Left>",  lambda e: self._navigate_images(-1))
        self.bind("<Right>", lambda e: self._navigate_images(1))
        self.bind("<Up>",    lambda e: self._navigate_folders(-1))
        self.bind("<Down>",  lambda e: self._navigate_folders(1))

    # ------------------------------------------------------------------
    # Background threshold helpers
    # ------------------------------------------------------------------

    def _current_bg_thresholds(self) -> tuple[float, float, float, float]:
        return (
            self._var_color_tol.get(),
            self._var_std_mult.get(),
            self._var_img_thresh.get(),
            self._var_fold_thresh.get(),
        )

    def _rebuild_profile(self) -> bool:
        if not self._bg_images:
            return False
        color_tol, std_mult, _, _ = self._current_bg_thresholds()
        result = build_bg_profile(self._bg_images, color_tol, std_mult)
        if result is None:
            return False
        self._bg_profile = result
        return True

    def _on_bg_threshold_change(self, *_args: object) -> None:
        if self._preview_dirty:
            return
        self._preview_dirty = True
        self.after(120, self._run_bg_preview)

    def _run_bg_preview(self) -> None:
        self._preview_dirty = False
        if not self._bg_images or not self._images:
            return
        if not self._rebuild_profile():
            return
        bg_mean, tolerance = self._bg_profile
        _, _, img_thresh, _ = self._current_bg_thresholds()
        path = self._images[self._current_idx]
        result = image_is_background(path, bg_mean, tolerance, img_thresh)
        self._update_bg_preview_indicator(result)

    def _update_bg_preview_indicator(self, preview_is_bg: bool) -> None:
        path = self._images[self._current_idx]
        try:
            im = Image.open(path)
            w, h = im.size
        except Exception:
            w = h = 0

        stored_flag = self._folder_bg_flags.get(self._current_folder, False) \
                      if self._current_folder else False

        if preview_is_bg:
            flag_text = "  [BACKGROUND FOLDER]" if stored_flag else "  [BACKGROUND — preview]"
        else:
            flag_text = "  [would UNFLAG — preview]" if stored_flag else ""

        focus_flag = self._focus_flags.get(path, False)
        focus_text = "  [IN FOCUS]" if focus_flag else ""

        self._img_info.config(
            text=f"{path.name}   {w}x{h}   "
                 f"{self._current_idx + 1}/{len(self._images)}{flag_text}{focus_text}",
            fg="#e74c3c" if preview_is_bg else "#888" if stored_flag else "#666",
        )

        is_bg_folder = stored_flag
        for i, tw in enumerate(self._thumb_widgets):
            if i == self._current_idx:
                tw.set_bg_flag(preview_is_bg)
                tw.set_selected(True)
            else:
                tw.set_bg_flag(is_bg_folder)
                tw.set_selected(False)

    # ------------------------------------------------------------------
    # Focus parameter helpers
    # ------------------------------------------------------------------

    def _current_focus_params(
        self,
    ) -> tuple[int, float, float, float, float, float, float, FocusMethod, int]:
        """Return (kernel, radius, grad_threshold, alpha, peak_percentile, score_cutoff, ceiling, method, lap_window)."""
        return (
            self._var_focus_kernel.get(),
            self._var_focus_radius.get(),
            self._var_focus_grad_thresh.get(),
            self._var_focus_alpha.get(),
            self._var_focus_peak_pctile.get(),
            self._var_focus_cutoff.get(),
            self._var_focus_ceiling.get(),
            self._var_focus_method.get(),
            self._var_focus_lap_window.get(),
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        topbar = tk.Frame(self, bg="#0d0d1a", pady=6)
        topbar.pack(fill="x", padx=12)

        btn_style = {
            "bg": "#1e1e3a", "fg": "#c9d1d9", "relief": "flat",
            "padx": 14, "pady": 6, "cursor": "hand2",
            "font": ("Courier New", 10, "bold"), "bd": 0,
            "activebackground": "#2d2d5a", "activeforeground": "#fff",
        }

        tk.Button(topbar, text="OPEN ROOT FOLDER",
                  command=self._open_root, **btn_style).pack(side="left", padx=(0, 8))
        tk.Button(topbar, text="SET AS BACKGROUND FOLDER",
                  command=self._set_bg_folder, **btn_style).pack(side="left", padx=(0, 8))

        self._bg_label = tk.Label(topbar, text="No background set",
                                  bg="#0d0d1a", fg="#666", font=("Courier New", 9))
        self._bg_label.pack(side="left", padx=12)

        self._status = tk.Label(topbar, text="", bg="#0d0d1a",
                                fg="#f39c12", font=("Courier New", 9))
        self._status.pack(side="right")

        self._build_bg_threshold_panel()
        self._build_focus_panel()

        # Main split: folder list | viewer
        split = tk.Frame(self, bg="#0d0d1a")
        split.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Folder list
        folder_frame = tk.Frame(split, bg="#111127", width=220)
        folder_frame.pack(side="left", fill="y", padx=(0, 8))
        folder_frame.pack_propagate(False)

        tk.Label(folder_frame, text="FOLDERS", bg="#111127", fg="#555",
                 font=("Courier New", 9, "bold"), pady=8).pack()

        folder_scroll = tk.Scrollbar(folder_frame, orient="vertical", bg="#0d0d1a")
        folder_scroll.pack(side="right", fill="y")

        self._folder_list = tk.Listbox(
            folder_frame, bg="#111127", fg="#c9d1d9",
            selectbackground="#2d2d5a", selectforeground="#fff",
            relief="flat", bd=0, font=("Courier New", 10),
            yscrollcommand=folder_scroll.set,
            activestyle="none", exportselection=False,
        )
        self._folder_list.pack(fill="both", expand=True)
        folder_scroll.config(command=self._folder_list.yview)
        self._folder_list.bind("<<ListboxSelect>>", self._on_folder_select)

        # Right pane
        right = tk.Frame(split, bg="#0d0d1a")
        right.pack(side="left", fill="both", expand=True)

        # Pack fixed-height widgets from the bottom first so the viewer
        # frame claims only the remaining space and cannot push them out.
        thumb_outer = tk.Frame(right, bg="#0d0d1a")
        thumb_outer.pack(side="bottom", fill="x")

        self._img_info = tk.Label(right, text="", bg="#0d0d1a",
                                  fg="#666", font=("Courier New", 9))
        self._img_info.pack(side="bottom")

        # Viewer frame expands to fill what is left.  <Configure> is bound
        # here (not on the label) so we size to available area, not image size.
        viewer_frame = tk.Frame(right, bg="#0d0d1a")
        viewer_frame.pack(side="top", expand=True, fill="both", pady=(0, 4))
        viewer_frame.bind("<Configure>", self._on_viewer_resize)
        self._viewer_frame = viewer_frame

        self._img_label = tk.Label(viewer_frame, bg="#0d0d1a")
        self._img_label.place(relx=0.5, rely=0.5, anchor="center")

        self._thumb_canvas = tk.Canvas(thumb_outer, bg="#0d0d1a", height=108,
                                       highlightthickness=0)
        hsb = tk.Scrollbar(thumb_outer, orient="horizontal",
                            command=self._thumb_canvas.xview)
        self._thumb_canvas.configure(xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        self._thumb_canvas.pack(fill="x")

        self._thumb_frame = tk.Frame(self._thumb_canvas, bg="#0d0d1a")
        self._thumb_canvas.create_window((0, 0), window=self._thumb_frame, anchor="nw")
        self._thumb_frame.bind(
            "<Configure>",
            lambda e: self._thumb_canvas.configure(
                scrollregion=self._thumb_canvas.bbox("all")),
        )
        self._thumb_canvas.bind(
            "<MouseWheel>",
            lambda e: self._thumb_canvas.xview_scroll(int(-e.delta / 60), "units"),
        )

    # ------------------------------------------------------------------
    # Background threshold panel
    # ------------------------------------------------------------------

    def _build_bg_threshold_panel(self) -> None:
        outer = tk.Frame(self, bg="#0d0d1a")
        outer.pack(fill="x", padx=12, pady=(0, 3))

        header = tk.Frame(outer, bg="#111127", cursor="hand2")
        header.pack(fill="x")

        self._bg_thresh_visible = tk.BooleanVar(value=False)
        self._bg_thresh_toggle_lbl = tk.Label(
            header, text="▶  BACKGROUND THRESHOLDS", bg="#111127", fg="#555",
            font=("Courier New", 9, "bold"), padx=8, pady=4, anchor="w",
        )
        self._bg_thresh_toggle_lbl.pack(side="left", fill="x", expand=True)
        header.bind("<Button-1>", self._toggle_bg_threshold_panel)
        self._bg_thresh_toggle_lbl.bind("<Button-1>", self._toggle_bg_threshold_panel)

        self._bg_thresh_panel = tk.Frame(outer, bg="#111127", pady=8)

        panel = self._bg_thresh_panel
        lbl_style   = {"bg": "#111127", "fg": "#888", "font": ("Courier New", 9),
                       "width": 20, "anchor": "w"}
        entry_style = {"bg": "#1a1a2e", "fg": "#c9d1d9", "relief": "flat",
                       "font": ("Courier New", 9), "width": 6,
                       "insertbackground": "#c9d1d9"}

        sliders: list[tuple[str, tk.Variable, float, float, int]] = [
            ("Color tolerance",   self._var_color_tol,   0.0, 120.0, 1),
            ("Std multiplier",    self._var_std_mult,    0.0,  10.0, 2),
            ("Image threshold",   self._var_img_thresh,  0.0,   1.0, 2),
            ("Folder threshold",  self._var_fold_thresh, 0.0,   1.0, 2),
        ]
        for label, var, lo, hi, digits in sliders:
            row = tk.Frame(panel, bg="#111127")
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, **lbl_style).pack(side="left")
            tk.Scale(row, from_=lo, to=hi, resolution=10**-digits,
                     orient="horizontal", variable=var,
                     bg="#111127", fg="#c9d1d9", troughcolor="#1a1a2e",
                     highlightthickness=0, relief="flat", length=300,
                     showvalue=False).pack(side="left", padx=(4, 8))
            tk.Entry(row, textvariable=var, **entry_style).pack(side="left")

        btn_row = tk.Frame(panel, bg="#111127")
        btn_row.pack(fill="x", padx=12, pady=(6, 2))
        bstyle = {"relief": "flat", "bd": 0, "padx": 14, "pady": 5,
                  "font": ("Courier New", 9, "bold"), "cursor": "hand2"}
        tk.Button(btn_row, text="RESET DEFAULTS", bg="#1e1e3a", fg="#888",
                  activebackground="#2d2d5a", activeforeground="#fff",
                  command=self._reset_bg_thresholds, **bstyle).pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="APPLY — RESCAN ALL FOLDERS",
                  bg="#1a3a1a", fg="#2ecc71",
                  activebackground="#2a5a2a", activeforeground="#fff",
                  command=self._apply_bg_thresholds, **bstyle).pack(side="left")
        tk.Label(panel, text="Editing previews current image only. Apply rescans all folders.",
                 bg="#111127", fg="#444", font=("Courier New", 8),
                 pady=4).pack(anchor="w", padx=12)

    def _toggle_bg_threshold_panel(self, _event: tk.Event | None = None) -> None:
        if self._bg_thresh_visible.get():
            self._bg_thresh_panel.pack_forget()
            self._bg_thresh_visible.set(False)
            self._bg_thresh_toggle_lbl.config(text="▶  BACKGROUND THRESHOLDS")
        else:
            self._bg_thresh_panel.pack(fill="x")
            self._bg_thresh_visible.set(True)
            self._bg_thresh_toggle_lbl.config(text="▼  BACKGROUND THRESHOLDS")

    def _reset_bg_thresholds(self) -> None:
        self._var_color_tol.set(DEFAULT_COLOR_TOLERANCE)
        self._var_std_mult.set(DEFAULT_STD_MULTIPLIER)
        self._var_img_thresh.set(DEFAULT_IMAGE_THRESHOLD)
        self._var_fold_thresh.set(DEFAULT_FOLDER_THRESHOLD)

    def _apply_bg_thresholds(self) -> None:
        if not self._bg_images:
            messagebox.showinfo("No background set", "Set a background folder first.")
            return
        if not self._rebuild_profile():
            messagebox.showerror("Profile error", "Could not build background profile.")
            return
        self._run_bg_analysis()

    # ------------------------------------------------------------------
    # Focus detection panel
    # ------------------------------------------------------------------

    def _build_focus_panel(self) -> None:
        outer = tk.Frame(self, bg="#0d0d1a")
        outer.pack(fill="x", padx=12, pady=(0, 6))

        header = tk.Frame(outer, bg="#0d1a1a", cursor="hand2")
        header.pack(fill="x")

        self._focus_panel_visible = tk.BooleanVar(value=False)
        self._focus_toggle_lbl = tk.Label(
            header, text="▶  FOCUS DETECTION", bg="#0d1a1a", fg="#2ecc71",
            font=("Courier New", 9, "bold"), padx=8, pady=4, anchor="w",
        )
        self._focus_toggle_lbl.pack(side="left", fill="x", expand=True)

        # Overlay toggle button in the header so it's always accessible
        self._overlay_btn = tk.Button(
            header, text="OVERLAY: ON",
            bg="#0d1a1a", fg="#1e90ff", relief="flat", bd=0,
            font=("Courier New", 9, "bold"), padx=8, pady=4, cursor="hand2",
            activebackground="#0d1a1a", activeforeground="#63b3ed",
            command=self._toggle_overlay,
        )
        self._overlay_btn.pack(side="right", padx=8)

        header.bind("<Button-1>", self._toggle_focus_panel)
        self._focus_toggle_lbl.bind("<Button-1>", self._toggle_focus_panel)

        self._focus_panel = tk.Frame(outer, bg="#0d1a1a", pady=8)

        panel = self._focus_panel
        lbl_style   = {"bg": "#0d1a1a", "fg": "#888", "font": ("Courier New", 9),
                       "width": 20, "anchor": "w"}
        entry_style = {"bg": "#0d2020", "fg": "#c9d1d9", "relief": "flat",
                       "font": ("Courier New", 9), "width": 6,
                       "insertbackground": "#c9d1d9"}

        # Method selector row
        method_row = tk.Frame(panel, bg="#0d1a1a")
        method_row.pack(fill="x", padx=12, pady=(4, 6))
        tk.Label(method_row, text="Focus method", **lbl_style).pack(side="left")
        radio_style = {
            "bg": "#0d1a1a", "fg": "#c9d1d9", "selectcolor": "#0d2020",
            "activebackground": "#0d1a1a", "activeforeground": "#fff",
            "font": ("Courier New", 9), "cursor": "hand2", "bd": 0,
        }
        tk.Radiobutton(
            method_row, text="Tenengrad",
            variable=self._var_focus_method, value=FOCUS_METHOD_TENENGRAD,
            command=self._on_focus_method_change, **radio_style,
        ).pack(side="left", padx=(4, 12))
        tk.Radiobutton(
            method_row, text="Laplacian variance",
            variable=self._var_focus_method, value=FOCUS_METHOD_LAPLACIAN,
            command=self._on_focus_method_change, **radio_style,
        ).pack(side="left")

        # Kernel size row — hidden when Laplacian is active; managed by _on_focus_method_change
        self._kernel_row = tk.Frame(panel, bg="#0d1a1a")
        tk.Label(self._kernel_row, text="Kernel size (odd)", **lbl_style).pack(side="left")
        tk.Scale(
            self._kernel_row, from_=1, to=7, resolution=2,
            orient="horizontal", variable=self._var_focus_kernel,
            bg="#0d1a1a", fg="#c9d1d9", troughcolor="#0d2020",
            highlightthickness=0, relief="flat", length=300,
            showvalue=False,
        ).pack(side="left", padx=(4, 8))
        tk.Entry(self._kernel_row, textvariable=self._var_focus_kernel,
                 **entry_style).pack(side="left")

        # Laplacian window row — hidden when Tenengrad is active; managed by _on_focus_method_change
        self._lap_window_row = tk.Frame(panel, bg="#0d1a1a")
        tk.Label(self._lap_window_row, text="Lap. window (px)", **lbl_style).pack(side="left")
        tk.Scale(
            self._lap_window_row, from_=3, to=63, resolution=2,
            orient="horizontal", variable=self._var_focus_lap_window,
            bg="#0d1a1a", fg="#c9d1d9", troughcolor="#0d2020",
            highlightthickness=0, relief="flat", length=300,
            showvalue=False,
        ).pack(side="left", padx=(4, 8))
        tk.Entry(self._lap_window_row, textvariable=self._var_focus_lap_window,
                 **entry_style).pack(side="left")

        # Show the correct row for the default method
        self._on_focus_method_change()

        # Remaining shared sliders
        # Kernel size row already added above; skip it from the loop.
        sliders: list[tuple[str, tk.Variable, float, float, float]] = [
            ("Blur radius (px)",    self._var_focus_radius,      0.0, 32.0, 1),
            ("Gradient threshold",  self._var_focus_grad_thresh, 0.0, 500.0, 0),
            ("Overlay alpha",       self._var_focus_alpha,       0.0,   1.0, 2),
            ("Peak percentile",     self._var_focus_peak_pctile, 80.0, 100.0, 1),
            ("Normalisation ceiling", self._var_focus_ceiling,   1.0, 5000.0, 0),
            ("In-focus cutoff",     self._var_focus_cutoff,      0.0,   1.0, 2),
        ]
        for label, var, lo, hi, digits in sliders:
            row = tk.Frame(panel, bg="#0d1a1a")
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, **lbl_style).pack(side="left")
            res = 10**-digits if digits > 0 else 1
            tk.Scale(row, from_=lo, to=hi, resolution=res,
                     orient="horizontal", variable=var,
                     bg="#0d1a1a", fg="#c9d1d9", troughcolor="#0d2020",
                     highlightthickness=0, relief="flat", length=300,
                     showvalue=False).pack(side="left", padx=(4, 8))
            tk.Entry(row, textvariable=var, **entry_style).pack(side="left")

        # Hint about kernel size
        tk.Label(panel,
                 text="Tenengrad kernel is forced to the nearest odd integer (1, 3, 5, 7).",
                 bg="#0d1a1a", fg="#444", font=("Courier New", 8)).pack(anchor="w", padx=12)

        btn_row = tk.Frame(panel, bg="#0d1a1a")
        btn_row.pack(fill="x", padx=12, pady=(8, 2))
        bstyle = {"relief": "flat", "bd": 0, "padx": 14, "pady": 5,
                  "font": ("Courier New", 9, "bold"), "cursor": "hand2"}
        tk.Button(btn_row, text="RUN FOCUS DETECTION ON FOLDER",
                  bg="#0d2a3a", fg="#1e90ff",
                  activebackground="#1a3a4a", activeforeground="#63b3ed",
                  command=self._run_focus_detection, **bstyle).pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="RESET DEFAULTS",
                  bg="#0d1a1a", fg="#555",
                  activebackground="#0d2020", activeforeground="#fff",
                  command=self._reset_focus_params, **bstyle).pack(side="left")

        self._focus_status = tk.Label(
            panel, text="", bg="#0d1a1a", fg="#1e90ff", font=("Courier New", 8), pady=2)
        self._focus_status.pack(anchor="w", padx=12)

    def _toggle_focus_panel(self, _event: tk.Event | None = None) -> None:
        if self._focus_panel_visible.get():
            self._focus_panel.pack_forget()
            self._focus_panel_visible.set(False)
            self._focus_toggle_lbl.config(text="▶  FOCUS DETECTION")
        else:
            self._focus_panel.pack(fill="x")
            self._focus_panel_visible.set(True)
            self._focus_toggle_lbl.config(text="▼  FOCUS DETECTION")

    def _toggle_overlay(self) -> None:
        self._focus_overlay_on = not self._focus_overlay_on
        label = "OVERLAY: ON" if self._focus_overlay_on else "OVERLAY: OFF"
        self._overlay_btn.config(text=label)
        self._show_current()

    def _reset_focus_params(self) -> None:
        self._var_focus_kernel.set(DEFAULT_FOCUS_KERNEL)
        self._var_focus_radius.set(DEFAULT_FOCUS_RADIUS)
        self._var_focus_grad_thresh.set(DEFAULT_FOCUS_THRESHOLD)
        self._var_focus_alpha.set(DEFAULT_FOCUS_ALPHA)
        self._var_focus_peak_pctile.set(DEFAULT_FOCUS_PEAK_PCTILE)
        self._var_focus_cutoff.set(DEFAULT_FOCUS_SCORE_CUTOFF)
        self._var_focus_ceiling.set(DEFAULT_FOCUS_CEILING)
        self._var_focus_method.set(DEFAULT_FOCUS_METHOD)
        self._var_focus_lap_window.set(DEFAULT_FOCUS_LAP_WINDOW)
        self._on_focus_method_change()

    def _on_focus_method_change(self) -> None:
        """Show the kernel row for Tenengrad or the window row for Laplacian."""
        if self._var_focus_method.get() == FOCUS_METHOD_LAPLACIAN:
            self._kernel_row.pack_forget()
            self._lap_window_row.pack(fill="x", padx=12, pady=2)
        else:
            self._lap_window_row.pack_forget()
            self._kernel_row.pack(fill="x", padx=12, pady=2)

    # ------------------------------------------------------------------
    # Folder management
    # ------------------------------------------------------------------

    def _open_root(self) -> None:
        d = filedialog.askdirectory(title="Select root image folder")
        if not d:
            return
        self._root_dir = Path(d)
        self._bg_folder = None
        self._bg_images = []
        self._bg_profile = None
        self._folder_bg_flags = {}
        self._current_folder_idx = -1
        self._focus_score_maps = {}
        self._focus_scores = {}
        self._focus_flags = {}
        self._bg_label.config(text="No background set", fg="#666")
        self._load_folders()

    def _load_folders(self) -> None:
        if not self._root_dir:
            return
        self._folders = get_folders(self._root_dir)
        self._refresh_folder_list()
        self._status.config(text=f"{len(self._folders)} folders found")

    def _refresh_folder_list(self) -> None:
        sel = self._folder_list.curselection()
        sel_idx = sel[0] if sel else None
        self._folder_list.delete(0, "end")
        for f in self._folders:
            rel = f.relative_to(self._root_dir)
            self._folder_list.insert("end", str(rel))
            if self._folder_bg_flags.get(f, False):
                self._folder_list.itemconfig(
                    self._folder_list.size() - 1, fg="#e74c3c")
        if sel_idx is not None:
            self._folder_list.selection_set(sel_idx)

    def _on_folder_select(self, _event: tk.Event | None = None) -> None:
        sel = self._folder_list.curselection()
        if not sel:
            return
        self._current_folder_idx = sel[0]
        self._load_folder(self._folders[sel[0]])

    def _load_folder(self, folder: Path) -> None:
        self._current_folder = folder
        self._images = sorted(
            p for p in folder.iterdir() if p.is_file() and is_image(p)
        )
        self._current_idx = 0
        # Clear focus cache when changing folders — params may have changed
        self._focus_score_maps = {}
        self._focus_scores = {}
        self._focus_flags = {}
        self._focus_status.config(text="")
        self._rebuild_thumbs()
        self._show_current()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate_images(self, delta: int) -> None:
        if not self._images:
            return
        self._current_idx = (self._current_idx + delta) % len(self._images)
        self._show_current()

    def _navigate_folders(self, delta: int) -> None:
        if not self._folders:
            return
        new_idx = max(0, min(len(self._folders) - 1, self._current_folder_idx + delta))
        self._current_folder_idx = new_idx
        self._folder_list.selection_clear(0, "end")
        self._folder_list.selection_set(new_idx)
        self._folder_list.see(new_idx)
        self._load_folder(self._folders[new_idx])

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def _on_viewer_resize(self, event: tk.Event) -> None:
        """Debounced redraw when the viewer label changes size."""
        if self._resize_pending:
            return
        self._resize_pending = True
        self.after(80, self._on_viewer_resize_flush)

    def _on_viewer_resize_flush(self) -> None:
        self._resize_pending = False
        self._show_current()

    def _show_current(self) -> None:
        if not self._images:
            self._img_label.config(image="", text="No images", fg="#555")
            self._img_info.config(text="")
            return

        path = self._images[self._current_idx]
        orig_w, orig_h = 0, 0

        # Use the viewer frame's allocated size as the display budget.
        # Fall back to MAIN_MAX_SIZE until the widget is realized.
        lw = self._viewer_frame.winfo_width()
        lh = self._viewer_frame.winfo_height()
        if lw < 10 or lh < 10:
            max_size = MAIN_MAX_SIZE
        else:
            max_size = (lw, lh)

        # Decide what to render: focus overlay (if available + enabled) or plain image
        photo: ImageTk.PhotoImage | None = None
        if self._focus_overlay_on and path in self._focus_score_maps:
            _, _, _, alpha, _, _, _, _, _ = self._current_focus_params()
            photo = render_focus_overlay(path, self._focus_score_maps[path],
                                         alpha, max_size)
            try:
                im = Image.open(path)
                orig_w, orig_h = im.size
            except Exception:
                pass

        if photo is None:
            try:
                im = Image.open(path).convert("RGB")
                orig_w, orig_h = im.size
                im.thumbnail(max_size, Image.LANCZOS)
                photo = ImageTk.PhotoImage(im)
            except Exception as exc:
                self._img_label.config(image="", text=f"Error: {exc}", fg="#e74c3c")

        if photo is not None:
            self._main_photo = photo
            self._img_label.config(image=self._main_photo, text="")

        is_bg = self._folder_bg_flags.get(self._current_folder, False) \
                if self._current_folder else False
        is_focused = self._focus_flags.get(path, False)

        parts: list[str] = [f"{path.name}   {orig_w}x{orig_h}   "
                            f"{self._current_idx + 1}/{len(self._images)}"]
        if is_bg:
            parts.append("[BACKGROUND FOLDER]")
        scores = self._focus_scores.get(path)
        if scores is not None:
            if is_focused:
                parts.append(f"[IN FOCUS  peak={scores.peak:.3f}]")
            else:
                parts.append(f"[peak={scores.peak:.3f}]")

        color = "#1e90ff" if is_focused else "#e74c3c" if is_bg else "#666"
        self._img_info.config(text="  ".join(parts), fg=color)

        for i, tw in enumerate(self._thumb_widgets):
            tw.set_selected(i == self._current_idx)

        if self._thumb_widgets:
            frac = self._current_idx / max(len(self._thumb_widgets) - 1, 1)
            self._thumb_canvas.xview_moveto(max(0.0, frac - 0.1))

    # ------------------------------------------------------------------
    # Thumbnails
    # ------------------------------------------------------------------

    def _rebuild_thumbs(self) -> None:
        for w in self._thumb_frame.winfo_children():
            w.destroy()
        self._thumb_widgets.clear()

        is_bg_folder = self._folder_bg_flags.get(self._current_folder, False) \
                       if self._current_folder else False
        for path in self._images:
            is_focused = self._focus_flags.get(path, False)
            tw = Thumbnail(self._thumb_frame, path,
                           on_click=self._thumb_clicked,
                           is_bg=is_bg_folder,
                           is_focused=is_focused,
                           bg="#0d0d1a")
            tw.pack(side="left", padx=3, pady=4)
            self._thumb_widgets.append(tw)

    def _thumb_clicked(self, path: Path) -> None:
        if path in self._images:
            self._current_idx = self._images.index(path)
            self._show_current()

    # ------------------------------------------------------------------
    # Background analysis
    # ------------------------------------------------------------------

    def _set_bg_folder(self) -> None:
        if not self._current_folder:
            messagebox.showinfo("No folder selected",
                                "Open a root folder and select a folder first.")
            return
        self._bg_folder = self._current_folder
        self._bg_images = [p for p in self._bg_folder.iterdir()
                           if p.is_file() and is_image(p)]
        self._bg_label.config(
            text=f"BG: .../{self._bg_folder.name}  ({len(self._bg_images)} imgs)",
            fg="#2ecc71",
        )
        if self._rebuild_profile():
            self._run_bg_analysis()

    def _run_bg_analysis(self) -> None:
        if not self._bg_profile or not self._folders or self._analysis_running:
            return

        self._analysis_running = True
        self._folder_bg_flags = {}
        n_folders = len(self._folders)
        self._status.config(text=f"Analysing 0 / {n_folders} folders...")
        self.update_idletasks()

        bg_mean, tolerance = self._bg_profile
        _, _, img_thresh, fold_thresh = self._current_bg_thresholds()

        def worker() -> None:
            results: dict[Path, bool] = {}
            if self._bg_folder is not None:
                results[self._bg_folder] = True
            targets = [f for f in self._folders if f != self._bg_folder]
            for i, folder in enumerate(targets):
                is_bg, _, _ = analyze_folder_is_background(
                    folder, bg_mean, tolerance, img_thresh, fold_thresh)
                results[folder] = is_bg
                self.after(0, lambda done=i + 1: self._status.config(
                    text=f"Analysing {done} / {n_folders} folders..."))
            self.after(0, lambda: self._apply_bg_analysis(
                results, bg_mean, tolerance, img_thresh, fold_thresh))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_bg_analysis(self,
                            flags: dict[Path, bool],
                            bg_mean: np.ndarray,
                            tolerance: np.ndarray,
                            img_thresh: float,
                            fold_thresh: float) -> None:
        self._folder_bg_flags = flags
        self._analysis_running = False

        n_bg = sum(flags.values())
        self._status.config(text=f"Done — {n_bg} / {len(flags)} folders flagged as background")

        color_tol, std_mult, _, _ = self._current_bg_thresholds()
        print("\n" + "=" * 60)
        print("BACKGROUND ANALYSIS RESULTS")
        print(f"Reference folder : {self._bg_folder}")
        print(f"Profile mean RGB : {bg_mean.round(1)}")
        print(f"Tolerance        : {tolerance.round(1)}")
        print(f"Thresholds       : color_tol={color_tol:.1f}  std_mult={std_mult:.2f}"
              f"  image={img_thresh:.2f}  folder={fold_thresh:.2f}")
        print("=" * 60)
        bg_folders = sorted(p for p, v in flags.items() if v)
        ok_folders = sorted(p for p, v in flags.items() if not v)
        print(f"\nFLAGGED AS BACKGROUND ({len(bg_folders)}):")
        for p in bg_folders:
            rel = p.relative_to(self._root_dir) if self._root_dir else p
            marker = " [reference]" if p == self._bg_folder else ""
            print(f"  [BG] {rel}{marker}")
        print(f"\nNOT BACKGROUND ({len(ok_folders)}):")
        for p in ok_folders:
            rel = p.relative_to(self._root_dir) if self._root_dir else p
            print(f"  [  ] {rel}")
        print("=" * 60 + "\n")

        self._refresh_folder_list()
        if self._current_folder:
            self._rebuild_thumbs()
            self._show_current()

    # ------------------------------------------------------------------
    # Focus detection
    # ------------------------------------------------------------------

    def _run_focus_detection(self) -> None:
        if not self._images:
            messagebox.showinfo("No folder", "Select a folder first.")
            return
        if self._focus_running:
            return

        self._focus_running = True
        self._focus_score_maps = {}
        self._focus_scores = {}
        self._focus_flags = {}

        kernel, radius, grad_thresh, _alpha, peak_pctile, score_cutoff, ceiling, method, lap_window = \
            self._current_focus_params()

        # Clamp kernel to valid odd value (only used by Tenengrad)
        kernel = max(1, kernel)
        if kernel % 2 == 0:
            kernel += 1

        images = list(self._images)
        n = len(images)
        self._focus_status.config(text=f"Processing 0 / {n}...")
        self.update_idletasks()

        def worker() -> None:
            score_maps: dict[Path, np.ndarray] = {}
            scores_map: dict[Path, FocusScores] = {}
            flags: dict[Path, bool] = {}

            for i, path in enumerate(images):
                norm_map, scores = compute_focus_for_image(
                    path, kernel, radius, grad_thresh, peak_pctile, ceiling,
                    method=method, laplacian_window=lap_window,
                )
                if norm_map is not None and scores is not None:
                    score_maps[path] = norm_map
                    scores_map[path] = scores
                    flags[path] = scores.peak >= score_cutoff
                self.after(0, lambda done=i + 1: self._focus_status.config(
                    text=f"Processing {done} / {n}..."))

            self.after(0, lambda: self._apply_focus_results(
                score_maps, scores_map, flags, score_cutoff, ceiling))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_focus_results(self,
                              score_maps: dict[Path, np.ndarray],
                              scores_map: dict[Path, FocusScores],
                              flags: dict[Path, bool],
                              score_cutoff: float,
                              ceiling: float = 1.0) -> None:
        self._focus_score_maps = score_maps
        self._focus_scores = scores_map
        self._focus_flags = flags
        self._focus_running = False

        n_focused = sum(flags.values())
        self._focus_status.config(
            text=f"Done — {n_focused} / {len(flags)} images flagged as in focus "
                 f"(peak >= {score_cutoff:.2f})")

        # Console output
        print("\n" + "=" * 60)
        print("FOCUS DETECTION RESULTS")
        print(f"Folder : {self._current_folder}")
        kernel, radius, grad_thresh, _, peak_pctile, _, _, method, lap_window = \
            self._current_focus_params()
        if method == FOCUS_METHOD_LAPLACIAN:
            print(f"Method : laplacian  window={lap_window}  radius={radius}"
                  f"  grad_thresh={grad_thresh}")
        else:
            print(f"Method : tenengrad  kernel={kernel}  radius={radius}"
                  f"  grad_thresh={grad_thresh}")
        print(f"Params : peak_pctile={peak_pctile}  cutoff={score_cutoff:.2f}"
              f"  ceiling={ceiling:.2f}")
        print("=" * 60)
        for path, scores in sorted(scores_map.items(), key=lambda x: x[1].peak, reverse=True):
            flag = "[FOCUS]" if flags.get(path, False) else "[     ]"
            print(f"  {flag}  peak={scores.peak:.3f}  center={scores.center:.3f}"
                  f"  whole={scores.whole:.3f}  {path.name}")
        print("=" * 60 + "\n")

        # Update thumbnail borders
        for i, tw in enumerate(self._thumb_widgets):
            path = self._images[i]
            tw.set_focus_flag(flags.get(path, False))
            tw.set_selected(i == self._current_idx)

        self._show_current()


if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()