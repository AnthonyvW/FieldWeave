from __future__ import annotations

import argparse
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# --red-scale 0.997748 --blue-scale 1.005603 --red-shift -0.5941 +0.0007 --blue-shift +0.0431 -0.6501
# ---------------------------------------------------------------------------
# Core correction logic
# ---------------------------------------------------------------------------

def scale_channel(channel: np.ndarray, scale: float) -> np.ndarray:
    h, w = channel.shape
    cx, cy = w / 2.0, h / 2.0
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = ((map_x - cx) / scale + cx).astype(np.float32)
    map_y = ((map_y - cy) / scale + cy).astype(np.float32)
    return cv2.remap(channel, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def shift_channel(channel: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = channel.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(channel, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def correct_chromatic_aberration(
    image: np.ndarray,
    red_scale: float = 1.0,
    blue_scale: float = 1.0,
    red_shift: tuple[float, float] = (0.0, 0.0),
    blue_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    b, g, r = cv2.split(image)
    if red_scale != 1.0:
        r = scale_channel(r, red_scale)
    if blue_scale != 1.0:
        b = scale_channel(b, blue_scale)
    rdx, rdy = red_shift
    if rdx != 0.0 or rdy != 0.0:
        r = shift_channel(r, rdx, rdy)
    bdx, bdy = blue_shift
    if bdx != 0.0 or bdy != 0.0:
        b = shift_channel(b, bdx, bdy)
    return cv2.merge([b, g, r])


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

@dataclass
class BlockDiagnostics:
    """Per-block measurements produced by auto_detect_ca."""
    centres_x: np.ndarray          # image-space x of each block centre
    centres_y: np.ndarray          # image-space y
    red_dx: np.ndarray             # measured red  displacement x (green-relative)
    red_dy: np.ndarray
    blue_dx: np.ndarray            # measured blue displacement x
    blue_dy: np.ndarray
    red_scale: float
    blue_scale: float
    red_shift: tuple[float, float]
    blue_shift: tuple[float, float]
    # Predicted displacements from the fitted model at each block centre
    red_dx_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    red_dy_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    blue_dx_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    blue_dy_pred: np.ndarray = field(default_factory=lambda: np.array([]))


def auto_detect_ca(
    image: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[float, float, tuple[float, float], tuple[float, float], BlockDiagnostics]:
    """
    Estimate radial CA scale factors and uniform lateral shifts using block matching.

    Tiles the image into 64×64 Hann-windowed blocks, measures the sub-pixel
    (dx, dy) displacement of red and blue relative to green via phase correlation,
    then fits the model:
        dx = (scale - 1) * (ix - cx) + tx
        dy = (scale - 1) * (iy - cy) + ty
    jointly for scale and lateral shift via re-weighted least squares.

    Returns
    -------
    (red_scale, blue_scale, red_shift, blue_shift, diagnostics)
    """
    h, w = image.shape[:2]

    # Downscale by half so that sub-pixel CA displacements become whole-pixel
    # offsets that phase correlation can reliably detect. Scale is dimensionless
    # so it transfers directly; shifts are in full-resolution pixels so they are
    # multiplied by 2 before being returned.
    small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    cx, cy = sw / 2.0, sh / 2.0
    f32 = small.astype(np.float32)
    b, g, r = cv2.split(f32)

    block_size = 64
    step = block_size // 2

    if region is not None:
        rx0, ry0, rx1, ry1 = region
        # Region is in full-res coords; convert to half-res
        rx0, ry0, rx1, ry1 = rx0 // 2, ry0 // 2, rx1 // 2, ry1 // 2
        rx0, rx1 = max(0, min(rx0, rx1)), min(sw, max(rx0, rx1))
        ry0, ry1 = max(0, min(ry0, ry1)), min(sh, max(ry0, ry1))
    else:
        rx0, ry0, rx1, ry1 = 0, 0, sw, sh

    max_radius = np.sqrt(cx**2 + cy**2)

    centres_x: list[float] = []
    centres_y: list[float] = []
    red_dx_obs: list[float] = []
    red_dy_obs: list[float] = []
    blue_dx_obs: list[float] = []
    blue_dy_obs: list[float] = []

    hann = np.outer(np.hanning(block_size), np.hanning(block_size)).astype(np.float32)

    for by in range(ry0, ry1 - block_size + 1, step):
        for bx in range(rx0, rx1 - block_size + 1, step):
            g_block = g[by:by + block_size, bx:bx + block_size] * hann
            r_block = r[by:by + block_size, bx:bx + block_size] * hann
            b_block = b[by:by + block_size, bx:bx + block_size] * hann

            # phaseCorrelate(a, b) = shift of b relative to a; negate to get
            # the correction direction.
            (rdx, rdy), _ = cv2.phaseCorrelate(g_block, r_block)
            (bdx, bdy), _ = cv2.phaseCorrelate(g_block, b_block)
            rdx, rdy, bdx, bdy = -rdx, -rdy, -bdx, -bdy

            # No hard displacement cap — the robust fitting handles outliers.
            # Only reject physically impossible values (wrap-around artifacts
            # from phase correlation appear as large jumps near ±block_size/2).
            if abs(rdx) > block_size // 4 or abs(rdy) > block_size // 4:
                continue
            if abs(bdx) > block_size // 4 or abs(bdy) > block_size // 4:
                continue

            centres_x.append(bx + block_size / 2.0)
            centres_y.append(by + block_size / 2.0)
            red_dx_obs.append(rdx)
            red_dy_obs.append(rdy)
            blue_dx_obs.append(bdx)
            blue_dy_obs.append(bdy)

    null_diag = BlockDiagnostics(
        np.array([]), np.array([]), np.array([]), np.array([]),
        np.array([]), np.array([]), 1.0, 1.0, (0.0, 0.0), (0.0, 0.0),
    )

    if len(centres_x) < 4:
        return 1.0, 1.0, (0.0, 0.0), (0.0, 0.0), null_diag

    cx_arr = np.array(centres_x) - cx
    cy_arr = np.array(centres_y) - cy
    n = len(cx_arr)
    A_x = np.column_stack([cx_arr, np.ones(n), np.zeros(n)])
    A_y = np.column_stack([cy_arr, np.zeros(n), np.ones(n)])
    A = np.vstack([A_x, A_y])

    # Radial weights for fitting: outer blocks contribute more since they have
    # larger CA signal. Normalise so weights sum to n (keeps scale comparable).
    radial_w = np.sqrt(cx_arr**2 + cy_arr**2) / max_radius
    radial_w = np.maximum(radial_w, 0.05)   # floor so centre blocks aren't zeroed
    radial_w = radial_w / radial_w.mean()   # normalise mean to 1
    fit_base_w = np.concatenate([radial_w, radial_w])  # same weight for x and y obs

    def fit_channel(
        dx_obs: list[float], dy_obs: list[float]
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        b_vec = np.concatenate([dx_obs, dy_obs])

        # Initial weighted fit (radial weighting)
        W0 = np.diag(fit_base_w)
        result, _, _, _ = np.linalg.lstsq(W0 @ A, W0 @ b_vec, rcond=None)

        # Outlier re-weighting: combine radial weight with Huber residual weight
        residuals = np.abs(b_vec - A @ result)
        mad = np.median(residuals) + 1e-9
        huber_w = np.minimum(1.0, mad / (residuals + 1e-9))
        combined_w = fit_base_w * huber_w
        W = np.diag(combined_w)
        result_w, _, _, _ = np.linalg.lstsq(W @ A, W @ b_vec, rcond=None)

        scale_offset, tx, ty = result_w
        pred = A @ result_w
        pred_dx = pred[:n]
        pred_dy = pred[n:]
        return float(1.0 + scale_offset), float(tx), float(ty), pred_dx, pred_dy

    red_scale, red_tx, red_ty, red_dx_pred, red_dy_pred = fit_channel(red_dx_obs, red_dy_obs)
    blue_scale, blue_tx, blue_ty, blue_dx_pred, blue_dy_pred = fit_channel(blue_dx_obs, blue_dy_obs)

    red_scale = max(0.980, min(1.020, red_scale))
    blue_scale = max(0.980, min(1.020, blue_scale))
    # Shifts were measured in half-res pixels; multiply by 2 for full-res
    red_shift = (max(-10.0, min(10.0, red_tx * 2)), max(-10.0, min(10.0, red_ty * 2)))
    blue_shift = (max(-10.0, min(10.0, blue_tx * 2)), max(-10.0, min(10.0, blue_ty * 2)))

    # Block centres are in half-res coords; scale back to full-res for the overlay
    diag = BlockDiagnostics(
        centres_x=np.array(centres_x) * 2,
        centres_y=np.array(centres_y) * 2,
        red_dx=np.array(red_dx_obs),
        red_dy=np.array(red_dy_obs),
        blue_dx=np.array(blue_dx_obs),
        blue_dy=np.array(blue_dy_obs),
        red_scale=red_scale,
        blue_scale=blue_scale,
        red_shift=red_shift,
        blue_shift=blue_shift,
        red_dx_pred=red_dx_pred,
        red_dy_pred=red_dy_pred,
        blue_dx_pred=blue_dx_pred,
        blue_dy_pred=blue_dy_pred,
    )

    return red_scale, blue_scale, red_shift, blue_shift, diag


def score_correction(
    image: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[float, float, int]:
    # Measure mean RMS channel displacement across all blocks.
    # Runs the same block matching as auto_detect_ca but skips fitting,
    # returning raw mean displacement of red and blue relative to green.
    # Lower = better aligned.
    h, w = image.shape[:2]
    small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    f32 = small.astype(np.float32)
    b, g, r = cv2.split(f32)

    block_size = 64
    step = block_size // 2

    if region is not None:
        rx0, ry0, rx1, ry1 = region
        rx0, ry0, rx1, ry1 = rx0 // 2, ry0 // 2, rx1 // 2, ry1 // 2
        rx0, rx1 = max(0, min(rx0, rx1)), min(sw, max(rx0, rx1))
        ry0, ry1 = max(0, min(ry0, ry1)), min(sh, max(ry0, ry1))
    else:
        rx0, ry0, rx1, ry1 = 0, 0, sw, sh

    hann = np.outer(np.hanning(block_size), np.hanning(block_size)).astype(np.float32)

    red_disps: list[float] = []
    blue_disps: list[float] = []

    for by in range(ry0, ry1 - block_size + 1, step):
        for bx in range(rx0, rx1 - block_size + 1, step):
            g_block = g[by:by + block_size, bx:bx + block_size] * hann
            r_block = r[by:by + block_size, bx:bx + block_size] * hann
            b_block = b[by:by + block_size, bx:bx + block_size] * hann

            (rdx, rdy), _ = cv2.phaseCorrelate(g_block, r_block)
            (bdx, bdy), _ = cv2.phaseCorrelate(g_block, b_block)

            if abs(rdx) > block_size // 4 or abs(rdy) > block_size // 4:
                continue
            if abs(bdx) > block_size // 4 or abs(bdy) > block_size // 4:
                continue

            red_disps.append(rdx**2 + rdy**2)
            blue_disps.append(bdx**2 + bdy**2)

    if not red_disps:
        return 0.0, 0.0, 0

    red_rms = float(np.sqrt(np.mean(red_disps)))
    blue_rms = float(np.sqrt(np.mean(blue_disps)))
    return red_rms, blue_rms, len(red_disps)


# ---------------------------------------------------------------------------
# GUI constants
# ---------------------------------------------------------------------------

SIDEBAR_W = 260
TOOLBAR_H = 44
MIN_W = 640
MIN_H = 480

ACCENT = "#e8552a"
BG = "#111213"
SURFACE = "#1c1e1f"
BORDER = "#2e3133"
TEXT = "#e8e6e3"
MUTED = "#6b7073"
REGION_COLOR = "#f0c040"
RED_VEC = "#ff4444"
BLUE_VEC = "#4488ff"
PRED_VEC = "#ffffff"

ZOOM_MIN = 0.05
ZOOM_MAX = 32.0
ZOOM_STEP = 1.15



def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CAApp(tk.Tk):
    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.title("Chromatic Aberration Corrector")
        self.configure(bg=BG)
        self.minsize(MIN_W, MIN_H)

        self._source: np.ndarray | None = None
        self._corrected: np.ndarray | None = None
        self._composite_photo: ImageTk.PhotoImage | None = None

        self._split_frac: float = 0.5
        self._dragging_split = False

        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._panning = False
        self._pan_start: tuple[int, int] = (0, 0)
        self._pan_origin: tuple[float, float] = (0.0, 0.0)

        self._canvas_w: int = 840
        self._canvas_h: int = 580

        self._region: tuple[int, int, int, int] | None = None
        self._selecting_region = False
        self._region_drag_start: tuple[int, int] | None = None
        self._region_drag_current: tuple[int, int] | None = None

        self._block_diag: BlockDiagnostics | None = None
        self._show_blocks = False

        self._pending_after: str | None = None
        self._resize_after: str | None = None
        self._worker: threading.Thread | None = None
        self._result_queue: np.ndarray | None = None

        self._build_ui()
        self.after(50, self._poll_worker)

        if initial_path is not None:
            self.after(100, lambda: self._load_path(initial_path))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.geometry(f"{self._canvas_w + SIDEBAR_W}x{self._canvas_h + TOOLBAR_H}")

        toolbar = tk.Frame(self, bg=SURFACE, height=TOOLBAR_H)
        toolbar.pack(fill="x", side="top")
        toolbar.pack_propagate(False)

        tk.Label(toolbar, text="CA CORRECTOR", bg=SURFACE, fg=ACCENT,
                 font=("Courier", 11, "bold"), padx=14).pack(side="left", pady=10)

        self._open_btn = self._toolbar_btn(toolbar, "OPEN IMAGE", self._open_image, primary=True)
        self._open_btn.pack(side="left", padx=8, pady=8)

        self._auto_btn = self._toolbar_btn(toolbar, "AUTO DETECT", self._auto_detect)
        self._auto_btn.config(state="disabled")
        self._auto_btn.pack(side="left", padx=4, pady=8)

        self._blocks_btn = self._toolbar_btn(toolbar, "SHOW BLOCKS", self._toggle_blocks)
        self._blocks_btn.config(state="disabled")
        self._blocks_btn.pack(side="left", padx=4, pady=8)

        self._region_btn = self._toolbar_btn(toolbar, "SELECT REGION", self._toggle_region_select)
        self._region_btn.config(state="disabled")
        self._region_btn.pack(side="left", padx=4, pady=8)

        self._clear_region_btn = self._toolbar_btn(toolbar, "CLEAR REGION", self._clear_region)
        self._clear_region_btn.config(state="disabled")
        self._clear_region_btn.pack(side="left", padx=4, pady=8)

        self._save_btn = self._toolbar_btn(toolbar, "SAVE RESULT", self._save_image)
        self._save_btn.config(state="disabled")
        self._save_btn.pack(side="left", padx=4, pady=8)

        self._score_btn = self._toolbar_btn(toolbar, "SCORE", self._score_current)
        self._score_btn.config(state="disabled")
        self._score_btn.pack(side="left", padx=4, pady=8)

        self._zoom_btn = self._toolbar_btn(toolbar, "RESET VIEW", self._reset_view)
        self._zoom_btn.pack(side="left", padx=4, pady=8)

        self._status_var = tk.StringVar(value="Open an image to begin.")
        tk.Label(toolbar, textvariable=self._status_var, bg=SURFACE, fg=MUTED,
                 font=("Courier", 8), padx=14).pack(side="right", pady=10)

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(main, bg="#0a0b0c", highlightthickness=0,
                                 cursor="sb_h_double_arrow")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan_motion)
        self._canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self._canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self._canvas.bind("<B3-Motion>", self._on_pan_motion)
        self._canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self._canvas.bind("<MouseWheel>", self._on_scroll)
        self._canvas.bind("<Button-4>", self._on_scroll)
        self._canvas.bind("<Button-5>", self._on_scroll)

        self._draw_placeholder()

        sidebar = tk.Frame(main, bg=SURFACE, width=SIDEBAR_W)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)

        self._sliders: dict[str, tk.DoubleVar] = {}
        self._build_sidebar(sidebar)

    def _toolbar_btn(self, parent: tk.Frame, text: str, cmd: object, primary: bool = False) -> tk.Button:
        return tk.Button(
            parent, text=text, command=cmd,
            bg=ACCENT if primary else BORDER,
            fg="white" if primary else TEXT,
            relief="flat", font=("Courier", 9, "bold"), padx=10, pady=4,
            activebackground="#c44020" if primary else "#3a3e40",
            activeforeground="white" if primary else TEXT,
            cursor="hand2",
        )

    def _build_sidebar(self, parent: tk.Frame) -> None:
        SECTION_DEFAULTS: dict[str, list[tuple[str, float]]] = {
            "RADIAL SCALE": [("red_scale", 1.0), ("blue_scale", 1.0)],
            "RED SHIFT (px)": [("red_dx", 0.0), ("red_dy", 0.0)],
            "BLUE SHIFT (px)": [("blue_dx", 0.0), ("blue_dy", 0.0)],
        }

        def section(label: str) -> None:
            header = tk.Frame(parent, bg=SURFACE)
            header.pack(fill="x", padx=14, pady=(14, 0))
            tk.Label(header, text=label, bg=SURFACE, fg=ACCENT,
                     font=("Courier", 8, "bold"), anchor="w").pack(side="left")
            keys = SECTION_DEFAULTS[label]
            tk.Button(
                header, text="RESET",
                command=lambda k=keys: self._reset_section(k),
                bg=SURFACE, fg=MUTED, relief="flat",
                font=("Courier", 7), padx=4, pady=0,
                activebackground=BORDER, activeforeground=TEXT, cursor="hand2",
            ).pack(side="right")
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(2, 6))

        def slider_row(key: str, label: str, from_: float, to: float,
                       default: float, resolution: float) -> None:
            var = tk.DoubleVar(value=default)
            self._sliders[key] = var

            row = tk.Frame(parent, bg=SURFACE)
            row.pack(fill="x", padx=14)
            tk.Label(row, text=label, bg=SURFACE, fg=TEXT,
                     font=("Courier", 8), anchor="w", width=14).pack(side="left")
            val_lbl = tk.Label(row, text=f"{default:+.4f}", bg=SURFACE, fg=MUTED,
                               font=("Courier", 8), width=8)
            val_lbl.pack(side="right")

            tk.Scale(
                parent, variable=var, from_=from_, to=to, resolution=resolution,
                orient="horizontal", bg=SURFACE, fg=TEXT, troughcolor=BORDER,
                activebackground=ACCENT, highlightthickness=0, sliderrelief="flat",
                bd=0, showvalue=False, length=SIDEBAR_W - 28,
            ).pack(fill="x", padx=14, pady=(1, 6))

            def on_change(*_: object) -> None:
                val_lbl.config(text=f"{var.get():+.4f}")
                self._schedule_update()

            var.trace_add("write", on_change)

        section("RADIAL SCALE")
        slider_row("red_scale",  "Red",  0.980, 1.020, 1.0, 0.0001)
        slider_row("blue_scale", "Blue", 0.980, 1.020, 1.0, 0.0001)

        section("RED SHIFT (px)")
        slider_row("red_dx", "X", -10.0, 10.0, 0.0, 0.1)
        slider_row("red_dy", "Y", -10.0, 10.0, 0.0, 0.1)

        section("BLUE SHIFT (px)")
        slider_row("blue_dx", "X", -10.0, 10.0, 0.0, 0.1)
        slider_row("blue_dy", "Y", -10.0, 10.0, 0.0, 0.1)

        tk.Frame(parent, bg=SURFACE).pack(expand=True, fill="both")

        tk.Button(
            parent, text="RESET ALL", command=self._reset_sliders,
            bg=BORDER, fg=MUTED, relief="flat", font=("Courier", 8, "bold"), pady=6,
            activebackground="#3a3e40", activeforeground=TEXT, cursor="hand2",
        ).pack(fill="x", padx=14, pady=10)

    # ------------------------------------------------------------------
    # Slider helpers
    # ------------------------------------------------------------------

    def _reset_section(self, keys: list[tuple[str, float]]) -> None:
        for key, val in keys:
            self._sliders[key].set(val)

    def _reset_sliders(self) -> None:
        for key, val in [("red_scale", 1.0), ("blue_scale", 1.0),
                         ("red_dx", 0.0), ("red_dy", 0.0),
                         ("blue_dx", 0.0), ("blue_dy", 0.0)]:
            self._sliders[key].set(val)

    def _get_params(self) -> dict[str, object]:
        return {
            "red_scale": self._sliders["red_scale"].get(),
            "blue_scale": self._sliders["blue_scale"].get(),
            "red_shift": (self._sliders["red_dx"].get(), self._sliders["red_dy"].get()),
            "blue_shift": (self._sliders["blue_dx"].get(), self._sliders["blue_dy"].get()),
        }

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("All", "*.*")]
        )
        if path:
            self._load_path(Path(path))

    def _load_path(self, image_path: Path) -> None:
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        except OSError as e:
            messagebox.showerror("Error", str(e))
            return
        if img is None or img.ndim != 3 or img.shape[2] < 3:
            messagebox.showerror("Error", "Could not open image or image is not colour.")
            return
        self._source = img
        self._corrected = img.copy()
        self._auto_btn.config(state="normal")
        self._region_btn.config(state="normal")
        self._save_btn.config(state="normal")
        self._score_btn.config(state="normal")
        self._status_var.set(f"{image_path.name}  [{img.shape[1]}x{img.shape[0]}]")
        self._fit_to_canvas()
        self._schedule_update()
        self._redraw()

    def _save_image(self) -> None:
        if self._corrected is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            cv2.imwrite(path, self._corrected)
            self._status_var.set(f"Saved: {Path(path).name}")
            params = self._get_params()
            rs = params["red_shift"]
            bs = params["blue_shift"]
            print(f"[SAVED] {path}")
            print(
                f"  python correct_ca.py INPUT --no-gui -o OUTPUT"
                f" --red-scale {params['red_scale']:.6f}"
                f" --blue-scale {params['blue_scale']:.6f}"
                f" --red-shift {rs[0]:+.4f} {rs[1]:+.4f}"
                f" --blue-shift {bs[0]:+.4f} {bs[1]:+.4f}"
            )
        except OSError as e:
            messagebox.showerror("Save Error", str(e))

    # ------------------------------------------------------------------
    # Block diagnostics overlay
    # ------------------------------------------------------------------

    def _toggle_blocks(self) -> None:
        self._show_blocks = not self._show_blocks
        if self._show_blocks:
            self._blocks_btn.config(bg=ACCENT, fg="white")
        else:
            self._blocks_btn.config(bg=BORDER, fg=TEXT)
        self._redraw()

    def _draw_block_vectors(self) -> None:
        """Draw per-block measured and model-predicted displacement arrows.
        
        Arrows are auto-scaled so the largest measured displacement spans ~40 canvas
        pixels, making them visible regardless of how small the actual CA offsets are.
        Block centres are also marked with small dots so coverage is visible even when
        displacements are so tiny that arrows would be invisible at any scale.
        """
        diag = self._block_diag
        if diag is None or len(diag.centres_x) == 0:
            return

        has_pred = len(diag.red_dx_pred) == len(diag.centres_x)

        all_dx = np.concatenate([diag.red_dx, diag.blue_dx])
        all_dy = np.concatenate([diag.red_dy, diag.blue_dy])
        max_disp = float(np.max(np.sqrt(all_dx**2 + all_dy**2)))
        # Scale so the largest arrow is 40 canvas pixels; floor prevents div-by-zero
        arrow_scale = 40.0 / max(max_disp, 1e-6)

        for i in range(len(diag.centres_x)):
            ix = float(diag.centres_x[i])
            iy = float(diag.centres_y[i])
            bx, by = self._image_to_canvas_coords(ix, iy)

            # Dot marking the block centre
            self._canvas.create_oval(bx - 2, by - 2, bx + 2, by + 2,
                                      fill=MUTED, outline="")

            self._draw_arrow(bx, by,
                float(diag.red_dx[i]) * arrow_scale,
                float(diag.red_dy[i]) * arrow_scale,
                RED_VEC, width=1)
            self._draw_arrow(bx, by,
                float(diag.blue_dx[i]) * arrow_scale,
                float(diag.blue_dy[i]) * arrow_scale,
                BLUE_VEC, width=1)

            if has_pred:
                self._draw_arrow(bx, by,
                    float(diag.red_dx_pred[i]) * arrow_scale,
                    float(diag.red_dy_pred[i]) * arrow_scale,
                    PRED_VEC, width=1, stipple=True)
                self._draw_arrow(bx, by,
                    float(diag.blue_dx_pred[i]) * arrow_scale,
                    float(diag.blue_dy_pred[i]) * arrow_scale,
                    PRED_VEC, width=1, stipple=True)

        lx, ly = 10, self._canvas_h - 56
        self._canvas.create_text(lx, ly,      text="── measured red",     anchor="w", fill=RED_VEC,  font=("Courier", 7))
        self._canvas.create_text(lx, ly + 12, text="── measured blue",    anchor="w", fill=BLUE_VEC, font=("Courier", 7))
        self._canvas.create_text(lx, ly + 24, text="── model prediction", anchor="w", fill=PRED_VEC, font=("Courier", 7))
        self._canvas.create_text(lx, ly + 36, text="·  block centre",     anchor="w", fill=MUTED,    font=("Courier", 7))
        self._canvas.create_text(lx, ly + 48,
            text=f"max disp {max_disp:.4f}px → ×{arrow_scale:.0f}  n={len(diag.centres_x)} blocks",
            anchor="w", fill=MUTED, font=("Courier", 7))

    def _draw_arrow(self, x: float, y: float, dx: float, dy: float,
                    colour: str, width: int = 1, stipple: bool = False) -> None:
        ex, ey = x + dx, y + dy
        kw: dict[str, object] = {"fill": colour, "width": width, "arrow": tk.LAST, "arrowshape": (5, 6, 2)}
        if stipple:
            kw["dash"] = (2, 2)
        self._canvas.create_line(x, y, ex, ey, **kw)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------

    def _toggle_region_select(self) -> None:
        self._selecting_region = not self._selecting_region
        if self._selecting_region:
            self._region_btn.config(bg=REGION_COLOR, fg="#111213")
            self._canvas.config(cursor="crosshair")
            self._status_var.set("Drag to select region for auto-detect.")
        else:
            self._region_btn.config(bg=BORDER, fg=TEXT)
            self._canvas.config(cursor="sb_h_double_arrow")
            self._region_drag_start = None
            self._region_drag_current = None
            self._status_var.set("Region selection cancelled.")
            self._redraw()

    def _clear_region(self) -> None:
        self._region = None
        self._clear_region_btn.config(state="disabled")
        self._redraw()

    def _canvas_to_image_coords(self, cx: float, cy: float) -> tuple[int, int]:
        ix = int(cx / self._zoom + self._pan_x)
        iy = int(cy / self._zoom + self._pan_y)
        if self._source is not None:
            h, w = self._source.shape[:2]
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))
        return ix, iy

    def _image_to_canvas_coords(self, ix: float, iy: float) -> tuple[float, float]:
        cx = (ix - self._pan_x) * self._zoom
        cy = (iy - self._pan_y) * self._zoom
        return cx, cy

    # ------------------------------------------------------------------
    # Auto-detect
    # ------------------------------------------------------------------

    def _auto_detect(self) -> None:
        if self._source is None:
            return

        n_passes = 3
        region_note = " (region)" if self._region is not None else ""

        # Always start from the original source with the current slider correction
        # pre-applied. This means each button click is an independent measurement
        # of the residual CA on the current result, so repeated clicks converge
        # rather than accumulate drift.
        params = self._get_params()
        working = correct_chromatic_aberration(
            self._source,
            red_scale=params["red_scale"],
            blue_scale=params["blue_scale"],
            red_shift=params["red_shift"],
            blue_shift=params["blue_shift"],
        )

        acc_red_scale = 1.0
        acc_blue_scale = 1.0
        acc_red_shift = (0.0, 0.0)
        acc_blue_shift = (0.0, 0.0)

        for i in range(n_passes):
            self._status_var.set(f"Auto-detect pass {i + 1}/{n_passes}...")
            self.update_idletasks()

            red_scale, blue_scale, red_shift, blue_shift, diag = auto_detect_ca(
                working, region=self._region
            )

            acc_red_scale *= red_scale
            acc_blue_scale *= blue_scale
            acc_red_shift = (acc_red_shift[0] + red_shift[0], acc_red_shift[1] + red_shift[1])
            acc_blue_shift = (acc_blue_shift[0] + blue_shift[0], acc_blue_shift[1] + blue_shift[1])

            working = correct_chromatic_aberration(
                working,
                red_scale=red_scale,
                blue_scale=blue_scale,
                red_shift=red_shift,
                blue_shift=blue_shift,
            )

        # Compose the residual correction with the existing slider values
        final_red_scale = params["red_scale"] * acc_red_scale
        final_blue_scale = params["blue_scale"] * acc_blue_scale
        final_red_shift = (params["red_shift"][0] + acc_red_shift[0],
                           params["red_shift"][1] + acc_red_shift[1])
        final_blue_shift = (params["blue_shift"][0] + acc_blue_shift[0],
                            params["blue_shift"][1] + acc_blue_shift[1])

        self._block_diag = diag
        self._blocks_btn.config(state="normal")

        self._sliders["red_scale"].set(round(final_red_scale, 6))
        self._sliders["blue_scale"].set(round(final_blue_scale, 6))
        self._sliders["red_dx"].set(round(final_red_shift[0], 4))
        self._sliders["red_dy"].set(round(final_red_shift[1], 4))
        self._sliders["blue_dx"].set(round(final_blue_shift[0], 4))
        self._sliders["blue_dy"].set(round(final_blue_shift[1], 4))

        # Score the final corrected result and print alongside the parameters
        corrected_final = correct_chromatic_aberration(
            self._source,
            red_scale=final_red_scale,
            blue_scale=final_blue_scale,
            red_shift=final_red_shift,
            blue_shift=final_blue_shift,
        )
        red_rms, blue_rms, n_scored = score_correction(corrected_final, region=self._region)
        raw_red_rms, raw_blue_rms, _ = score_correction(self._source, region=self._region)

        summary = (
            f"Auto{region_note} ({n_passes} passes): "
            f"red scale={final_red_scale:.5f} shift=({final_red_shift[0]:+.2f},{final_red_shift[1]:+.2f})  "
            f"blue scale={final_blue_scale:.5f} shift=({final_blue_shift[0]:+.2f},{final_blue_shift[1]:+.2f})  "
            f"[{len(diag.centres_x)} blocks/pass]"
        )
        print(f"[AUTO{region_note}] {n_passes} passes, {len(diag.centres_x)} blocks/pass")
        print(f"  red  scale={final_red_scale:.6f}  shift=({final_red_shift[0]:+.4f},{final_red_shift[1]:+.4f})  "
              f"RMS: {raw_red_rms:.4f} → {red_rms:.4f}")
        print(f"  blue scale={final_blue_scale:.6f}  shift=({final_blue_shift[0]:+.4f},{final_blue_shift[1]:+.4f})  "
              f"RMS: {raw_blue_rms:.4f} → {blue_rms:.4f}")
        self._status_var.set(summary)
        if self._show_blocks:
            self._redraw()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_current(self) -> None:
        if self._source is None:
            return
        self._status_var.set("Scoring...")
        self.update_idletasks()
        params = self._get_params()
        corrected = correct_chromatic_aberration(
            self._source,
            red_scale=params["red_scale"],
            blue_scale=params["blue_scale"],
            red_shift=params["red_shift"],
            blue_shift=params["blue_shift"],
        )
        red_rms, blue_rms, n = score_correction(corrected, region=self._region)
        raw_red_rms, raw_blue_rms, _ = score_correction(self._source, region=self._region)
        region_note = " (region)" if self._region is not None else ""
        msg = (
            f"Score{region_note}  [{n} blocks]  "
            f"red RMS: {raw_red_rms:.4f} → {red_rms:.4f}  "
            f"blue RMS: {raw_blue_rms:.4f} → {blue_rms:.4f}  "
            f"(lower = better aligned)"
        )
        print(f"[SCORE{region_note}] {n} blocks")
        print(f"  red  RMS: {raw_red_rms:.4f} → {red_rms:.4f}  "
              f"(red_scale={params['red_scale']:.5f} "
              f"shift=({params['red_shift'][0]:+.4f},{params['red_shift'][1]:+.4f}))")
        print(f"  blue RMS: {raw_blue_rms:.4f} → {blue_rms:.4f}  "
              f"(blue_scale={params['blue_scale']:.5f} "
              f"shift=({params['blue_shift'][0]:+.4f},{params['blue_shift'][1]:+.4f}))")
        self._status_var.set(msg)

    # ------------------------------------------------------------------
    # Zoom / pan
    # ------------------------------------------------------------------

    def _fit_to_canvas(self) -> None:
        if self._source is None:
            return
        ih, iw = self._source.shape[:2]
        self._zoom = min(self._canvas_w / iw, self._canvas_h / ih)
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _reset_view(self) -> None:
        self._fit_to_canvas()
        self._redraw()

    def _clamp_pan(self) -> None:
        if self._source is None:
            return
        ih, iw = self._source.shape[:2]
        visible_w = self._canvas_w / self._zoom
        visible_h = self._canvas_h / self._zoom
        self._pan_x = max(0.0, min(self._pan_x, max(0.0, iw - visible_w)))
        self._pan_y = max(0.0, min(self._pan_y, max(0.0, ih - visible_h)))

    def _zoom_around(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        ix = canvas_x / self._zoom + self._pan_x
        iy = canvas_y / self._zoom + self._pan_y
        self._zoom = new_zoom
        self._pan_x = ix - canvas_x / self._zoom
        self._pan_y = iy - canvas_y / self._zoom
        self._clamp_pan()

    # ------------------------------------------------------------------
    # Background correction worker
    # ------------------------------------------------------------------

    def _schedule_update(self) -> None:
        if self._source is None:
            return
        if self._pending_after is not None:
            self.after_cancel(self._pending_after)
        self._pending_after = self.after(120, self._start_worker)

    def _start_worker(self) -> None:
        self._pending_after = None
        if self._source is None:
            return
        if self._worker is not None and self._worker.is_alive():
            self._pending_after = self.after(80, self._start_worker)
            return
        params = self._get_params()
        source_copy = self._source.copy()

        def run() -> None:
            result = correct_chromatic_aberration(
                source_copy,
                red_scale=params["red_scale"],
                blue_scale=params["blue_scale"],
                red_shift=params["red_shift"],
                blue_shift=params["blue_shift"],
            )
            self._result_queue = result

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _poll_worker(self) -> None:
        if self._result_queue is not None:
            self._corrected = self._result_queue
            self._result_queue = None
            self._redraw()
        self.after(50, self._poll_worker)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_placeholder(self) -> None:
        self._canvas.delete("all")
        cx, cy = self._canvas_w // 2, self._canvas_h // 2
        self._canvas.create_text(cx, cy, text="Open an image to begin",
                                 fill=MUTED, font=("Courier", 12))

    def _render_view(self, img: np.ndarray) -> Image.Image:
        ih, iw = img.shape[:2]
        cw, ch = self._canvas_w, self._canvas_h
        x0, y0 = self._pan_x, self._pan_y
        x1, y1 = x0 + cw / self._zoom, y0 + ch / self._zoom
        x0c, y0c = max(0.0, x0), max(0.0, y0)
        x1c, y1c = min(float(iw), x1), min(float(ih), y1)

        out = Image.new("RGB", (cw, ch), (10, 11, 12))
        if x1c <= x0c or y1c <= y0c:
            return out

        crop = bgr_to_pil(img).crop((int(x0c), int(y0c), int(x1c), int(y1c)))
        dst_x = int((x0c - x0) * self._zoom)
        dst_y = int((y0c - y0) * self._zoom)
        dst_w = int((x1c - x0c) * self._zoom)
        dst_h = int((y1c - y0c) * self._zoom)

        if dst_w > 0 and dst_h > 0:
            resample = Image.NEAREST if self._zoom >= 4.0 else Image.LANCZOS
            out.paste(crop.resize((dst_w, dst_h), resample), (dst_x, dst_y))

        return out

    def _redraw(self) -> None:
        cw, ch = self._canvas_w, self._canvas_h
        if self._source is None:
            self._draw_placeholder()
            return

        src_view = self._render_view(self._source)
        cor_view = self._render_view(self._corrected if self._corrected is not None else self._source)
        sx = max(0, min(int(self._split_frac * cw), cw))

        composite = Image.new("RGB", (cw, ch))
        composite.paste(cor_view.crop((0, 0, sx, ch)), (0, 0))
        composite.paste(src_view.crop((sx, 0, cw, ch)), (sx, 0))

        self._composite_photo = ImageTk.PhotoImage(composite)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._composite_photo)

        if self._show_blocks and self._block_diag is not None:
            self._draw_block_vectors()

        self._draw_overlay(sx, cw, ch)

    def _draw_overlay(self, sx: int, cw: int, ch: int) -> None:
        self._canvas.create_line(sx, 0, sx, ch, fill=ACCENT, width=2)
        if sx > 90:
            self._canvas.create_text(sx - 10, 16, text="CORRECTED", anchor="e",
                                     fill=ACCENT, font=("Courier", 8, "bold"))
        if sx < cw - 90:
            self._canvas.create_text(sx + 10, 16, text="ORIGINAL", anchor="w",
                                     fill=TEXT, font=("Courier", 8, "bold"))

        pct = int(self._zoom * 100)
        self._canvas.create_text(cw - 6, ch - 6, text=f"{pct}%", anchor="se",
                                  fill=MUTED, font=("Courier", 8))

        mid = ch // 2
        self._canvas.create_oval(sx - 11, mid - 11, sx + 11, mid + 11,
                                  fill=ACCENT, outline="white", width=1.5)
        self._canvas.create_line(sx - 5, mid, sx + 5, mid, fill="white", width=2)
        self._canvas.create_line(sx, mid - 5, sx, mid + 5, fill="white", width=2)

        if self._region is not None:
            x0, y0, x1, y1 = self._region
            cx0, cy0 = self._image_to_canvas_coords(x0, y0)
            cx1, cy1 = self._image_to_canvas_coords(x1, y1)
            self._canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                           outline=REGION_COLOR, width=2, dash=(6, 3))
            self._canvas.create_text(cx0 + 4, cy0 + 4, text="AUTO REGION", anchor="nw",
                                      fill=REGION_COLOR, font=("Courier", 7, "bold"))

        if self._selecting_region and self._region_drag_start and self._region_drag_current:
            rx0, ry0 = self._region_drag_start
            rx1, ry1 = self._region_drag_current
            self._canvas.create_rectangle(rx0, ry0, rx1, ry1,
                                           outline=REGION_COLOR, width=2, dash=(4, 2))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_canvas_configure(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._canvas_w = event.width
        self._canvas_h = event.height
        if self._resize_after is not None:
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(60, self._on_resize_done)

    def _on_resize_done(self) -> None:
        self._resize_after = None
        self._clamp_pan()
        self._redraw()

    def _on_press(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._selecting_region and self._source is not None:
            self._region_drag_start = (event.x, event.y)
            self._region_drag_current = (event.x, event.y)
            return

        sx = int(self._split_frac * self._canvas_w)
        if abs(event.x - sx) <= 12:
            self._dragging_split = True
            self._canvas.config(cursor="sb_h_double_arrow")
        else:
            self._panning = True
            self._pan_start = (event.x, event.y)
            self._pan_origin = (self._pan_x, self._pan_y)
            self._canvas.config(cursor="fleur")

    def _on_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._selecting_region and self._region_drag_start is not None:
            self._region_drag_current = (event.x, event.y)
            self._redraw()
            return
        if self._dragging_split:
            self._split_frac = max(0.0, min(1.0, event.x / max(1, self._canvas_w)))
            self._redraw()
        elif self._panning and self._source is not None:
            dx = (event.x - self._pan_start[0]) / self._zoom
            dy = (event.y - self._pan_start[1]) / self._zoom
            self._pan_x = self._pan_origin[0] - dx
            self._pan_y = self._pan_origin[1] - dy
            self._clamp_pan()
            self._redraw()

    def _on_release(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._selecting_region and self._region_drag_start is not None:
            ix0, iy0 = self._canvas_to_image_coords(*self._region_drag_start)
            ix1, iy1 = self._canvas_to_image_coords(event.x, event.y)
            self._region = (min(ix0, ix1), min(iy0, iy1), max(ix0, ix1), max(iy0, iy1))
            self._region_drag_start = None
            self._region_drag_current = None
            self._selecting_region = False
            self._region_btn.config(bg=BORDER, fg=TEXT)
            self._clear_region_btn.config(state="normal")
            self._canvas.config(cursor="sb_h_double_arrow")
            rw = self._region[2] - self._region[0]
            rh = self._region[3] - self._region[1]
            self._status_var.set(f"Region set: {rw}×{rh} px — press AUTO DETECT to use it.")
            self._redraw()
            return

        self._dragging_split = False
        self._panning = False
        self._canvas.config(cursor="sb_h_double_arrow")

    def _on_pan_start(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._panning = True
        self._pan_start = (event.x, event.y)
        self._pan_origin = (self._pan_x, self._pan_y)
        self._canvas.config(cursor="fleur")

    def _on_pan_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if not self._panning or self._source is None:
            return
        dx = (event.x - self._pan_start[0]) / self._zoom
        dy = (event.y - self._pan_start[1]) / self._zoom
        self._pan_x = self._pan_origin[0] - dx
        self._pan_y = self._pan_origin[1] - dy
        self._clamp_pan()
        self._redraw()

    def _on_pan_end(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._panning = False
        self._canvas.config(cursor="sb_h_double_arrow")

    def _on_scroll(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._source is None:
            return
        factor = ZOOM_STEP if (event.num == 4 or event.delta > 0) else 1.0 / ZOOM_STEP
        self._zoom_around(event.x, event.y, factor)
        self._redraw()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def process_image(
    input_path: Path,
    output_path: Path,
    red_scale: float,
    blue_scale: float,
    red_shift: tuple[float, float],
    blue_shift: tuple[float, float],
    auto: bool,
) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Image must be a colour (BGR) image.")

    if auto:
        detected_red, detected_blue, detected_red_shift, detected_blue_shift, _ = auto_detect_ca(image)
        print(f"  Auto-detected red  scale={detected_red:.6f}  shift=({detected_red_shift[0]:+.4f},{detected_red_shift[1]:+.4f})")
        print(f"  Auto-detected blue scale={detected_blue:.6f}  shift=({detected_blue_shift[0]:+.4f},{detected_blue_shift[1]:+.4f})")
        red_scale = detected_red
        blue_scale = detected_blue
        red_shift = detected_red_shift
        blue_shift = detected_blue_shift

    corrected = correct_chromatic_aberration(
        image,
        red_scale=red_scale,
        blue_scale=blue_scale,
        red_shift=red_shift,
        blue_shift=blue_shift,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), corrected)
    print(f"  Saved: {output_path}")
    print(
        f"  Args: --red-scale {red_scale:.6f} --blue-scale {blue_scale:.6f}"
        f" --red-shift {red_shift[0]:+.4f} {red_shift[1]:+.4f}"
        f" --blue-shift {blue_shift[0]:+.4f} {blue_shift[1]:+.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Correct chromatic aberration in an image or folder of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, nargs="?", default=None,
                   help="Input image or folder (omit to launch GUI)")
    p.add_argument("--no-gui", action="store_true",
                   help="Run as CLI only; implied when input is a folder")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output file or folder")
    p.add_argument("--red-scale", type=float, default=1.0)
    p.add_argument("--blue-scale", type=float, default=1.0)
    p.add_argument("--red-shift", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"))
    p.add_argument("--blue-shift", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"))
    p.add_argument("--auto", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    red_shift = (args.red_shift[0], args.red_shift[1])
    blue_shift = (args.blue_shift[0], args.blue_shift[1])

    if args.input is not None and args.input.is_dir():
        images = sorted(
            p for p in args.input.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            print(f"No supported images found in {args.input}")
            raise SystemExit(1)

        out_dir = args.output or (args.input.parent / (args.input.name + "_ca_corrected"))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {len(images)} image(s): {args.input} -> {out_dir}")

        errors: list[str] = []
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {img_path.name}")
            try:
                process_image(
                    input_path=img_path,
                    output_path=out_dir / img_path.name,
                    red_scale=args.red_scale,
                    blue_scale=args.blue_scale,
                    red_shift=red_shift,
                    blue_shift=blue_shift,
                    auto=args.auto,
                )
            except (ValueError, OSError) as e:
                print(f"  Error: {e}")
                errors.append(img_path.name)

        print(f"Done. {len(images) - len(errors)}/{len(images)} succeeded.")
        if errors:
            print(f"Failed: {', '.join(errors)}")
        return

    if args.no_gui:
        if args.input is None:
            print("Error: --no-gui requires an input file or folder.")
            raise SystemExit(1)
        output = args.output or args.input.with_stem(args.input.stem + "_ca_corrected")
        print(f"[1/1] {args.input.name}")
        try:
            process_image(
                input_path=args.input,
                output_path=output,
                red_scale=args.red_scale,
                blue_scale=args.blue_scale,
                red_shift=red_shift,
                blue_shift=blue_shift,
                auto=args.auto,
            )
        except (ValueError, OSError) as e:
            print(f"Error: {e}")
            raise SystemExit(1)
        return

    CAApp(initial_path=args.input).mainloop()


if __name__ == "__main__":
    main()