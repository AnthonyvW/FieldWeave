from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font
from PIL import Image, ImageTk

# ── constants ────────────────────────────────────────────────────────────────
POINT_RADIUS = 6
POINT_COLOR = "#ff3c3c"
CROSSHAIR = "#ff3c3c"
ZOOM_STEP = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 40.0


# ── image panel ──────────────────────────────────────────────────────────────
class ImagePanel(tk.Frame):
    """A labelled canvas that holds one image and records one click point.

    Viewport transform
    ------------------
    A canvas pixel (cx, cy) maps to an original-image pixel (ox, oy) via:

        ox = (cx - pan_x) / zoom
        oy = (cy - pan_y) / zoom

    where zoom  = pixels-on-canvas per original-pixel,
          pan_x/pan_y = canvas coords of the top-left corner of the image.
    """

    def __init__(self, parent: tk.Widget, label: str, on_point_set: callable) -> None:
        super().__init__(parent, bg="#1a1a2e")
        self.on_point_set = on_point_set
        self.original_size: tuple[int, int] | None = None
        self.point_original: tuple[float, float] | None = None  # original px

        self._photo: ImageTk.PhotoImage | None = None
        self._canvas_image_id: int | None = None
        self._point_items: list[int] = []

        # viewport state
        self._zoom: float = 1.0     # canvas-px per original-px
        self._pan_x: float = 0.0   # canvas x of image top-left
        self._pan_y: float = 0.0   # canvas y of image top-left
        self._fit_zoom: float = 1.0  # zoom at initial fit-to-canvas
        self._drag_start: tuple[int, int] | None = None
        self._drag_pan_start: tuple[float, float] = (0.0, 0.0)

        # header
        hdr = tk.Frame(self, bg="#16213e", pady=6)
        hdr.pack(fill="x")
        mono = font.Font(family="Courier", size=10, weight="bold")
        tk.Label(hdr, text=label, fg="#e0e0ff", bg="#16213e",
                 font=mono).pack(side="left", padx=12)
        self.coord_var = tk.StringVar(value="no point selected")
        tk.Label(hdr, textvariable=self.coord_var, fg="#7878aa", bg="#16213e",
                 font=mono).pack(side="right", padx=12)

        # zoom label
        self._zoom_var = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._zoom_var, fg="#4466aa", bg="#16213e",
                 font=mono).pack(side="right", padx=8)

        # canvas
        self.canvas = tk.Canvas(self, bg="#0d0d1a", cursor="crosshair",
                                 highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_resize)
        # scroll-to-zoom
        self.canvas.bind("<MouseWheel>", self._on_scroll)   # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_scroll)     # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_scroll)     # Linux scroll down
        # middle-click or right-click drag to pan
        for btn, motion, release in (
            ("<ButtonPress-2>", "<B2-Motion>", "<ButtonRelease-2>"),
            ("<ButtonPress-3>", "<B3-Motion>", "<ButtonRelease-3>"),
        ):
            self.canvas.bind(btn, self._on_pan_start)
            self.canvas.bind(motion, self._on_pan_move)
            self.canvas.bind(release, self._on_pan_end)

        self._original_image: Image.Image | None = None

    # ── public ───────────────────────────────────────────────────────────────
    def load(self, path: Path) -> None:
        img = Image.open(path).convert("RGB")
        self.original_size = img.size
        self._original_image = img
        self.point_original = None
        self.coord_var.set("scroll to zoom  |  right-drag to pan  |  click to mark")
        self._reset_viewport()

    def clear_point(self) -> None:
        for item in self._point_items:
            self.canvas.delete(item)
        self._point_items = []
        self.point_original = None
        if self._original_image is not None:
            self.coord_var.set("scroll to zoom  |  right-drag to pan  |  click to mark")
        else:
            self.coord_var.set("no point selected")

    # ── viewport helpers ─────────────────────────────────────────────────────
    def _reset_viewport(self) -> None:
        """Fit image to canvas and centre it."""
        if self._original_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        iw, ih = self._original_image.size
        self._fit_zoom = min(cw / iw, ch / ih)
        self._zoom = self._fit_zoom
        self._pan_x = (cw - iw * self._zoom) / 2
        self._pan_y = (ch - ih * self._zoom) / 2
        self._render()

    def _canvas_to_orig(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self._pan_x) / self._zoom, (cy - self._pan_y) / self._zoom

    def _orig_to_canvas(self, ox: float, oy: float) -> tuple[float, float]:
        return ox * self._zoom + self._pan_x, oy * self._zoom + self._pan_y

    # ── rendering ────────────────────────────────────────────────────────────
    def _render(self) -> None:
        if self._original_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        iw, ih = self._original_image.size
        disp_w = max(1, int(round(iw * self._zoom)))
        disp_h = max(1, int(round(ih * self._zoom)))

        disp = self._original_image.resize((disp_w, disp_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(disp)

        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
        self._canvas_image_id = self.canvas.create_image(
            int(self._pan_x), int(self._pan_y), anchor="nw", image=self._photo
        )

        # keep point marker in sync
        for item in self._point_items:
            self.canvas.delete(item)
        self._point_items = []
        if self.point_original is not None:
            cx, cy = self._orig_to_canvas(*self.point_original)
            self._draw_crosshair(cx, cy)

        rel = self._zoom / self._fit_zoom
        self._zoom_var.set(f"{rel:.1f}x")

    def _on_resize(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._reset_viewport()

    # ── scroll-to-zoom ───────────────────────────────────────────────────────
    def _on_scroll(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        # determine direction
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        # zoom around the cursor position
        mx, my = event.x, event.y
        self._pan_x = mx - (mx - self._pan_x) * (new_zoom / self._zoom)
        self._pan_y = my - (my - self._pan_y) * (new_zoom / self._zoom)
        self._zoom = new_zoom
        self._render()

    # ── pan ──────────────────────────────────────────────────────────────────
    def _on_pan_start(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._drag_start = (event.x, event.y)
        self._drag_pan_start = (self._pan_x, self._pan_y)
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._pan_x = self._drag_pan_start[0] + dx
        self._pan_y = self._drag_pan_start[1] + dy
        self._render()
        self.canvas.config(cursor="fleur")

    def _on_pan_end(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._drag_start = None
        self.canvas.config(cursor="crosshair")

    # ── click to mark ────────────────────────────────────────────────────────
    def _on_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._original_image is None or self.original_size is None:
            return
        iw, ih = self.original_size
        ox, oy = self._canvas_to_orig(event.x, event.y)
        # clamp to image
        ox = max(0.0, min(float(iw - 1), ox))
        oy = max(0.0, min(float(ih - 1), oy))
        self.point_original = (ox, oy)

        for item in self._point_items:
            self.canvas.delete(item)
        self._point_items = []
        cx, cy = self._orig_to_canvas(ox, oy)
        self._draw_crosshair(cx, cy)

        self.coord_var.set(f"x={ox:.1f}  y={oy:.1f}")
        self.on_point_set()

    def _draw_crosshair(self, cx: float, cy: float) -> None:
        r = POINT_RADIUS
        arm = 14
        self._point_items = [
            self.canvas.create_line(cx - arm, cy, cx + arm, cy,
                                    fill=CROSSHAIR, width=1),
            self.canvas.create_line(cx, cy - arm, cx, cy + arm,
                                    fill=CROSSHAIR, width=1),
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline=POINT_COLOR, width=2),
        ]


# ── main app ─────────────────────────────────────────────────────────────────
class OverlapApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tile Overlap Calculator")
        self.configure(bg="#1a1a2e")
        self.minsize(900, 560)

        self._paths: list[Path | None] = [None, None]
        self._build_ui()

    def _build_ui(self) -> None:
        mono = font.Font(family="Courier", size=10)
        bold_mono = font.Font(family="Courier", size=11, weight="bold")

        # ── top toolbar ──────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg="#0d0d1a", pady=8)
        toolbar.pack(fill="x")

        tk.Label(toolbar, text="TILE OVERLAP CALCULATOR",
                 fg="#c8c8ff", bg="#0d0d1a",
                 font=font.Font(family="Courier", size=12, weight="bold")
                 ).pack(side="left", padx=16)

        btn_style = dict(font=mono, bg="#2a2a4a", fg="#c8c8ff",
                         activebackground="#3a3a6a", activeforeground="#ffffff",
                         relief="flat", padx=10, pady=4, cursor="hand2",
                         bd=0)

        tk.Button(toolbar, text="[ clear all ]",
                  command=self._clear_all, **btn_style).pack(side="right", padx=8)

        # ── image panels ─────────────────────────────────────────────────────
        panels_frame = tk.Frame(self, bg="#1a1a2e")
        panels_frame.pack(fill="both", expand=True, padx=8, pady=4)
        panels_frame.columnconfigure(0, weight=1)
        panels_frame.columnconfigure(1, weight=1)
        panels_frame.rowconfigure(1, weight=1)

        for i, label in enumerate(["IMAGE A  (reference)", "IMAGE B  (overlapping tile)"]):
            hdr = tk.Frame(panels_frame, bg="#1a1a2e")
            hdr.grid(row=0, column=i, sticky="ew", padx=4, pady=(4, 0))

            path_var = tk.StringVar(value="no file selected")
            setattr(self, f"_path_var_{i}", path_var)

            tk.Button(
                hdr, text=f"[ open image {chr(65+i)} ]",
                command=lambda idx=i: self._open_image(idx),
                **btn_style
            ).pack(side="left")
            tk.Label(hdr, textvariable=path_var, fg="#555577",
                     bg="#1a1a2e", font=mono).pack(side="left", padx=8)

        self._panels = [
            ImagePanel(panels_frame, "A  — click the same feature on both images",
                       self._on_point_set),
            ImagePanel(panels_frame, "B  — click the same feature on both images",
                       self._on_point_set),
        ]
        self._panels[0].grid(row=1, column=0, sticky="nsew", padx=(4, 2), pady=4)
        self._panels[1].grid(row=1, column=1, sticky="nsew", padx=(2, 4), pady=4)

        # ── result bar ───────────────────────────────────────────────────────
        result_bar = tk.Frame(self, bg="#0d0d1a", pady=10)
        result_bar.pack(fill="x", side="bottom")

        self._result_var = tk.StringVar(
            value="open two images and click the same feature on each to calculate overlap"
        )
        tk.Label(result_bar, textvariable=self._result_var,
                 fg="#a0ffa0", bg="#0d0d1a", font=bold_mono,
                 justify="center").pack()

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _open_image(self, idx: int) -> None:
        path = filedialog.askopenfilename(
            title=f"Select Image {chr(65 + idx)}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        self._paths[idx] = p
        getattr(self, f"_path_var_{idx}").set(p.name)
        self._panels[idx].load(p)
        self._result_var.set("click the same feature on both images")

    def _clear_all(self) -> None:
        for panel in self._panels:
            panel.clear_point()
        self._result_var.set("open two images and click the same feature on each to calculate overlap")

    def _on_point_set(self) -> None:
        pa = self._panels[0].point_original
        pb = self._panels[1].point_original

        if pa is None or pb is None:
            return
        if self._panels[0].original_size is None or self._panels[1].original_size is None:
            return

        wa, ha = self._panels[0].original_size
        wb, hb = self._panels[1].original_size

        # The same physical feature is at pixel pa in image A and pb in image B.
        # Overlap is how much of each tile is shared with its neighbour.
        #
        # For a horizontal neighbour (B is to the right of A):
        #   feature is (wa - pa.x) pixels from the right edge of A
        #   feature is pb.x pixels from the left edge of B
        #   horizontal overlap = (wa - pa.x) + pb.x  ... as px in A space
        #
        # For a vertical neighbour (B is below A):
        #   vertical overlap = (ha - pa.y) + pb.y
        #
        # We report overlap as percentage of each tile dimension.

        horiz_px_a = (wa - pa[0]) + pb[0]
        vert_px_a  = (ha - pa[1]) + pb[1]

        horiz_pct_a = horiz_px_a / wa * 100
        vert_pct_a  = vert_px_a  / ha * 100
        horiz_pct_b = horiz_px_a / wb * 100
        vert_pct_b  = vert_px_a  / hb * 100

        lines = [
            f"horizontal overlap:  {horiz_px_a:.1f} px  "
            f"({horiz_pct_a:.1f}% of A,  {horiz_pct_b:.1f}% of B)",
            f"vertical overlap:    {vert_px_a:.1f} px  "
            f"({vert_pct_a:.1f}% of A,  {vert_pct_b:.1f}% of B)",
        ]
        self._result_var.set("    |    ".join(lines))


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = OverlapApp()
    app.mainloop()