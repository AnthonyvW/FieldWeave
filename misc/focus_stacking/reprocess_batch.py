from __future__ import annotations

import re
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

Image.MAX_IMAGE_PIXELS = None

THUMB_SOURCE = (100, 75)
THUMB_STACKED = (220, 165)
CELL_H = THUMB_STACKED[1] + 24  # thumb height + label + padding
CELL_PAD = 2
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SOURCE_COLS = 8
LIST_COLS = 2
VIRT_BUFFER_ROWS = 3  # extra rows to render above/below viewport

BG_DARK = "#111111"
BG_MID = "#1a1a1a"
BG_SEL = "#2a4a6a"
BG_PREVIEW = "#0d0d0d"
BG_LIST_SEL = "#1a4a8a"


def parse_stacked_name(filename: str) -> tuple[int, int] | None:
    stem = Path(filename).stem
    m = re.match(r"^(\d+)_(\d+)$", stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def find_source_folder(root: Path, filename: str) -> Path | None:
    parsed = parse_stacked_name(filename)
    if parsed is None:
        return None
    y_trunc, x_trunc = parsed
    candidate = root / f"x{x_trunc}0000_y{y_trunc}0000"
    return candidate if candidate.is_dir() else None


def set_cell_bg(cell: tk.Frame, color: str) -> None:
    cell.configure(bg=color)
    for child in cell.winfo_children():
        child.configure(bg=color)


def make_scrollable_canvas(parent: tk.Widget, bg: str) -> tuple[tk.Frame, tk.Canvas, tk.Frame]:
    frame = tk.Frame(parent, bg=bg)
    canvas = tk.Canvas(frame, bg=bg, highlightthickness=0)
    vbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(fill=tk.BOTH, expand=True)

    inner = tk.Frame(canvas, bg=bg)
    win = canvas.create_window((0, 0), window=inner, anchor="nw")

    def on_inner_configure(_e: tk.Event) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_configure(e: tk.Event) -> None:
        canvas.itemconfig(win, width=e.width)

    def on_mousewheel(e: tk.Event) -> None:
        if e.num == 4:
            canvas.yview_scroll(-1, "units")
        elif e.num == 5:
            canvas.yview_scroll(1, "units")
        else:
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    inner.bind("<Configure>", on_inner_configure)
    canvas.bind("<Configure>", on_canvas_configure)
    canvas.bind("<MouseWheel>", on_mousewheel)
    canvas.bind("<Button-4>", on_mousewheel)
    canvas.bind("<Button-5>", on_mousewheel)

    return frame, canvas, inner


class ResizableImageCanvas(tk.Canvas):
    """Canvas that fits an image to its current size, centred, on resize."""

    def __init__(self, parent: tk.Widget, bg: str = BG_PREVIEW, **kwargs: object) -> None:
        super().__init__(parent, bg=bg, highlightthickness=0, **kwargs)
        self._pil_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._placeholder_text = ""
        self.bind("<Configure>", self._on_resize)

    def set_image(self, path: Path) -> None:
        self._pil_image = Image.open(path)
        self._placeholder_text = ""
        self._redraw()

    def set_placeholder(self, text: str) -> None:
        self._pil_image = None
        self._photo = None
        self._placeholder_text = text
        self._redraw()

    def _on_resize(self, _e: tk.Event) -> None:
        self._redraw()

    def _redraw(self) -> None:
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        if self._pil_image is None:
            self.create_text(w // 2, h // 2, text=self._placeholder_text,
                             fill="#444444", font=("Courier", 10))
            return
        img = self._pil_image.copy()
        img.thumbnail((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.create_image(w // 2, h // 2, anchor="center", image=self._photo)


class VirtualThumbGrid(tk.Frame):
    """
    Virtually-scrolled thumbnail grid. Tracks scroll position as a raw pixel
    offset so the viewport frame never exceeds tkinter's 32 767 px limit.
    The scrollbar is driven manually via yscrollcommand fractions.
    """

    def __init__(self, parent: tk.Widget, cols: int, cell_h: int,
                 on_select: object, bg: str = BG_DARK, **kwargs: object) -> None:
        super().__init__(parent, bg=bg, **kwargs)
        self._cols = cols
        self._cell_h = cell_h
        self._on_select = on_select
        self._bg = bg
        self._entries: list[dict] = []
        self._rendered: dict[int, tk.Frame] = {}
        self._thumbs: dict[int, ImageTk.PhotoImage] = {}
        self._selected_idx: int | None = None
        self._scroll_px: int = 0   # current top-of-viewport in content pixels
        self._build()

    def _build(self) -> None:
        self._vbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._on_scroll_cmd)
        self._vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._viewport = tk.Frame(self, bg=self._bg)
        self._viewport.pack(fill=tk.BOTH, expand=True)
        self._viewport.bind("<Configure>", self._on_viewport_configure)

        for w in (self._viewport,):
            w.bind("<MouseWheel>", self._on_mousewheel)
            w.bind("<Button-4>", self._on_mousewheel)
            w.bind("<Button-5>", self._on_mousewheel)

    def _on_scroll_cmd(self, action: str, value: str, unit: str = "") -> None:
        total_h = self._total_height()
        view_h = self._viewport.winfo_height()
        max_scroll = max(0, total_h - view_h)
        if action == "moveto":
            self._scroll_px = int(float(value) * total_h)
        elif action == "scroll":
            amount = int(value)
            if unit == "units":
                self._scroll_px += amount * (self._cell_h + CELL_PAD)
            else:
                self._scroll_px += amount * view_h
        self._scroll_px = max(0, min(self._scroll_px, max_scroll))
        self._apply_scroll()

    def _on_mousewheel(self, e: tk.Event) -> None:
        if e.num == 4:
            delta = -3
        elif e.num == 5:
            delta = 3
        else:
            delta = -int(e.delta / 40)
        total_h = self._total_height()
        view_h = self._viewport.winfo_height()
        max_scroll = max(0, total_h - view_h)
        self._scroll_px = max(0, min(self._scroll_px + delta * (self._cell_h // 4), max_scroll))
        self._apply_scroll()

    def _on_viewport_configure(self, _e: tk.Event) -> None:
        self._apply_scroll()

    def _apply_scroll(self) -> None:
        self._update_scrollbar()
        self._update_visible()

    def _update_scrollbar(self) -> None:
        total_h = self._total_height()
        view_h = self._viewport.winfo_height()
        if total_h == 0:
            self._vbar.set(0.0, 1.0)
            return
        lo = self._scroll_px / total_h
        hi = min(1.0, (self._scroll_px + view_h) / total_h)
        self._vbar.set(lo, hi)

    def _total_rows(self) -> int:
        return (len(self._entries) + self._cols - 1) // self._cols

    def _total_height(self) -> int:
        return self._total_rows() * (self._cell_h + CELL_PAD) + CELL_PAD

    def _cell_w(self) -> int:
        w = self._viewport.winfo_width()
        return max(1, (w - CELL_PAD * (self._cols + 1)) // self._cols)

    def load(self, entries: list[dict]) -> None:
        for frame in self._rendered.values():
            frame.destroy()
        self._rendered.clear()
        self._thumbs.clear()
        self._selected_idx = None
        self._scroll_px = 0
        self._entries = entries
        self._apply_scroll()

    def _visible_row_range(self) -> tuple[int, int]:
        view_h = self._viewport.winfo_height()
        if view_h == 0 or not self._entries:
            return 0, 0
        row_stride = self._cell_h + CELL_PAD
        first_row = max(0, self._scroll_px // row_stride - VIRT_BUFFER_ROWS)
        last_row = min(self._total_rows() - 1,
                       (self._scroll_px + view_h) // row_stride + VIRT_BUFFER_ROWS)
        return first_row, last_row

    def _update_visible(self) -> None:
        if not self._entries:
            return
        first_row, last_row = self._visible_row_range()
        needed = set(range(first_row, last_row + 1))

        for row_idx in list(self._rendered):
            if row_idx not in needed:
                self._rendered[row_idx].destroy()
                del self._rendered[row_idx]

        cw = self._cell_w()
        row_stride = self._cell_h + CELL_PAD
        view_w = self._viewport.winfo_width()

        for row_idx in needed:
            # y relative to viewport top
            y = CELL_PAD + row_idx * row_stride - self._scroll_px
            if row_idx in self._rendered:
                self._rendered[row_idx].place(x=0, y=y, width=view_w, height=self._cell_h)
                continue
            row_frame = tk.Frame(self._viewport, bg=self._bg)
            row_frame.place(x=0, y=y, width=view_w, height=self._cell_h)
            self._rendered[row_idx] = row_frame

            for col in range(self._cols):
                entry_idx = row_idx * self._cols + col
                if entry_idx >= len(self._entries):
                    break
                self._build_cell(row_frame, entry_idx, col, cw)

    def _build_cell(self, row_frame: tk.Frame, entry_idx: int, col: int, cw: int) -> None:
        entry = self._entries[entry_idx]
        is_sel = entry_idx == self._selected_idx
        bg = BG_LIST_SEL if is_sel else self._bg

        x = CELL_PAD + col * (cw + CELL_PAD)
        cell = tk.Frame(row_frame, bg=bg, padx=3, pady=3)
        cell.place(x=x, y=0, width=cw, height=self._cell_h)

        # Load thumb if not cached
        if entry_idx not in self._thumbs:
            try:
                img = Image.open(entry["path"])
                img.thumbnail(THUMB_STACKED, Image.LANCZOS)
                img = img.convert("RGB")
                self._thumbs[entry_idx] = ImageTk.PhotoImage(img)
            except Exception:
                self._thumbs[entry_idx] = None  # type: ignore[assignment]

        thumb = self._thumbs.get(entry_idx)
        if thumb:
            img_lbl = tk.Label(cell, image=thumb, bg=bg, cursor="hand2", bd=0)
            img_lbl.image = thumb
            img_lbl.pack()
        else:
            img_lbl = tk.Label(cell, text="[err]", font=("Courier", 9),
                               bg=bg, fg="#553333", cursor="hand2", width=14, height=6)
            img_lbl.pack()

        name_lbl = tk.Label(cell, text=entry["label_text"], font=("Courier", 8),
                            bg=bg, fg=entry["label_fg"], wraplength=cw - 10)
        name_lbl.pack()

        for w in (cell, img_lbl, name_lbl):
            w.bind("<Button-1>", lambda e, n=entry_idx: self._select(n))
            w.bind("<MouseWheel>", self._on_mousewheel)
            w.bind("<Button-4>", self._on_mousewheel)
            w.bind("<Button-5>", self._on_mousewheel)

    def _select(self, idx: int) -> None:
        prev = self._selected_idx
        self._selected_idx = idx

        # Repaint affected rows
        for entry_idx in (prev, idx):
            if entry_idx is None:
                continue
            row_idx = entry_idx // self._cols
            if row_idx in self._rendered:
                self._rendered[row_idx].destroy()
                del self._rendered[row_idx]

        self._update_visible()
        self._on_select(idx, self._entries[idx])

    def reload_thumb(self, path: Path) -> None:
        for i, entry in enumerate(self._entries):
            if entry["path"] == path:
                try:
                    img = Image.open(path)
                    img.thumbnail(THUMB_STACKED, Image.LANCZOS)
                    img = img.convert("RGB")
                    self._thumbs[i] = ImageTk.PhotoImage(img)
                except Exception:
                    return
                row_idx = i // self._cols
                if row_idx in self._rendered:
                    self._rendered[row_idx].destroy()
                    del self._rendered[row_idx]
                self._update_visible()
                return


class SourceImagePanel(tk.Frame):
    """Right panel: source image grid + side-by-side preview area."""

    def __init__(self, parent: tk.Widget, **kwargs: object) -> None:
        super().__init__(parent, **kwargs)
        self._thumbs: list[ImageTk.PhotoImage | None] = []
        self._source_paths: list[Path] = []
        self._stacked_path: Path | None = None
        self._source_folder: Path | None = None
        self._root_dir: Path | None = None
        self._cell_widgets: list[tk.Frame] = []
        self._selected: set[int] = set()
        self._anchor: int | None = None
        self._show_stacked = tk.BooleanVar(value=True)
        self._on_thumb_reloaded: object | None = None
        self._build()

    def set_thumb_reload_callback(self, cb: object) -> None:
        self._on_thumb_reloaded = cb

    def _build(self) -> None:
        top = tk.Frame(self, bg=BG_MID)
        top.pack(fill=tk.X, pady=(0, 2))

        self._title = tk.Label(top, text="Source images", font=("Courier", 11, "bold"),
                               bg=BG_MID, fg="#888888", anchor="w")
        self._title.pack(side=tk.LEFT, padx=8, pady=6)

        self._restack_btn = tk.Button(
            top, text="Re-run focusweave", font=("Courier", 10),
            bg="#2a3a2a", fg="#88cc88", relief=tk.FLAT, padx=10, pady=4,
            cursor="hand2", command=self._rerun_focusweave, state=tk.DISABLED)
        self._restack_btn.pack(side=tk.RIGHT, padx=8, pady=4)

        self._delete_btn = tk.Button(
            top, text="Delete selected", font=("Courier", 10),
            bg="#3a2a2a", fg="#cc8888", relief=tk.FLAT, padx=10, pady=4,
            cursor="hand2", command=self._delete_selected, state=tk.DISABLED)
        self._delete_btn.pack(side=tk.RIGHT, padx=4, pady=4)

        self._toggle_btn = tk.Checkbutton(
            top, text="Show stacked", font=("Courier", 10),
            bg=BG_MID, fg="#888888", selectcolor="#2a2a2a",
            activebackground=BG_MID, activeforeground="#aaaaaa",
            variable=self._show_stacked, command=self._on_toggle_stacked)
        self._toggle_btn.pack(side=tk.RIGHT, padx=8, pady=4)

        vpane = tk.PanedWindow(self, orient=tk.VERTICAL, bg=BG_MID, sashwidth=4)
        vpane.pack(fill=tk.BOTH, expand=True)

        scroll_frame, self._canvas, self._thumb_inner = make_scrollable_canvas(vpane, BG_DARK)
        vpane.add(scroll_frame, minsize=120)

        self._preview_pane = tk.PanedWindow(vpane, orient=tk.HORIZONTAL, bg=BG_PREVIEW, sashwidth=4)
        vpane.add(self._preview_pane, minsize=200)

        source_frame = tk.Frame(self._preview_pane, bg=BG_PREVIEW)
        self._preview_pane.add(source_frame, stretch="always")
        tk.Label(source_frame, text="source", font=("Courier", 8),
                 bg=BG_PREVIEW, fg="#333333").pack(side=tk.BOTTOM, pady=2)
        self._source_canvas = ResizableImageCanvas(source_frame)
        self._source_canvas.pack(fill=tk.BOTH, expand=True)
        self._source_canvas.set_placeholder("Select a source image to preview")

        self._stacked_frame = tk.Frame(self._preview_pane, bg=BG_PREVIEW)
        self._preview_pane.add(self._stacked_frame, stretch="always")
        tk.Label(self._stacked_frame, text="focus stacked", font=("Courier", 8),
                 bg=BG_PREVIEW, fg="#333333").pack(side=tk.BOTTOM, pady=2)
        self._stacked_canvas = ResizableImageCanvas(self._stacked_frame)
        self._stacked_canvas.pack(fill=tk.BOTH, expand=True)

        self._status = tk.Label(self, text="", font=("Courier", 9),
                                bg=BG_MID, fg="#555555", anchor="w")
        self._status.pack(fill=tk.X, padx=8, pady=(2, 4))

    def _on_toggle_stacked(self) -> None:
        if self._show_stacked.get():
            self._preview_pane.add(self._stacked_frame, stretch="always")
        else:
            self._preview_pane.forget(self._stacked_frame)

    def load_for(self, stacked_path: Path, source_folder: Path, root_dir: Path) -> None:
        self._stacked_path = stacked_path
        self._source_folder = source_folder
        self._root_dir = root_dir
        self._selected.clear()
        self._anchor = None
        self._thumbs.clear()
        self._source_paths.clear()
        self._cell_widgets.clear()

        for w in self._thumb_inner.winfo_children():
            w.destroy()

        self._source_canvas.set_placeholder("Loading...")
        self._stacked_canvas.set_placeholder("")
        self._status.configure(text="")
        self._delete_btn.configure(state=tk.DISABLED)

        paths = sorted(p for p in source_folder.iterdir()
                       if p.suffix.lower() in SUPPORTED_EXTS)
        self._source_paths = paths
        self._title.configure(text=f"{stacked_path.name}  —  {len(paths)} source images")

        for i, path in enumerate(paths):
            row, col = divmod(i, SOURCE_COLS)
            cell = tk.Frame(self._thumb_inner, bg=BG_DARK, padx=2, pady=2)
            cell.grid(row=row, column=col, sticky="nsew")
            self._thumb_inner.columnconfigure(col, weight=1)
            self._cell_widgets.append(cell)

            try:
                img = Image.open(path)
                img.thumbnail(THUMB_SOURCE, Image.LANCZOS)
                img = img.convert("RGB")
                thumb = ImageTk.PhotoImage(img)
            except Exception:
                thumb = None
            self._thumbs.append(thumb)

            if thumb:
                img_lbl = tk.Label(cell, image=thumb, bg=BG_DARK, cursor="hand2", bd=0)
                img_lbl.image = thumb
                img_lbl.pack()
            else:
                img_lbl = tk.Label(cell, text="[err]", font=("Courier", 8),
                                   bg=BG_DARK, fg="#553333", cursor="hand2", width=8, height=3)
                img_lbl.pack()

            idx = i
            for widget in (cell, img_lbl):
                widget.bind("<Button-1>", lambda e, n=idx: self._on_click(e, n))
                widget.bind("<Shift-Button-1>", lambda e, n=idx: self._on_shift_click(n))

        self._source_canvas.set_placeholder("Select a source image to preview")
        self._restack_btn.configure(state=tk.NORMAL)
        self._status.configure(text=str(source_folder))
        self._load_stacked_preview()

    def _load_stacked_preview(self) -> None:
        if self._stacked_path is None:
            return
        try:
            self._stacked_canvas.set_image(self._stacked_path)
        except Exception as exc:
            self._stacked_canvas.set_placeholder(f"Cannot load: {exc}")

    def _on_click(self, _event: tk.Event, idx: int) -> None:
        self._anchor = idx
        self._set_selection({idx})
        self._show_preview(idx)

    def _on_shift_click(self, idx: int) -> None:
        if self._anchor is None:
            self._anchor = idx
            self._set_selection({idx})
            self._show_preview(idx)
            return
        lo, hi = sorted((self._anchor, idx))
        self._set_selection(set(range(lo, hi + 1)))
        self._show_preview(idx)

    def _set_selection(self, indices: set[int]) -> None:
        for i in self._selected - indices:
            if i < len(self._cell_widgets):
                set_cell_bg(self._cell_widgets[i], BG_DARK)
        for i in indices - self._selected:
            if i < len(self._cell_widgets):
                set_cell_bg(self._cell_widgets[i], BG_SEL)
        self._selected = indices
        self._delete_btn.configure(state=tk.NORMAL if indices else tk.DISABLED)
        if len(indices) > 1:
            self._status.configure(text=f"{len(indices)} images selected")

    def _show_preview(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._source_paths):
            return
        path = self._source_paths[idx]
        if len(self._selected) == 1:
            self._status.configure(text=str(path))
        try:
            self._source_canvas.set_image(path)
        except Exception as exc:
            self._source_canvas.set_placeholder(f"Cannot load: {exc}")

    def _delete_selected(self) -> None:
        if not self._selected:
            return
        indices = sorted(self._selected)
        names = [self._source_paths[i].name for i in indices if i < len(self._source_paths)]
        if len(names) == 1:
            msg = f"Permanently delete:\n{names[0]}?"
        else:
            preview = "\n".join(names[:8])
            more = f"\n…and {len(names) - 8} more" if len(names) > 8 else ""
            msg = f"Permanently delete {len(names)} files?\n\n{preview}{more}"

        if not messagebox.askyesno("Delete files", msg, icon=messagebox.WARNING):
            return

        errors: list[str] = []
        for i in indices:
            if i >= len(self._source_paths):
                continue
            try:
                self._source_paths[i].unlink()
            except OSError as exc:
                errors.append(str(exc))

        if errors:
            messagebox.showerror("Some deletions failed", "\n".join(errors))

        self._selected.clear()
        self._anchor = None
        self._delete_btn.configure(state=tk.DISABLED)
        self._source_canvas.set_placeholder("Select a source image to preview")

        if self._stacked_path and self._source_folder and self._root_dir:
            self.load_for(self._stacked_path, self._source_folder, self._root_dir)

    def _rerun_focusweave(self) -> None:
        if self._source_folder is None or self._stacked_path is None:
            return
        cmd = ["focusweave", str(self._source_folder), "--output", str(self._stacked_path)]
        self._status.configure(text=f"Running: {' '.join(cmd)}")
        self.update_idletasks()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            messagebox.showerror("focusweave not found",
                                 "The 'focusweave' command was not found on your PATH.")
            self._status.configure(text="focusweave not found")
            return
        if result.returncode == 0:
            self._load_stacked_preview()
            if self._on_thumb_reloaded:
                self._on_thumb_reloaded(self._stacked_path)
            messagebox.showinfo("Done", f"focusweave completed for:\n{self._stacked_path.name}")
        else:
            messagebox.showerror("focusweave failed",
                                 f"Exit code {result.returncode}\n\n{result.stderr[:1000]}")
        self._status.configure(text=f"Done — {self._stacked_path.name}")


class StackedListPanel(tk.Frame):
    """Left panel: virtually-scrolled thumbnail grid of focus-stacked images."""

    def __init__(self, parent: tk.Widget, on_select: object, **kwargs: object) -> None:
        super().__init__(parent, **kwargs)
        self._on_select_cb = on_select
        self._root_dir: Path | None = None
        self._entries: list[dict] = []
        self._build()

    def _build(self) -> None:
        top = tk.Frame(self, bg=BG_MID)
        top.pack(fill=tk.X)

        tk.Label(top, text="Focus stacked", font=("Courier", 11, "bold"),
                 bg=BG_MID, fg="#888888", anchor="w").pack(side=tk.LEFT, padx=8, pady=6)

        tk.Button(top, text="Open folder", font=("Courier", 10),
                  bg="#1e2a3a", fg="#88aacc", relief=tk.FLAT, padx=10, pady=4,
                  cursor="hand2", command=self._pick_folder).pack(side=tk.RIGHT, padx=8, pady=4)

        self._folder_label = tk.Label(self, text="No folder selected", font=("Courier", 8),
                                      bg=BG_MID, fg="#444444", anchor="w", wraplength=340)
        self._folder_label.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._grid = VirtualThumbGrid(self, cols=LIST_COLS, cell_h=CELL_H,
                                      on_select=self._on_grid_select, bg=BG_DARK)
        self._grid.pack(fill=tk.BOTH, expand=True)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(title="Select root folder")
        if not path:
            return
        root = Path(path)
        stacked = root / "focus_stacked"
        if not stacked.is_dir():
            messagebox.showerror("Invalid folder",
                                 f"No 'focus_stacked' subfolder found in:\n{root}")
            return
        self._root_dir = root
        self._load_list()

    def _load_list(self) -> None:
        if self._root_dir is None:
            return
        stacked_dir = self._root_dir / "focus_stacked"
        paths = sorted(f for f in stacked_dir.iterdir()
                       if f.suffix.lower() in SUPPORTED_EXTS)

        self._folder_label.configure(text=f"{self._root_dir}  ({len(paths)} images)")

        entries = []
        for f in paths:
            source = find_source_folder(self._root_dir, f.name)
            entries.append({
                "path": f,
                "source_folder": source,
                "label_text": f.name + ("  [!]" if source is None else ""),
                "label_fg": "#cc8866" if source is None else "#888888",
            })
        self._entries = entries
        self._grid.load(entries)

    def _on_grid_select(self, idx: int, entry: dict) -> None:
        self._on_select_cb(entry["path"], entry["source_folder"], self._root_dir)

    def reload_thumb(self, path: Path) -> None:
        self._grid.reload_thumb(path)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Focus Restack")
        self.configure(bg=BG_MID)
        self.geometry("1600x1000")
        self.minsize(1000, 600)
        self._build()

    def _build(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Vertical.TScrollbar", background="#2a2a2a",
                        troughcolor=BG_DARK, bordercolor=BG_MID, arrowcolor="#555555")

        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#0d0d0d", sashwidth=4)
        pane.pack(fill=tk.BOTH, expand=True)

        self._list_panel = StackedListPanel(pane, on_select=self._on_image_selected, bg=BG_MID)
        pane.add(self._list_panel, minsize=300, width=480)

        self._source_panel = SourceImagePanel(pane, bg=BG_MID)
        self._source_panel.set_thumb_reload_callback(self._list_panel.reload_thumb)
        pane.add(self._source_panel, minsize=600)

    def _on_image_selected(self, stacked_path: Path, source_folder: Path | None,
                           root_dir: Path) -> None:
        if source_folder is None:
            messagebox.showwarning("Source folder not found",
                                   f"No matching source folder found for:\n{stacked_path.name}\n\n"
                                   "Expected: x…_y… folder in root directory.")
            return
        self._source_panel.load_for(stacked_path, source_folder, root_dir)


if __name__ == "__main__":
    app = App()
    app.mainloop()